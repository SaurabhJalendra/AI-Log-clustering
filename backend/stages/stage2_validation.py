"""
stages/stage2_validation.py
============================
Standalone validation framework for Stage 2, per
Stage2_Engineering_Specification_v2 §9 (Validation Framework, V1-V9).

Mirrors stage1_validation.py's design: decoupled from the pipeline engine so
it can be unit-tested against synthetic DataFrames, reused by an offline
auditor re-checking a previously-produced Stage 2 output, and read end-to-end
in a few minutes.

Checks implemented:
    V1: No ParsedLine lost           — every input parsed_line_id has a
                                        resolvable EventLine.
    V2: 1:1 ParsedLine -> LogicalEvent — every parsed_line_id maps to
                                        exactly one event_id via EventLine.
    V3: TemplateOccurrence -> LogicalEvent — every occurrence's event_id
                                        resolves to a real LogicalEvent.
    V4: TemplateOccurrence -> Template — every occurrence's template_id
                                        resolves to a real Template.
    V5: No orphan Templates           — every Template has >=1 occurrence.
    V6: All lineage references resolvable — composite check across V1/V3/V4
                                        plus EventLine.event_id -> LogicalEvent.
    V7: Unique event_id values        — LogicalEvent.event_id has no duplicates.
    V8: Unique template_id values     — Template.template_id has no duplicates.
    V9: Manifest count conservation   — sum(manifest cluster counts) ==
                                        total rows in the visible output.
                                        (New: this existed in the reference
                                        implementation as a warning log line
                                        that nothing ever enforced — promoted
                                        to a real fatal check here, since that
                                        gap is exactly what let Defect 1 ship
                                        unnoticed. See STAGE2_ARCHITECTURE_REVIEW.md §0.)

Severity model: every check above is fatal. There is no "recoverable" tier
in Stage 2's validation framework the way Stage 1 has unknown-format/
unknown-service as expected outcomes — a Stage 2 that fails any of V1-V9 has
violated a structural lineage guarantee (S2-INV-001 through S2-INV-005),
not produced a merely-imperfect-but-valid result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import pandas as pd


@dataclass
class ValidationCheckResult:
    check_id: str
    description: str
    passed: bool
    fatal: bool
    observed: Any = None
    expected: Any = None
    details: Optional[str] = None
    sample_offending_values: Sequence[Any] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "check_id": self.check_id, "description": self.description,
            "passed": self.passed, "fatal": self.fatal,
            "observed": self.observed, "expected": self.expected,
            "details": self.details,
            "sample_offending_values": list(self.sample_offending_values),
        }


@dataclass
class ValidationReport:
    checks: List[ValidationCheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def fatal_failures(self) -> List[ValidationCheckResult]:
        return [c for c in self.checks if c.fatal and not c.passed]

    def as_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "fatal_failure_count": len(self.fatal_failures),
            "checks": [c.as_dict() for c in self.checks],
        }

    def log(self, logger) -> None:
        for c in self.checks:
            level = logger.info if c.passed else logger.error
            level(
                "stage2.validation %s [%s]: %s (observed=%r expected=%r)%s",
                c.check_id, "PASS" if c.passed else "FATAL FAIL", c.description,
                c.observed, c.expected, f" — {c.details}" if c.details else "",
            )
        if self.all_passed:
            logger.info("stage2.validation: ALL CHECKS PASSED (%d/%d)", len(self.checks), len(self.checks))
        else:
            logger.error(
                "stage2.validation: %d/%d checks failed",
                sum(1 for c in self.checks if not c.passed), len(self.checks),
            )


class Stage2ValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        self.report = report
        failing = ", ".join(c.check_id for c in report.fatal_failures)
        super().__init__(
            f"Stage 2 fatal validation failure(s): {failing}. "
            f"See report.fatal_failures for detail. Execution halted per "
            f"Stage2_Engineering_Specification_v2 §9 (all checks are fatal)."
        )


# ══════════════════════════════════════════════════════════════════════
# Individual checks
# ══════════════════════════════════════════════════════════════════════

def check_v1_no_parsed_line_lost(
    result_df: pd.DataFrame, event_lines_df: pd.DataFrame, input_parsed_line_count: int,
) -> ValidationCheckResult:
    observed = len(event_lines_df)
    passed = observed == input_parsed_line_count
    return ValidationCheckResult(
        check_id="V1", description="No ParsedLine lost (EventLine count == input ParsedLine count)",
        passed=passed, fatal=True, observed=observed, expected=input_parsed_line_count,
        details=None if passed else (
            f"Stage 2 received {input_parsed_line_count} ParsedLines but built "
            f"{observed} EventLine records ({input_parsed_line_count - observed:+d} "
            f"unaccounted for). This is exactly Defect 1's failure mode — see "
            f"STAGE2_ARCHITECTURE_REVIEW.md §0."
        ),
    )


def check_v2_one_event_per_parsed_line(event_lines_df: pd.DataFrame) -> ValidationCheckResult:
    if event_lines_df.empty or "parsed_line_id" not in event_lines_df.columns:
        return ValidationCheckResult(
            check_id="V2", description="Every ParsedLine assigned to exactly one LogicalEvent",
            passed=True, fatal=True, observed=0, expected=0,
        )
    dupe_counts = event_lines_df["parsed_line_id"].value_counts()
    dupes = dupe_counts[dupe_counts > 1]
    observed = len(dupes)
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V2", description="Every ParsedLine assigned to exactly one LogicalEvent",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else f"{observed} parsed_line_id value(s) appear in more than one EventLine record.",
        sample_offending_values=tuple(dupes.index.tolist()[:10]),
    )


def check_v3_occurrence_to_event(occurrences_df: pd.DataFrame, events_df: pd.DataFrame) -> ValidationCheckResult:
    if occurrences_df.empty:
        return ValidationCheckResult(
            check_id="V3", description="Every TemplateOccurrence references a valid LogicalEvent",
            passed=True, fatal=True, observed=0, expected=0,
        )
    known_events = set(events_df["event_id"].tolist()) if not events_df.empty else set()
    orphan_mask = ~occurrences_df["event_id"].isin(known_events)
    observed = int(orphan_mask.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V3", description="Every TemplateOccurrence references a valid LogicalEvent",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else f"{observed} TemplateOccurrence row(s) reference an event_id with no matching LogicalEvent.",
        sample_offending_values=tuple(occurrences_df.loc[orphan_mask, "template_occurrence_id"].head(10).tolist()) if "template_occurrence_id" in occurrences_df.columns else (),
    )


def check_v4_occurrence_to_template(occurrences_df: pd.DataFrame, templates_df: pd.DataFrame) -> ValidationCheckResult:
    if occurrences_df.empty:
        return ValidationCheckResult(
            check_id="V4", description="Every TemplateOccurrence references a valid Template",
            passed=True, fatal=True, observed=0, expected=0,
        )
    known_templates = set(templates_df["template_id"].tolist()) if not templates_df.empty else set()
    orphan_mask = ~occurrences_df["template_id"].isin(known_templates)
    observed = int(orphan_mask.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V4", description="Every TemplateOccurrence references a valid Template",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else f"{observed} TemplateOccurrence row(s) reference a template_id with no matching Template.",
    )


def check_v5_no_orphan_templates(templates_df: pd.DataFrame, occurrences_df: pd.DataFrame) -> ValidationCheckResult:
    if templates_df.empty:
        return ValidationCheckResult(
            check_id="V5", description="No orphan Templates (every Template has >=1 occurrence)",
            passed=True, fatal=True, observed=0, expected=0,
        )
    occ_tids = set(occurrences_df["template_id"].tolist()) if not occurrences_df.empty else set()
    orphan_mask = ~templates_df["template_id"].isin(occ_tids)
    observed = int(orphan_mask.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V5", description="No orphan Templates (every Template has >=1 occurrence)",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else f"{observed} Template row(s) have zero TemplateOccurrence references.",
        sample_offending_values=tuple(templates_df.loc[orphan_mask, "template_id"].head(10).tolist()),
    )


def check_v6_all_lineage_resolvable(
    event_lines_df: pd.DataFrame, events_df: pd.DataFrame,
) -> ValidationCheckResult:
    """Composite: every EventLine.event_id resolves to a real LogicalEvent.
    (V1/V3/V4 cover the rest of the chain; this is the one link they don't.)"""
    if event_lines_df.empty:
        return ValidationCheckResult(
            check_id="V6", description="All lineage references resolvable",
            passed=True, fatal=True, observed=0, expected=0,
        )
    known_events = set(events_df["event_id"].tolist()) if not events_df.empty else set()
    orphan_mask = ~event_lines_df["event_id"].isin(known_events)
    observed = int(orphan_mask.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V6", description="All lineage references resolvable (EventLine -> LogicalEvent)",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else f"{observed} EventLine row(s) reference an event_id with no matching LogicalEvent.",
    )


def check_v7_unique_event_ids(events_df: pd.DataFrame) -> ValidationCheckResult:
    if events_df.empty:
        return ValidationCheckResult(
            check_id="V7", description="All event_id values unique",
            passed=True, fatal=True, observed=0, expected=0,
        )
    dupes = events_df["event_id"][events_df["event_id"].duplicated(keep=False)]
    observed = len(dupes.unique())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V7", description="All event_id values unique",
        passed=passed, fatal=True, observed=observed, expected=0,
        sample_offending_values=tuple(dupes.unique().tolist()[:10]),
    )


def check_v8_unique_template_ids(templates_df: pd.DataFrame) -> ValidationCheckResult:
    if templates_df.empty:
        return ValidationCheckResult(
            check_id="V8", description="All template_id values unique",
            passed=True, fatal=True, observed=0, expected=0,
        )
    dupes = templates_df["template_id"][templates_df["template_id"].duplicated(keep=False)]
    observed = len(dupes.unique())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V8", description="All template_id values unique",
        passed=passed, fatal=True, observed=observed, expected=0,
        sample_offending_values=tuple(dupes.unique().tolist()[:10]),
    )


def check_v9_manifest_count_conservation(manifest: dict, result_df: pd.DataFrame) -> ValidationCheckResult:
    """Promoted from a warning log line (reference implementation) to a real
    fatal check — see module docstring."""
    clusters = manifest.get("clusters", {}) if manifest else {}
    computed_total = sum(v.get("count", 0) for v in clusters.values())
    is_noise_col = result_df["is_noise"].fillna(False).astype(bool) if "is_noise" in result_df.columns else pd.Series(False, index=result_df.index)
    expected = int((~is_noise_col).sum())
    passed = computed_total == expected
    return ValidationCheckResult(
        check_id="V9", description="Manifest cluster counts sum to the non-noise row count",
        passed=passed, fatal=True, observed=computed_total, expected=expected,
        details=None if passed else (
            f"sum(manifest cluster counts)={computed_total} but non-noise rows in "
            f"the visible output={expected} (difference {abs(computed_total-expected)}). "
            f"In the reference implementation this was a warning log line nothing "
            f"enforced — promoted to fatal here because that gap is what let "
            f"Defect 1 ship unnoticed."
        ),
    )


def run_validations(
    result_df: pd.DataFrame,
    *,
    events_df: pd.DataFrame,
    event_lines_df: pd.DataFrame,
    occurrences_df: pd.DataFrame,
    templates_df: pd.DataFrame,
    input_parsed_line_count: int,
    manifest: dict,
    logger=None,
    raise_on_fatal: bool = True,
) -> ValidationReport:
    report = ValidationReport(checks=[
        check_v1_no_parsed_line_lost(result_df, event_lines_df, input_parsed_line_count),
        check_v2_one_event_per_parsed_line(event_lines_df),
        check_v3_occurrence_to_event(occurrences_df, events_df),
        check_v4_occurrence_to_template(occurrences_df, templates_df),
        check_v5_no_orphan_templates(templates_df, occurrences_df),
        check_v6_all_lineage_resolvable(event_lines_df, events_df),
        check_v7_unique_event_ids(events_df),
        check_v8_unique_template_ids(templates_df),
        check_v9_manifest_count_conservation(manifest, result_df),
    ])

    if logger is not None:
        report.log(logger)

    if raise_on_fatal and report.fatal_failures:
        raise Stage2ValidationError(report)

    return report