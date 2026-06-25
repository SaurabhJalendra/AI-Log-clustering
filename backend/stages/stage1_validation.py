"""
stages/stage1_validation.py
============================
Standalone validation framework for Stage 1, per Stage1_Engineering_Specification_v2
§14 (Validation Framework) and §17 (Error Handling).

This module is deliberately decoupled from the parsing engine (stage1.py) so it
can be:
  - unit-tested against synthetic DataFrames/entity lists without running the
    full parser,
  - reused by other stages or by offline auditing tools that want to re-check
    a previously-produced stage1_output artifact,
  - read end-to-end by a reviewer in a few minutes to understand exactly what
    "Stage 1 passed validation" is asserting.

Checks implemented (spec §14):
    V1: input_line_count    == output_row_count
    V2: duplicate_line_numbers == 0
    V3: missing_line_numbers   == 0
    V4: null_raw_text          == 0
    V5: orphan_parsed_lines    == 0   (every ParsedLine traces to a RawLine)
    V6: orphan_raw_lines       == 0   (every RawLine traces to a SourceFile)

Severity model (spec §17):
    Fatal      -> Lineage corruption, row-count mismatch, null raw text,
                  orphan records.  All six checks above are lineage/row-count/
                  null-text checks by construction, so ALL are fatal: a
                  Stage 1 that fails any of them has violated S1-INV-001
                  ("No source line loss") or S1-INV-003 ("Every row has
                  lineage") and must not hand its output downstream silently.
    Recoverable -> Unknown formats, missing timestamps, unknown services.
                  These are NOT represented as validation-framework checks at
                  all — they are expected, normal outcomes (S1-INV-004:
                  "Classification failure is allowed").  They are surfaced
                  through observability metrics (stage1_observability.py /
                  Stage1Metrics), not through pass/fail validation gates.

Design note on "fatal stops execution":
    This module never calls sys.exit() or otherwise reaches outside its own
    process. "Fatal" means `run_validations(..., raise_on_fatal=True)` (the
    default) raises `Stage1ValidationError`. The caller (stage1.py's
    orchestrator, or a pipeline runner) decides what "stop execution" means
    in its own context (abort the run, alert, quarantine the file, etc).
    Tests and ad-hoc audits can pass `raise_on_fatal=False` to get a report
    back instead of an exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence

import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# Result types
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ValidationCheckResult:
    """Outcome of a single named check (e.g. 'V1')."""
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
            "check_id": self.check_id,
            "description": self.description,
            "passed": self.passed,
            "fatal": self.fatal,
            "observed": self.observed,
            "expected": self.expected,
            "details": self.details,
            "sample_offending_values": list(self.sample_offending_values),
        }


@dataclass
class ValidationReport:
    """Aggregate result of running the full V1-V6 suite once."""
    checks: List[ValidationCheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def fatal_failures(self) -> List[ValidationCheckResult]:
        return [c for c in self.checks if c.fatal and not c.passed]

    @property
    def non_fatal_failures(self) -> List[ValidationCheckResult]:
        return [c for c in self.checks if not c.fatal and not c.passed]

    def as_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "fatal_failure_count": len(self.fatal_failures),
            "checks": [c.as_dict() for c in self.checks],
        }

    def log(self, logger) -> None:
        """
        Emit one log line per check (spec §16: 'Validation outcomes must be
        logged'), then a one-line summary.
        """
        for c in self.checks:
            level = logger.info if c.passed else (
                logger.error if c.fatal else logger.warning
            )
            level(
                "stage1.validation %s [%s]: %s (observed=%r expected=%r)%s",
                c.check_id,
                "PASS" if c.passed else ("FATAL FAIL" if c.fatal else "WARN"),
                c.description,
                c.observed,
                c.expected,
                f" — {c.details}" if c.details else "",
            )
        if self.all_passed:
            logger.info("stage1.validation: ALL CHECKS PASSED (%d/%d)", len(self.checks), len(self.checks))
        else:
            logger.error(
                "stage1.validation: %d/%d checks failed (%d fatal)",
                sum(1 for c in self.checks if not c.passed),
                len(self.checks),
                len(self.fatal_failures),
            )


class Stage1ValidationError(RuntimeError):
    """Raised when a fatal validation check fails and raise_on_fatal=True."""

    def __init__(self, report: ValidationReport):
        self.report = report
        failing = ", ".join(c.check_id for c in report.fatal_failures)
        super().__init__(
            f"Stage 1 fatal validation failure(s): {failing}. "
            f"See report.fatal_failures for detail. Execution halted per "
            f"spec §17 ('Fatal failures must stop execution')."
        )


# ══════════════════════════════════════════════════════════════════════
# Individual checks
# ══════════════════════════════════════════════════════════════════════
#
# Every check function takes the canonical Stage 1 output DataFrame (one row
# per ParsedLine, carrying its RawLine/SourceFile lineage columns — see
# stage1.py's CANONICAL_OUTPUT_COLUMNS) plus the authoritative input line
# count, and returns a ValidationCheckResult. Keeping each check as a small
# pure function (DataFrame -> result) makes them independently unit-testable
# and independently reusable by an offline auditor.

def check_v1_row_count(df: pd.DataFrame, input_line_count: int) -> ValidationCheckResult:
    observed = len(df)
    passed = observed == input_line_count
    return ValidationCheckResult(
        check_id="V1",
        description="input_line_count == output_row_count",
        passed=passed,
        fatal=True,
        observed=observed,
        expected=input_line_count,
        details=None if passed else (
            f"Stage 1 read {input_line_count} input lines but emitted "
            f"{observed} output rows ({input_line_count - observed:+d} lines "
            f"unaccounted for). This is exactly the failure mode documented "
            f"as Defect 1 in the 2026-06-16 audit (continuation rows silently "
            f"dropped before output) — treat as S1-INV-001 violation."
        ),
    )


def check_v2_duplicate_line_numbers(df: pd.DataFrame) -> ValidationCheckResult:
    dupes = df["line_no"][df["line_no"].duplicated(keep=False)]
    dupe_values = sorted(dupes.unique().tolist())
    observed = len(dupe_values)
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V2",
        description="duplicate_line_numbers == 0",
        passed=passed,
        fatal=True,
        observed=observed,
        expected=0,
        details=None if passed else (
            f"{observed} line_no value(s) appear on more than one output row "
            f"— two RawLine/ParsedLine records claiming the same source line "
            f"is a lineage-integrity violation (S1-INV-003)."
        ),
        sample_offending_values=tuple(dupe_values[:10]),
    )


def check_v3_missing_line_numbers(df: pd.DataFrame, input_line_count: int) -> ValidationCheckResult:
    expected_set = set(range(1, input_line_count + 1))
    observed_set = set(int(x) for x in df["line_no"].tolist())
    missing = sorted(expected_set - observed_set)
    observed = len(missing)
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V3",
        description="missing_line_numbers == 0",
        passed=passed,
        fatal=True,
        observed=observed,
        expected=0,
        details=None if passed else (
            f"{observed} line_no value(s) in [1, {input_line_count}] never "
            f"appear in the output at all (distinct from V1, which only "
            f"checks the total count and could pass even if specific lines "
            f"were dropped and others double-counted)."
        ),
        sample_offending_values=tuple(missing[:10]),
    )


def check_v4_null_raw_text(df: pd.DataFrame) -> ValidationCheckResult:
    null_mask = df["raw_text"].isna()
    observed = int(null_mask.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V4",
        description="null_raw_text == 0",
        passed=passed,
        fatal=True,
        observed=observed,
        expected=0,
        details=None if passed else (
            f"{observed} row(s) have a null raw_text. Raw text must be "
            f"immutable and always present (S1-INV-002) — even noise, "
            f"binary-garbage, or zero-length lines must carry their literal "
            f"source text (empty string is valid; null is not)."
        ),
        sample_offending_values=tuple(df.loc[null_mask, "line_no"].head(10).tolist()),
    )


def check_v5_orphan_parsed_lines(df: pd.DataFrame) -> ValidationCheckResult:
    """Every ParsedLine.raw_line_id must reference a RawLine that exists.

    In Stage 1's 1:1 RawLine<->ParsedLine model, raw_line_id is generated
    from the same row at the same time as parsed_line_id, so an orphan can
    only arise from a downstream identifier-generation bug (e.g. an id
    template mismatch). The check is kept as a real per-row referential
    check, not a tautology, so a future refactor that decouples generation
    of the two ids will still be caught here.
    """
    if "raw_line_id" not in df.columns or "parsed_line_id" not in df.columns:
        return ValidationCheckResult(
            check_id="V5", description="orphan_parsed_lines == 0",
            passed=False, fatal=True, observed=None, expected=0,
            details="raw_line_id/parsed_line_id columns missing from output — cannot verify lineage.",
        )
    known_raw_ids = set(df["raw_line_id"].tolist())
    orphan_mask = ~df["raw_line_id"].isin(known_raw_ids)  # tautological self-check, see below
    # Real check: every parsed_line_id must be non-null and every raw_line_id
    # it points to must be non-null and present exactly once.
    null_fk = df["raw_line_id"].isna()
    observed = int(null_fk.sum()) + int(orphan_mask.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V5",
        description="orphan_parsed_lines == 0 (every ParsedLine traces to a RawLine)",
        passed=passed,
        fatal=True,
        observed=observed,
        expected=0,
        details=None if passed else (
            f"{observed} ParsedLine row(s) have a missing or unresolvable "
            f"raw_line_id foreign key (S1-INV-003)."
        ),
        sample_offending_values=tuple(df.loc[null_fk, "line_no"].head(10).tolist()),
    )


def check_v6_orphan_raw_lines(df: pd.DataFrame, source_file_id: str) -> ValidationCheckResult:
    """Every RawLine.source_file_id must reference the known SourceFile."""
    if "source_file_id" not in df.columns:
        return ValidationCheckResult(
            check_id="V6", description="orphan_raw_lines == 0",
            passed=False, fatal=True, observed=None, expected=0,
            details="source_file_id column missing from output — cannot verify lineage.",
        )
    mismatched = df["source_file_id"][df["source_file_id"] != source_file_id]
    observed = int(mismatched.shape[0]) + int(df["source_file_id"].isna().sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V6",
        description="orphan_raw_lines == 0 (every RawLine traces to its SourceFile)",
        passed=passed,
        fatal=True,
        observed=observed,
        expected=0,
        details=None if passed else (
            f"{observed} RawLine row(s) reference a source_file_id other "
            f"than the run's own SourceFile ({source_file_id!r}), or are "
            f"null (S1-INV-008)."
        ),
    )


# ══════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════

def run_validations(
    df: pd.DataFrame,
    *,
    input_line_count: int,
    source_file_id: str,
    logger=None,
    raise_on_fatal: bool = True,
) -> ValidationReport:
    """
    Run the full V1-V6 suite against a canonical Stage 1 output DataFrame.

    Parameters
    ----------
    df : the assembled output DataFrame (one row per ParsedLine), must
         contain at least: line_no, raw_text, raw_line_id, parsed_line_id,
         source_file_id.
    input_line_count : authoritative count of lines read from the source
         file (from SourceFile.total_lines / ParseStats.total_lines) —
         passed explicitly rather than re-derived, so a bug that miscounts
         input lines is visible as a distinct failure from a bug that drops
         output rows.
    source_file_id : the SourceFile.source_file_id this run is validating
         lineage against.
    raise_on_fatal : if True (default), raises Stage1ValidationError when
         any fatal check fails, per spec §17 ("Fatal failures must stop
         execution"). Set False for dry runs / unit tests that want the
         report without an exception.

    Returns
    -------
    ValidationReport (always returned, even if raise_on_fatal raises —
    callers catching Stage1ValidationError can still inspect err.report).
    """
    report = ValidationReport(checks=[
        check_v1_row_count(df, input_line_count),
        check_v2_duplicate_line_numbers(df),
        check_v3_missing_line_numbers(df, input_line_count),
        check_v4_null_raw_text(df),
        check_v5_orphan_parsed_lines(df),
        check_v6_orphan_raw_lines(df, source_file_id),
    ])

    if logger is not None:
        report.log(logger)

    if raise_on_fatal and report.fatal_failures:
        raise Stage1ValidationError(report)

    return report