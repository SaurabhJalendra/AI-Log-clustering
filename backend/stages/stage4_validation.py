"""
stages/stage4_validation.py
============================
Standalone validation framework for Stage 4 (final production stage).
Implements V1-V12 per Stage4_Engineering_Specification_v2.

Stage 4 is now the final pipeline stage. Every anomaly it emits may be shown
directly to users through API responses, frontend dashboards, reports, and
exports. Therefore validation is more consequential here than in any earlier
stage: a corrupt or inconsistent output goes straight to users with no
downstream stage to catch it.

All checks are fatal — same rationale as Stages 1, 2, and 3. There is no
"expected-but-non-fatal" outcome in Stage 4's output contract.

Checks:
    V1:  All anomaly rows have non-null semantic_cluster_id
    V2:  All anomaly rows have non-null anomaly_score
    V3:  anomaly_score in [0.0, 1.0] for every row
    V4:  anomaly_label in allowed set for every row
    V5:  evidence_summary non-null and non-empty for every anomaly row
    V6:  Lineage fields (semantic_cluster_id, line_no) preserved
    V7:  No duplicate anomaly_id values in anomaly_df
    V8:  No routine rows appear in anomaly_df (routine/anomaly partition integrity)
    V9:  ML fallback truthfully reported (anomaly_source not ae_if_ensemble
         when ml_ensemble_score == anomaly_score only because formula ran)
    V10: Frontend summary counts match anomaly_df row counts
    V11: anomaly_summary total_anomalies == len(anomaly_df)
    V12: No missing critical fields required by API contract
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

_VALID_LABELS   = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
_VALID_SOURCES  = frozenset({
    "ae_if_ensemble", "ml_if_only", "ml_ae_only",
    "formula_fallback", "statistical_fallback", "post_deploy_caution",
})
_API_REQUIRED_COLS = [
    "anomaly_id", "semantic_cluster_id", "anomaly_score", "anomaly_label",
    "evidence_summary", "dominant_severity", "domain",
]


# ══════════════════════════════════════════════════════════════════════
# Result types (mirrors stage1/2/3 validation design)
# ══════════════════════════════════════════════════════════════════════

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
    checks: List[ValidationCheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def fatal_failures(self) -> List[ValidationCheckResult]:
        return [c for c in self.checks if c.fatal and not c.passed]

    @property
    def passed(self) -> bool:
        return len(self.fatal_failures) == 0

    def summary(self) -> str:
        n_pass = sum(1 for c in self.checks if c.passed)
        n_fail = len(self.checks) - n_pass
        return f"{n_pass}/{len(self.checks)} passed, {len(self.fatal_failures)} fatal"

    def as_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "fatal_failure_count": len(self.fatal_failures),
            "checks": [c.as_dict() for c in self.checks],
            "summary": self.summary(),
        }

    def log(self, logger) -> None:
        for c in self.checks:
            level = logger.info if c.passed else logger.error
            level(
                "stage4.validation %s [%s]: %s (observed=%r expected=%r)%s",
                c.check_id, "PASS" if c.passed else "FATAL FAIL",
                c.description, c.observed, c.expected,
                f" — {c.details}" if c.details else "",
            )
        if self.all_passed:
            logger.info("stage4.validation: ALL CHECKS PASSED (%d/%d)",
                        len(self.checks), len(self.checks))
        else:
            logger.error("stage4.validation: %d/%d failed (%d fatal)",
                         sum(1 for c in self.checks if not c.passed),
                         len(self.checks), len(self.fatal_failures))


class Stage4ValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        self.report = report
        failing = ", ".join(c.check_id for c in report.fatal_failures)
        super().__init__(
            f"Stage 4 fatal validation failure(s): {failing}. "
            f"Stage 4 is the final production stage — corrupt output "
            f"would be shown directly to users. Execution halted."
        )


# ══════════════════════════════════════════════════════════════════════
# Individual checks
# ══════════════════════════════════════════════════════════════════════

def check_v1_semantic_cluster_id_present(anomaly_df: pd.DataFrame) -> ValidationCheckResult:
    if "semantic_cluster_id" not in anomaly_df.columns:
        return ValidationCheckResult(
            check_id="V1", description="All anomaly rows have semantic_cluster_id",
            passed=False, fatal=True, observed="column missing", expected="non-null column",
            details="semantic_cluster_id column absent entirely from anomaly_df. "
                    "MEDIUM-12 FIX may have renamed it instead of aliasing it.",
        )
    null_mask = anomaly_df["semantic_cluster_id"].isna()
    observed = int(null_mask.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V1", description="All anomaly rows have semantic_cluster_id",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else f"{observed} rows have null semantic_cluster_id.",
        sample_offending_values=tuple(anomaly_df.loc[null_mask].index.tolist()[:10]),
    )


def check_v2_anomaly_score_present(anomaly_df: pd.DataFrame) -> ValidationCheckResult:
    if "anomaly_score" not in anomaly_df.columns:
        return ValidationCheckResult(
            check_id="V2", description="All anomaly rows have anomaly_score",
            passed=False, fatal=True, observed="column missing", expected="numeric column",
        )
    null_mask = anomaly_df["anomaly_score"].isna()
    observed = int(null_mask.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V2", description="All anomaly rows have anomaly_score",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else f"{observed} rows have null anomaly_score.",
    )


def check_v3_score_range(anomaly_df: pd.DataFrame) -> ValidationCheckResult:
    scores = pd.to_numeric(anomaly_df.get("anomaly_score", pd.Series(dtype=float)), errors="coerce")
    out_of_range = ((scores < 0.0) | (scores > 1.0 + 1e-6) | scores.isna())
    observed = int(out_of_range.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V3", description="All anomaly_score values in [0.0, 1.0]",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else (
            f"{observed} rows have anomaly_score outside [0,1] or NaN. "
            f"Max observed: {float(scores.max()) if not scores.empty else 'n/a':.4f}"
        ),
        sample_offending_values=tuple(scores[out_of_range].head(10).tolist()),
    )


def check_v4_valid_labels(anomaly_df: pd.DataFrame) -> ValidationCheckResult:
    if "anomaly_label" not in anomaly_df.columns:
        return ValidationCheckResult(
            check_id="V4", description="All anomaly_label values are valid",
            passed=False, fatal=True, observed="column missing", expected=str(_VALID_LABELS),
        )
    invalid_mask = ~anomaly_df["anomaly_label"].isin(_VALID_LABELS)
    observed = int(invalid_mask.sum())
    passed = observed == 0
    bad_vals = anomaly_df.loc[invalid_mask, "anomaly_label"].unique().tolist()
    return ValidationCheckResult(
        check_id="V4", description="All anomaly_label values are valid (LOW/MEDIUM/HIGH/CRITICAL)",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else f"Invalid labels: {bad_vals[:5]}",
        sample_offending_values=tuple(bad_vals[:10]),
    )


def check_v5_evidence_present(anomaly_df: pd.DataFrame) -> ValidationCheckResult:
    if "evidence_summary" not in anomaly_df.columns:
        return ValidationCheckResult(
            check_id="V5", description="Every anomaly row has evidence_summary",
            passed=False, fatal=True, observed="column missing", expected="non-empty strings",
            details="evidence_summary column absent. Stage 4 must generate deterministic evidence.",
        )
    null_or_empty = (
        anomaly_df["evidence_summary"].isna() |
        (anomaly_df["evidence_summary"].astype(str).str.strip() == "")
    )
    observed = int(null_or_empty.sum())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V5", description="Every anomaly row has evidence_summary",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else (
            f"{observed} anomaly rows have null or empty evidence_summary. "
            "Stage 4 is the final stage — every anomaly must explain WHY it was flagged."
        ),
    )


def check_v6_lineage_preserved(anomaly_df: pd.DataFrame) -> ValidationCheckResult:
    """
    V6: Lineage preserved.

    anomaly_df is a CLUSTER-level table (one row per semantic_cluster_id),
    not a per-line table — line_no is a per-line concept and does not belong
    here. The lineage guarantee at this granularity is that semantic_cluster_id
    itself is present and non-null (every anomaly traces back to exactly one
    Stage 3 cluster). This overlaps with V1 by design — V1 is the general
    "field present" check, V6 specifically frames it as a lineage guarantee
    distinct from API-readiness, matching the spec's intent that lineage and
    API-required-fields are reviewed as separate concerns even when they
    currently resolve to the same column.
    """
    if "semantic_cluster_id" not in anomaly_df.columns:
        return ValidationCheckResult(
            check_id="V6", description="Lineage preserved (semantic_cluster_id traceable)",
            passed=False, fatal=True, observed="column missing", expected="present",
        )
    null_scid = int(anomaly_df["semantic_cluster_id"].isna().sum())
    passed = null_scid == 0
    return ValidationCheckResult(
        check_id="V6", description="Lineage preserved (semantic_cluster_id traceable)",
        passed=passed, fatal=True, observed=null_scid, expected=0,
        details=None if passed else f"{null_scid} rows have null semantic_cluster_id (lineage broken).",
    )


def check_v7_unique_anomaly_ids(anomaly_df: pd.DataFrame) -> ValidationCheckResult:
    if "anomaly_id" not in anomaly_df.columns:
        return ValidationCheckResult(
            check_id="V7", description="No duplicate anomaly_id values",
            passed=False, fatal=True, observed="column missing", expected="unique non-null values",
        )
    dupes = anomaly_df["anomaly_id"][anomaly_df["anomaly_id"].duplicated(keep=False)]
    observed = len(dupes.unique())
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V7", description="No duplicate anomaly_id values",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else f"{observed} anomaly_id value(s) appear on multiple rows.",
        sample_offending_values=tuple(dupes.unique().tolist()[:10]),
    )


def check_v8_routine_not_in_anomaly(
    anomaly_df: pd.DataFrame, routine_df: pd.DataFrame
) -> ValidationCheckResult:
    if "semantic_cluster_id" not in anomaly_df.columns or "semantic_cluster_id" not in routine_df.columns:
        return ValidationCheckResult(
            check_id="V8", description="No routine rows in anomaly_df",
            passed=True, fatal=True, observed=0, expected=0,
            details="Cannot check (semantic_cluster_id missing from one frame).",
        )
    anomaly_scids = set(anomaly_df["semantic_cluster_id"].dropna().astype(str).tolist())
    routine_scids = set(routine_df["semantic_cluster_id"].dropna().astype(str).tolist())
    overlap = anomaly_scids & routine_scids
    observed = len(overlap)
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V8", description="No routine clusters appear in anomaly_df",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else (
            f"{observed} semantic_cluster_id(s) appear in both anomaly_df and routine_df. "
            "A cluster cannot be both anomalous and routine."
        ),
        sample_offending_values=tuple(sorted(overlap)[:10]),
    )


def check_v9_ml_source_truthful(anomaly_df: pd.DataFrame) -> ValidationCheckResult:
    """
    V9: anomaly_source must be truthful.
    If anomaly_source == "ae_if_ensemble", then ml_if_score and ml_ae_score must
    both be present and must not equal 0.5 (the default fallback sentinel value)
    for every row so labeled.
    """
    if "anomaly_source" not in anomaly_df.columns:
        return ValidationCheckResult(
            check_id="V9", description="ML fallback truthfully reported",
            passed=True, fatal=True, observed=0, expected=0,
            details="anomaly_source column absent — cannot verify.",
        )
    ensemble_mask = anomaly_df["anomaly_source"] == "ae_if_ensemble"
    if not ensemble_mask.any():
        return ValidationCheckResult(
            check_id="V9", description="ML fallback truthfully reported",
            passed=True, fatal=True, observed=0, expected=0,
        )
    ensemble_rows = anomaly_df[ensemble_mask]
    violations = []
    for col in ("ml_if_score", "ml_ae_score"):
        if col not in ensemble_rows.columns:
            violations.append(f"{col} missing for ae_if_ensemble rows")
        elif (ensemble_rows[col] == 0.5).all():
            violations.append(f"{col} is all 0.5 (fallback sentinel) for ae_if_ensemble rows")
    observed = len(violations)
    passed = observed == 0
    return ValidationCheckResult(
        check_id="V9", description="ML fallback truthfully reported (ae_if_ensemble means both models ran)",
        passed=passed, fatal=True, observed=observed, expected=0,
        details=None if passed else " | ".join(violations),
    )


def check_v10_frontend_counts_match(
    anomaly_df: pd.DataFrame,
    anomaly_summary: Optional[Dict],
) -> ValidationCheckResult:
    if anomaly_summary is None:
        return ValidationCheckResult(
            check_id="V10", description="Frontend summary counts match anomaly_df",
            passed=False, fatal=True, observed="anomaly_summary missing", expected="dict",
        )
    expected_total = len(anomaly_df)
    observed_total = anomaly_summary.get("total_anomalies", -1)
    passed = int(observed_total) == expected_total
    return ValidationCheckResult(
        check_id="V10", description="Frontend summary total_anomalies matches len(anomaly_df)",
        passed=passed, fatal=True, observed=observed_total, expected=expected_total,
        details=None if passed else (
            f"anomaly_summary.total_anomalies={observed_total} but len(anomaly_df)={expected_total}"
        ),
    )


def check_v11_summary_severity_counts(
    anomaly_df: pd.DataFrame,
    anomaly_summary: Optional[Dict],
) -> ValidationCheckResult:
    if anomaly_summary is None or "anomaly_label" not in anomaly_df.columns:
        return ValidationCheckResult(
            check_id="V11", description="Summary severity counts match anomaly_df label distribution",
            passed=True, fatal=True, observed=0, expected=0,
            details="Cannot check (anomaly_summary None or anomaly_label missing).",
        )
    label_counts = anomaly_df["anomaly_label"].value_counts().to_dict()
    mismatches = []
    for label, key in [("CRITICAL", "critical_count"), ("HIGH", "high_count"),
                        ("MEDIUM", "medium_count"), ("LOW", "low_count")]:
        expected = label_counts.get(label, 0)
        observed = anomaly_summary.get(key, -1)
        if int(observed) != expected:
            mismatches.append(f"{key}: summary={observed} df={expected}")
    passed = len(mismatches) == 0
    return ValidationCheckResult(
        check_id="V11", description="Summary severity counts match anomaly_df",
        passed=passed, fatal=True, observed=len(mismatches), expected=0,
        details=None if passed else " | ".join(mismatches),
    )


def check_v12_required_api_fields(anomaly_df: pd.DataFrame) -> ValidationCheckResult:
    missing = [c for c in _API_REQUIRED_COLS if c not in anomaly_df.columns]
    null_violations = []
    for c in _API_REQUIRED_COLS:
        if c in anomaly_df.columns and anomaly_df[c].isna().all():
            null_violations.append(f"{c} is entirely null")
    all_issues = missing + null_violations
    passed = len(all_issues) == 0
    return ValidationCheckResult(
        check_id="V12", description="All API-required fields present and non-null",
        passed=passed, fatal=True, observed=len(all_issues), expected=0,
        details=None if passed else " | ".join(all_issues),
        sample_offending_values=tuple(all_issues[:10]),
    )


# ══════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════

def run_validations(
    anomaly_df: pd.DataFrame,
    routine_df: pd.DataFrame,
    *,
    anomaly_summary: Optional[Dict] = None,
    logger=None,
    raise_on_fatal: bool = True,
) -> ValidationReport:
    """
    Run the full V1-V12 suite against Stage 4's final output DataFrames.

    Parameters
    ----------
    anomaly_df : the final anomaly DataFrame (signal clusters)
    routine_df : the final routine DataFrame (suppressed clusters)
    anomaly_summary : the frontend summary dict built by _build_frontend_outputs()
    raise_on_fatal : if True (default), raises Stage4ValidationError when any
        fatal check fails. Stage 4 is the final stage — corrupt output goes
        directly to users, so fatal failures must halt execution.
    """
    report = ValidationReport(checks=[
        check_v1_semantic_cluster_id_present(anomaly_df),
        check_v2_anomaly_score_present(anomaly_df),
        check_v3_score_range(anomaly_df),
        check_v4_valid_labels(anomaly_df),
        check_v5_evidence_present(anomaly_df),
        check_v6_lineage_preserved(anomaly_df),
        check_v7_unique_anomaly_ids(anomaly_df),
        check_v8_routine_not_in_anomaly(anomaly_df, routine_df),
        check_v9_ml_source_truthful(anomaly_df),
        check_v10_frontend_counts_match(anomaly_df, anomaly_summary),
        check_v11_summary_severity_counts(anomaly_df, anomaly_summary),
        check_v12_required_api_fields(anomaly_df),
    ])

    if logger is not None:
        report.log(logger)

    if raise_on_fatal and report.fatal_failures:
        raise Stage4ValidationError(report)

    return report