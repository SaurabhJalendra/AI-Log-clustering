#!/usr/bin/env python3
"""
validate_pipeline.py
──────────────────────────────────────────────────────────────────────
Assertion runner + benchmark comparison for the log analysis pipeline.

Run from terminal after every pipeline change:

    python validate_pipeline.py --pipeline_output outputs/sample_logs_output.csv

    python validate_pipeline.py --pipeline_output outputs/test_logs_01_output.csv \
                                --ground_truth validation/master_ground_truth.csv

The validator auto-discovers the Stage 4 anomaly CSV that corresponds to
the pipeline output file.  Given --pipeline_output outputs/FOO_output.csv
it looks for                  outputs/FOO_stage4_anomaly.csv

If that file exists the Stage 4 assertions (A9–A17) are also run.
If it doesn't exist, the validator prints exactly what to save and where,
then continues with A1–A8 only.

Assertions A1–A8  — hard rules on the per-line pipeline output CSV.
Assertions A9–A17 — hard rules on the Stage 4 anomaly_df CSV.
Never delete an assertion.  Only add new ones (A18, A19, …).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# ASSERTION REGISTRY
# ══════════════════════════════════════════════════════════════════════
# Each assertion is a function (df) -> List[str] of failure messages.
# Empty list = PASS.  Non-empty = FAIL with detail lines.

_ASSERTIONS_S3:  Dict[str, callable] = {}   # A1–A8   (pipeline output)
_ASSERTIONS_S4:  Dict[str, callable] = {}   # A9–A17  (stage 4 anomaly_df)


def _register_s3(aid: str):
    def _decorator(fn):
        _ASSERTIONS_S3[aid] = fn
        return fn
    return _decorator


def _register_s4(aid: str):
    def _decorator(fn):
        _ASSERTIONS_S4[aid] = fn
        return fn
    return _decorator


# ══════════════════════════════════════════════════════════════════════
# STAGE 3 ASSERTIONS  (A1–A8) — operate on pipeline output CSV
# ══════════════════════════════════════════════════════════════════════

# ── A1: No blank domain ───────────────────────────────────────────────
@_register_s3("A1")
def assert_no_blank_domain(df: pd.DataFrame) -> List[str]:
    """No row may have an empty, null, or whitespace-only domain."""
    blank = df[df["domain"].isna() | (df["domain"].astype(str).str.strip() == "")]
    if blank.empty:
        return []
    sample = blank["line_no"].head(10).tolist()
    return [f"{len(blank)} rows have blank domain. First 10 line_no: {sample}"]


# ── A2: timestamp_parsed_ok rows must have parseable timestamps ────────
@_register_s3("A2")
def assert_timestamp_parsed_ok_has_value(df: pd.DataFrame) -> List[str]:
    """Rows with timestamp_parsed_ok=True must have a non-empty timestamp_parsed."""
    ok_rows = df[df["timestamp_parsed_ok"].fillna(False).astype(bool)]
    bad = ok_rows[
        ok_rows["timestamp_parsed"].isna()
        | (ok_rows["timestamp_parsed"].astype(str).str.strip() == "")
    ]
    if bad.empty:
        return []
    sample = bad["line_no"].head(10).tolist()
    return [
        f"{len(bad)} rows have timestamp_parsed_ok=True but empty timestamp_parsed. "
        f"Line numbers: {sample}"
    ]


# ── A3: No format_tag produces timestamps >2 h from corpus median ──────
@_register_s3("A3")
def assert_timestamp_no_format_drift(df: pd.DataFrame) -> List[str]:
    """No format_tag median timestamp should be >2 hours from the overall corpus median."""
    ts = pd.to_datetime(df["timestamp_parsed"], errors="coerce", utc=True)
    valid = ts.dropna()
    if valid.empty:
        return []
    corpus_median_ns = valid.astype("int64").median()
    corpus_median    = pd.Timestamp(int(corpus_median_ns), unit="ns", tz="UTC")
    two_hours_ns     = 2 * 3_600 * 1_000_000_000
    failures = []
    for tag in df["format_tag"].dropna().unique():
        tag_ts = ts[df["format_tag"] == tag].dropna()
        if tag_ts.empty:
            continue
        tag_median_ns = tag_ts.astype("int64").median()
        drift_ns      = abs(tag_median_ns - corpus_median_ns)
        if drift_ns > two_hours_ns:
            drift_h = drift_ns / 3_600 / 1_000_000_000
            failures.append(
                f"format_tag='{tag}' median is {drift_h:.1f} h from corpus median "
                f"({corpus_median.isoformat()}). Likely TZ offset bug."
            )
    return failures


# ── A4: Identical normalised_message → same template_id ────────────────
@_register_s3("A4")
def assert_normalized_message_unique_template(df: pd.DataFrame) -> List[str]:
    """Two rows with identical normalised_message must share the same template_id."""
    col = "normalised_message" if "normalised_message" in df.columns else "normalized_message"
    if col not in df.columns or "template_id" not in df.columns:
        return []
    check = df[
        ~df["is_noise"].fillna(False).astype(bool)
        & df[col].notna()
        & df["template_id"].notna()
    ][[col, "template_id", "line_no"]].copy()
    if check.empty:
        return []
    multi  = check.groupby(col)["template_id"].nunique()
    splits = multi[multi > 1]
    if splits.empty:
        return []
    failures = []
    for norm_msg, n_tids in splits.head(5).items():
        tids = check[check[col] == norm_msg]["template_id"].unique().tolist()
        failures.append(
            f"normalised_message '{str(norm_msg)[:60]}' maps to "
            f"{n_tids} template_ids: {tids}"
        )
    if len(splits) > 5:
        failures.append(f"… and {len(splits) - 5} more normalised_message splits.")
    return failures


# ── A5: attempt=N/M with N>M → impossible_attempt_count ───────────────
@_register_s3("A5")
def assert_impossible_attempt_count(df: pd.DataFrame) -> List[str]:
    """Every row where attempt=N/M and N>M must have singleton_class=impossible_attempt_count."""
    _re     = re.compile(r"\battempt[=:]\s*(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
    msg_col = "message" if "message" in df.columns else (
              "normalised_message" if "normalised_message" in df.columns else
              "normalized_message")
    failures = []
    for _, row in df.iterrows():
        text = str(row.get(msg_col, "") or "")
        for m in _re.finditer(text):
            n_val, d_val = int(m.group(1)), int(m.group(2))
            if n_val > d_val:
                sc = str(row.get("singleton_class", "") or "")
                if sc != "impossible_attempt_count":
                    failures.append(
                        f"line_no={row.get('line_no', '?')}  attempt={n_val}/{d_val}  "
                        f"singleton_class='{sc}' (expected impossible_attempt_count)  "
                        f"msg='{text[:80]}'"
                    )
    return failures


# ── A6: No missing line numbers ────────────────────────────────────────
@_register_s3("A6")
def assert_no_missing_line_numbers(df: pd.DataFrame) -> List[str]:
    """Output line_no values must be sequential with no gaps."""
    if "line_no" not in df.columns:
        return ["line_no column missing from output"]
    nums = df["line_no"].dropna().astype(int).sort_values().tolist()
    if not nums:
        return ["line_no column is entirely empty"]
    expected = list(range(nums[0], nums[-1] + 1))
    missing  = sorted(set(expected) - set(nums))
    if not missing:
        return []
    sample = missing[:20]
    return [f"{len(missing)} line numbers missing. First 20: {sample}"]


# ── A7: db-replica / deadlock / transactions table → domain=database ──
@_register_s3("A7")
def assert_db_messages_have_database_domain(df: pd.DataFrame) -> List[str]:
    """Messages with db-replica, deadlock, or transactions table must have domain=database."""
    msg_col = "message" if "message" in df.columns else (
              "normalised_message" if "normalised_message" in df.columns else
              "normalized_message")
    if msg_col not in df.columns:
        return []
    mask = df[msg_col].astype(str).str.contains(
        r"db-replica|deadlock|transactions table", case=False, na=False, regex=True
    )
    bad = df[mask & (df["domain"] != "database")]
    failures = []
    for _, row in bad.iterrows():
        failures.append(
            f"line_no={row.get('line_no', '?')}  domain='{row.get('domain', '')}'  "
            f"(expected database)  msg='{str(row.get(msg_col, ''))[:80]}'"
        )
    return failures


# ── A8: is_merged=True rows must have a valid merged_into pointer ──────
@_register_s3("A8")
def assert_merged_rows_have_merged_into(df: pd.DataFrame) -> List[str]:
    """
    Rows where is_merged=True must have a non-empty merged_into value
    pointing to the parent template_id that absorbed them.
    """
    if "is_merged" not in df.columns or "merged_into" not in df.columns:
        return ["is_merged or merged_into column missing from output"]

    def _is_merged_true(val) -> bool:
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return s in ("✓", "true", "1", "yes")

    def _is_empty(val) -> bool:
        if pd.isna(val):
            return True
        s = str(val).strip()
        return s in ("", "nan", "none", "na")

    # Every row marked is_merged=True must have a merged_into pointing to its parent.
    bad = df[
        df["is_merged"].apply(_is_merged_true)
        & df["merged_into"].apply(_is_empty)
    ]
    if bad.empty:
        return []
    sample = bad["line_no"].head(10).tolist()
    return [
        f"{len(bad)} rows have is_merged=True but no merged_into value. "
        f"Every merged row must point to its parent template. First 10 line_no: {sample}"
    ]


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 ASSERTIONS  (A9–A17) — operate on stage4 anomaly_df CSV
# ══════════════════════════════════════════════════════════════════════

# ── A9: Every row must have an anomaly_label ──────────────────────────
@_register_s4("A9")
def assert_anomaly_label_present(df: pd.DataFrame) -> List[str]:
    """Every row in anomaly_df must have a non-empty anomaly_label."""
    if "anomaly_label" not in df.columns:
        return ["anomaly_label column missing from Stage 4 output"]
    bad = df[df["anomaly_label"].isna() | (df["anomaly_label"].astype(str).str.strip() == "")]
    if bad.empty:
        return []
    return [f"{len(bad)} rows are missing anomaly_label"]


# ── A10: anomaly_label values must be within the allowed set ──────────
@_register_s4("A10")
def assert_anomaly_label_valid_values(df: pd.DataFrame) -> List[str]:
    """anomaly_label must be one of CRITICAL, HIGH, MEDIUM, LOW."""
    if "anomaly_label" not in df.columns:
        return []
    allowed = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    bad_vals = set(df["anomaly_label"].dropna().astype(str).unique()) - allowed
    if not bad_vals:
        return []
    return [f"Unexpected anomaly_label values: {bad_vals}"]


# ── A11: anomaly_score must be in [0, 1] ─────────────────────────────
@_register_s4("A11")
def assert_anomaly_score_range(df: pd.DataFrame) -> List[str]:
    """anomaly_score must be a float in [0.0, 1.0]."""
    if "anomaly_score" not in df.columns:
        return ["anomaly_score column missing from Stage 4 output"]
    scores = pd.to_numeric(df["anomaly_score"], errors="coerce")
    out_of_range = df[(scores < 0) | (scores > 1) | scores.isna()]
    if out_of_range.empty:
        return []
    sample = out_of_range.index[:10].tolist()
    return [
        f"{len(out_of_range)} rows have anomaly_score outside [0,1] or non-numeric. "
        f"Row indices: {sample}"
    ]


# ── A12: Label ↔ score thresholds must be consistent ──────────────────
# Fix 1: Read calibrated thresholds from anomaly_df columns
# (threshold_critical, threshold_high, threshold_medium) added by Stage 4's
# percentile-calibration step S4-3.  These vary per run — e.g. on the
# klares-app-7 run CRITICAL≥0.949, HIGH≥0.746, MEDIUM≥0.671.  Using the old
# fixed values (CRITICAL≥0.80, HIGH≥0.60, MEDIUM≥0.35) caused false failures
# for scores like 0.830 that are correctly labeled HIGH under calibrated thresholds.
# Falls back to the fixed thresholds with a warning if the columns are absent.
_A12_FALLBACK_THRESHOLDS = {
    "critical": 0.80,
    "high":     0.60,
    "medium":   0.35,
}


def _read_a12_thresholds(df: pd.DataFrame) -> tuple[float, float, float, bool]:
    """
    Return (t_critical, t_high, t_medium, is_calibrated).

    Reads the first non-null value from threshold_critical / threshold_high /
    threshold_medium columns (they are constant per run — same on every row).
    Returns fixed fallback values and is_calibrated=False if absent.
    """
    def _first_valid(col_name: str, fallback: float) -> tuple[float, bool]:
        if col_name not in df.columns:
            return fallback, False
        vals = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if vals.empty:
            return fallback, False
        return float(vals.iloc[0]), True

    t_crit,  ok_c = _first_valid("threshold_critical", _A12_FALLBACK_THRESHOLDS["critical"])
    t_high,  ok_h = _first_valid("threshold_high",     _A12_FALLBACK_THRESHOLDS["high"])
    t_med,   ok_m = _first_valid("threshold_medium",   _A12_FALLBACK_THRESHOLDS["medium"])
    is_calibrated  = ok_c and ok_h and ok_m
    return t_crit, t_high, t_med, is_calibrated


@_register_s4("A12")
def assert_label_score_consistency(df: pd.DataFrame) -> List[str]:
    """anomaly_label must match Stage 4's calibrated score thresholds (or fixed fallback if absent)."""
    if "anomaly_label" not in df.columns or "anomaly_score" not in df.columns:
        return []

    scores = pd.to_numeric(df["anomaly_score"], errors="coerce")
    t_crit, t_high, t_med, is_calibrated = _read_a12_thresholds(df)

    warnings: List[str] = []
    if not is_calibrated:
        warnings.append(
            "threshold_critical / threshold_high / threshold_medium columns not found in "
            "anomaly_df — falling back to fixed thresholds "
            f"(CRITICAL≥{t_crit}, HIGH≥{t_high}, MEDIUM≥{t_med}). "
            "Add Stage 4's S4-3 calibration columns for accurate A12 checks."
        )

    def _expected_label(s: float) -> Optional[str]:
        if pd.isna(s):
            return None
        if s >= t_crit:
            return "CRITICAL"
        if s >= t_high:
            return "HIGH"
        if s >= t_med:
            return "MEDIUM"
        return "LOW"

    expected = scores.apply(_expected_label)
    # Align on index before comparing to avoid pandas reindex warning
    mismatch_mask = df["anomaly_label"].astype(str) != expected.astype(str)
    mismatch      = df[mismatch_mask & expected.notna()]
    if mismatch.empty:
        return warnings   # calibration warning only, all rows correct
    failures = list(warnings)
    sample   = mismatch.head(5)
    for idx, row in sample.iterrows():
        sc = scores.loc[idx]
        failures.append(
            f"row {idx}: score={sc:.3f} → expected '{expected.loc[idx]}' "
            f"but got '{row['anomaly_label']}' "
            f"(thresholds: CRITICAL≥{t_crit:.3f}, HIGH≥{t_high:.3f}, MEDIUM≥{t_med:.3f})"
        )
    if len(mismatch) > 5:
        failures.append(f"… and {len(mismatch) - 5} more mismatches.")
    return failures


# ── A13: impossible_attempt_count must be MEDIUM or above ─────────────
@_register_s4("A13")
def assert_impossible_attempt_count_severity(df: pd.DataFrame) -> List[str]:
    """Rows with singleton_class=impossible_attempt_count must not be labeled LOW."""
    if "singleton_class" not in df.columns or "anomaly_label" not in df.columns:
        return []
    bad = df[
        (df["singleton_class"].astype(str) == "impossible_attempt_count")
        & (df["anomaly_label"].astype(str) == "LOW")
    ]
    if bad.empty:
        return []
    eids = bad.get(
        "event_id",
        bad.get("semantic_cluster_id", bad.index.to_series())
    ).head(10).tolist()
    return [
        f"{len(bad)} impossible_attempt_count events are labeled LOW "
        f"(expected MEDIUM or above). Event IDs: {eids}"
    ]


# ── A14: singleton_class must be a known value ────────────────────────
@_register_s4("A14")
def assert_singleton_class_valid(df: pd.DataFrame) -> List[str]:
    """singleton_class must be one of the known taxonomy values emitted by Stage 3."""
    if "singleton_class" not in df.columns:
        return []
    # Taxonomy reflects what classify_singletons() in stage3.py actually emits.
    # Removed stale values: true_anomaly_error, true_anomaly_warn, normal, noise.
    # Added: known_normal, noise_filtered (introduced in ACCURACY-FIX-B3).
    allowed = {
        "true_anomaly",
        "unseen_variant",
        "impossible_attempt_count",
        "known_normal",
        "noise_filtered",
        "nan", "",
    }
    vals = df["singleton_class"].dropna().astype(str).unique()
    bad  = {v for v in vals if v not in allowed}
    if not bad:
        return []
    return [
        f"Unexpected singleton_class values: {bad}. "
        f"Allowed: {allowed - {'nan', ''}}."
    ]


# ── A15: trend_direction must be a known value ────────────────────────
@_register_s4("A15")
def assert_trend_direction_valid(df: pd.DataFrame) -> List[str]:
    """trend_direction must be 'rising', 'falling', or 'stable'."""
    if "trend_direction" not in df.columns:
        return []
    allowed = {"rising", "falling", "stable"}
    vals    = df["trend_direction"].dropna().astype(str).str.strip().unique()
    bad     = {v for v in vals if v not in allowed}
    if not bad:
        return []
    return [f"Unexpected trend_direction values: {bad}"]


# ── A16: gt_comparison must be a known value ──────────────────────────
@_register_s4("A16")
def assert_gt_comparison_valid(df: pd.DataFrame) -> List[str]:
    """gt_comparison must be one of the known values."""
    if "gt_comparison" not in df.columns:
        return []
    allowed = {
        "elevated", "suppressed", "baseline",
        "new_event", "no_gt", "insufficient_data",
        "nan", "",
    }
    vals = df["gt_comparison"].dropna().astype(str).unique()
    bad  = {v for v in vals if v not in allowed}
    if not bad:
        return []
    return [f"Unexpected gt_comparison values: {bad}"]


# ── A17: CRITICAL events must not have trend_direction=falling ────────
@_register_s4("A17")
def assert_critical_events_not_falling(df: pd.DataFrame) -> List[str]:
    """A CRITICAL anomaly_label with trend_direction=falling and count>50 and trend_ratio<0.3 is a likely scoring bug."""
    if "anomaly_label" not in df.columns or "trend_direction" not in df.columns:
        return []
    count_col = pd.to_numeric(df.get("count", pd.Series(dtype=float)), errors="coerce").fillna(0)
    trend_ratio_col = pd.to_numeric(df.get("trend_ratio", pd.Series(dtype=float)), errors="coerce").fillna(1.0)
    bad = df[
        (df["anomaly_label"].astype(str) == "CRITICAL")
        & (df["trend_direction"].astype(str) == "falling")
        & (count_col > 50)
        & (trend_ratio_col < 0.3)
    ]
    if bad.empty:
        return []
    eid_col = next(
        (c for c in ("event_id", "semantic_cluster_id", "cluster_id") if c in df.columns),
        None,
    )
    sample = bad[eid_col].head(5).tolist() if eid_col else bad.index[:5].tolist()
    return [
        f"{len(bad)} CRITICAL events have trend_direction=falling with count>50 and ratio<0.3 — "
        f"review scoring weights. Sample IDs: {sample}"
    ]


# ══════════════════════════════════════════════════════════════════════
# PIPELINE OUTPUT ASSERTIONS  (A18–A20) — new Stage 2/3 columns
# ══════════════════════════════════════════════════════════════════════

# ── shared helper ─────────────────────────────────────────────────────
def _is_noise_mask(df: pd.DataFrame) -> "pd.Series":
    """
    Return a boolean Series that is True for noise rows.
    Handles both real bool columns and string-encoded booleans
    (e.g. 'True'/'False') that come from round-tripping through CSV.
    """
    if "is_noise" not in df.columns:
        import pandas as _pd
        return _pd.Series(False, index=df.index)
    col = df["is_noise"]
    # In newer pandas, string columns may have dtype='string' (StringDtype) rather
    # than dtype=object, so check for non-bool rather than dtype==object.
    if not pd.api.types.is_bool_dtype(col):
        return col.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    return col.fillna(False).astype(bool)


# ── A18: anomaly_signal must be a known value ─────────────────────────
@_register_s3("A18")
def assert_anomaly_signal_valid(df: pd.DataFrame) -> List[str]:
    """anomaly_signal must be one of: true_anomaly, unseen_variant, routine, noise_filtered."""
    if "anomaly_signal" not in df.columns:
        return ["anomaly_signal column missing from pipeline output (added by Stage 3)"]
    allowed = {"true_anomaly", "unseen_variant", "routine", "noise_filtered"}
    check = df[~_is_noise_mask(df)]
    vals  = check["anomaly_signal"].dropna().astype(str).str.strip().unique()
    bad   = {v for v in vals if v and v != "nan" and v not in allowed}
    if not bad:
        return []
    return [
        f"Unexpected anomaly_signal values: {bad}. "
        f"Allowed: {allowed}."
    ]


# ── A19: is_routine must be consistent with anomaly_signal ────────────
@_register_s3("A19")
def assert_is_routine_consistent(df: pd.DataFrame) -> List[str]:
    """is_routine must be True iff anomaly_signal == 'routine'."""
    if "anomaly_signal" not in df.columns or "is_routine" not in df.columns:
        return []

    def _as_bool(val) -> Optional[bool]:
        if pd.isna(val):
            return None
        return str(val).strip().lower() in ("true", "1", "yes")

    failures = []
    check = df[
        df["anomaly_signal"].notna()
        & df["is_routine"].notna()
        & ~_is_noise_mask(df)
    ]
    for _, row in check.iterrows():
        signal   = str(row["anomaly_signal"]).strip()
        is_rtn   = _as_bool(row["is_routine"])
        expected = (signal == "routine")
        if is_rtn is not None and is_rtn != expected:
            failures.append(
                f"line_no={row.get('line_no','?')}  anomaly_signal='{signal}'  "
                f"is_routine={is_rtn} (expected {expected})"
            )
    if not failures:
        return []
    sample = failures[:5]
    if len(failures) > 5:
        sample.append(f"… and {len(failures) - 5} more.")
    return sample


# ── A20: domain_confidence must be in [0, 1] ─────────────────────────
@_register_s3("A20")
def assert_domain_confidence_range(df: pd.DataFrame) -> List[str]:
    """domain_confidence must be a float in [0.0, 1.0] on every non-noise row."""
    if "domain_confidence" not in df.columns:
        return ["domain_confidence column missing from pipeline output (added by Stage 2)"]
    check  = df[~_is_noise_mask(df)]
    scores = pd.to_numeric(check["domain_confidence"], errors="coerce")
    bad    = check[(scores < 0) | (scores > 1) | scores.isna()]
    if bad.empty:
        return []
    sample = bad["line_no"].head(10).tolist()
    return [
        f"{len(bad)} non-noise rows have domain_confidence outside [0,1] or non-numeric. "
        f"First 10 line_no: {sample}"
    ]


# ══════════════════════════════════════════════════════════════════════
# STAGE 4 ASSERTIONS  (A21–A22) — new ML columns on anomaly_df
# ══════════════════════════════════════════════════════════════════════

# ── A21: anomaly_source must be a known value ─────────────────────────
@_register_s4("A21")
def assert_anomaly_source_valid(df: pd.DataFrame) -> List[str]:
    """anomaly_source must be one of the values emitted by the S4-ML scoring path."""
    if "anomaly_source" not in df.columns:
        return ["anomaly_source column missing from Stage 4 output (added by S4-ML-6)"]
    # Full enum from stage4.py S4-ML-6:
    #   formula_fallback    — service too small to train models
    #   statistical_fallback — IF+AE disagreed; honest uncertainty fallback
    #   ae_if_ensemble      — both IF and AE trained; ensemble used
    #   ml_if_only          — only IF trained (AE failed/insufficient data)
    #   ml_ae_only          — only AE trained (IF failed)
    #   post_deploy_caution — ML scores dampened for new-template grace period
    allowed = {
        "formula_fallback",
        "statistical_fallback",
        "ae_if_ensemble",
        "ml_if_only",
        "ml_ae_only",
        "post_deploy_caution",
    }
    vals = df["anomaly_source"].dropna().astype(str).str.strip().unique()
    bad  = {v for v in vals if v and v != "nan" and v not in allowed}
    if not bad:
        return []
    return [
        f"Unexpected anomaly_source values: {bad}. "
        f"Allowed: {allowed}."
    ]


# ── A22: ml_confidence must be in [0, 1] ─────────────────────────────
@_register_s4("A22")
def assert_ml_confidence_range(df: pd.DataFrame) -> List[str]:
    """ml_confidence must be a float in [0.0, 1.0] on every anomaly_df row."""
    if "ml_confidence" not in df.columns:
        return ["ml_confidence column missing from Stage 4 output (added by S4-ML-6)"]
    scores = pd.to_numeric(df["ml_confidence"], errors="coerce")
    bad    = df[(scores < 0) | (scores > 1) | scores.isna()]
    if bad.empty:
        return []
    eid_col = next(
        (c for c in ("event_id", "semantic_cluster_id", "cluster_id") if c in df.columns),
        None,
    )
    sample = bad[eid_col].head(10).tolist() if eid_col else bad.index[:10].tolist()
    return [
        f"{len(bad)} rows have ml_confidence outside [0,1] or non-numeric. "
        f"Sample IDs: {sample}"
    ]


# ══════════════════════════════════════════════════════════════════════
# BENCHMARK  (compares pipeline output against master_ground_truth.csv)
# ══════════════════════════════════════════════════════════════════════

_BENCHMARK_TARGETS = {
    # Original four fields
    "domain":                   0.90,
    "severity":                 0.99,
    "template_id":              0.95,
    "singleton_class":          0.85,
    # Added: anomaly_signal — direct output of Stage 3 classify_singletons()
    # Slightly lower target than singleton_class because it is a derived
    # mapping (singleton → signal) so any singleton error propagates here too.
    "anomaly_signal":           0.85,
    # Added: domain_confidence_band — coarse bucket (low/medium/high) so a
    # human labeller can mark confidence without needing an exact float.
    # Comparison is done bucket-to-bucket; see run_benchmark() for bucketing.
    "domain_confidence_band":   0.80,
}
_GT_COL_MAP = {
    "domain":                   "expected_domain",
    "severity":                 "expected_severity",
    "template_id":              "expected_template_id",
    "singleton_class":          "expected_singleton_class",
    "anomaly_signal":           "expected_anomaly_signal",
    "domain_confidence_band":   "expected_domain_confidence_band",
}


def _confidence_to_band(val) -> str:
    """
    Bucket a domain_confidence float into a coarse band that a human labeller
    can assign without needing an exact float value:
        low    → [0.00, 0.50)
        medium → [0.50, 0.75)
        high   → [0.75, 1.00]
    Returns "" on non-numeric input so the benchmark skips those rows.
    """
    try:
        f = float(val)
    except (TypeError, ValueError):
        return ""
    if f < 0.50:
        return "low"
    if f < 0.75:
        return "medium"
    return "high"


def run_benchmark(pipeline_df: pd.DataFrame, gt_path: Path, source_file: str) -> Dict:
    gt_df = pd.read_csv(gt_path, dtype=str)
    if "source_file" in gt_df.columns and source_file:
        gt_df = gt_df[gt_df["source_file"] == source_file]
    if gt_df.empty:
        return {"status": "no_ground_truth_for_file", "source_file": source_file}
    gt_df["line_no"] = pd.to_numeric(gt_df["line_no"], errors="coerce")

    # Derive domain_confidence_band on the pipeline side before merging so
    # the benchmark can compare bucket-to-bucket rather than float-to-float.
    pipeline_df = pipeline_df.copy()
    if "domain_confidence" in pipeline_df.columns:
        pipeline_df["domain_confidence_band"] = pipeline_df["domain_confidence"].apply(
            _confidence_to_band
        )

    merged  = pipeline_df.merge(gt_df, on="line_no", how="inner", suffixes=("", "_gt"))
    results = {"status": "ok", "n_labeled_rows": len(merged), "fields": {}}

    # ── Coverage warning: singleton_class NaN gap ─────────────────────
    # When both the pipeline output and the ground truth have NaN for
    # singleton_class, the benchmark silently skips those rows and the
    # accuracy number looks fine — even though most rows are unchecked.
    # Flag this explicitly so it never looks like full coverage.
    if "singleton_class" in merged.columns and "expected_singleton_class" in merged.columns:
        both_nan = (
            merged["singleton_class"].isna()
            & (
                merged["expected_singleton_class"].isna()
                | (merged["expected_singleton_class"].astype(str).str.strip() == "")
            )
        )
        n_both_nan = int(both_nan.sum())
        if n_both_nan > 0:
            results["singleton_class_nan_gap"] = (
                f"{n_both_nan}/{len(merged)} rows have singleton_class=NaN in both "
                f"pipeline output and ground truth — skipped by benchmark. "
                f"Label expected_singleton_class for ERROR/WARN rows to get real coverage."
            )

    for field, gt_col in _GT_COL_MAP.items():
        if field not in merged.columns or gt_col not in merged.columns:
            continue
        valid = merged[
            merged[gt_col].notna()
            & (merged[gt_col].astype(str).str.strip() != "")
        ]
        if valid.empty:
            continue
        n_total   = len(valid)

        # For singleton_class: pipeline emits NaN for routine rows that were
        # never classified as anomalies.  Ground truth labels these as
        # known_normal.  Treat pipeline NaN / "" as known_normal so the
        # benchmark doesn't penalise correct routine behaviour.
        if field == "singleton_class":
            def _normalise_sc(series: pd.Series) -> pd.Series:
                return (
                    series.fillna("known_normal")
                    .astype(str).str.strip().str.lower()
                    .replace({"nan": "known_normal", "": "known_normal"})
                )
            pipe_vals = _normalise_sc(valid[field])
            gt_vals   = _normalise_sc(valid[gt_col])
        else:
            pipe_vals = valid[field].astype(str).str.strip().str.lower()
            gt_vals   = valid[gt_col].astype(str).str.strip().str.lower()

        n_correct = (pipe_vals == gt_vals).sum()
        accuracy = n_correct / n_total
        target   = _BENCHMARK_TARGETS.get(field, 0.90)
        passed   = accuracy >= target
        results["fields"][field] = {
            "accuracy":  round(accuracy, 4),
            "target":    target,
            "pass":      passed,
            "n_correct": int(n_correct),
            "n_total":   int(n_total),
        }
        if not passed:
            wrong = valid[pipe_vals != gt_vals][
                ["line_no", "message", field, gt_col]
            ].head(10)
            results["fields"][field]["mismatch_sample"] = wrong.to_dict("records")
    return results


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _derive_s4_path(pipeline_output_path: Path) -> Path:
    """
    Derive the stage4_anomaly.csv path from the pipeline output path.

    Handles two layouts:

    Layout A — run folder (test_run.py output):
        outputs/runs/RUN_ID/stage3_output.csv
        →  outputs/runs/RUN_ID/stage4_anomaly.csv   (sibling file)

    Layout B — flat outputs dir (notebook/manual export):
        outputs/FOO_output.csv
        →  outputs/FOO_stage4_anomaly.csv
    """
    parent = pipeline_output_path.parent
    stem   = pipeline_output_path.stem          # e.g. "stage3_output"

    # Layout A: pipeline output is named stage3_output.csv — look for
    # stage4_anomaly.csv as a sibling in the same run folder.
    if stem == "stage3_output":
        return parent / "stage4_anomaly.csv"

    # Layout B: strip trailing _output suffix and append _stage4_anomaly.
    if stem.endswith("_output"):
        stem = stem[: -len("_output")]
    return parent / f"{stem}_stage4_anomaly.csv"


def _load_df(path: Path, bool_cols=()) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: str(v).strip().lower() in ("true", "1", "yes", "✓")
            )
    if "line_no" in df.columns:
        df["line_no"] = pd.to_numeric(df["line_no"], errors="coerce")
    return df


def _sep(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _run_assertions(registry: Dict, df: pd.DataFrame, label: str) -> bool:
    """Run all assertions in registry against df.  Returns True if all pass."""
    print(f"\n{label}")
    _sep()
    all_pass = True
    for aid, fn in sorted(registry.items()):
        try:
            failures = fn(df)
        except Exception as exc:
            failures = [f"INTERNAL ERROR: {exc}"]
        status = "✅ PASS" if not failures else "❌ FAIL"
        doc    = (fn.__doc__ or "").strip().split("\n")[0]
        print(f"  {aid}  {status}  — {doc}")
        if failures:
            all_pass = False
            for msg in failures[:3]:
                print(f"         ↳ {msg}")
            if len(failures) > 3:
                print(f"         ↳ … and {len(failures) - 3} more.")
    _sep()
    if all_pass:
        print(f"  ✅  All {label} assertions PASSED.\n")
    else:
        print(f"  ❌  One or more {label} assertions FAILED.\n")
    return all_pass


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate log pipeline output — A1–A8, A18–A20 (pipeline) + "
                    "A9–A17, A21–A22 (Stage 4) + benchmark."
    )
    parser.add_argument("--pipeline_output", required=True,
                        help="Path to pipeline output CSV (per-line stage 3 output)")
    parser.add_argument("--ground_truth",
                        default="validation/master_ground_truth.csv",
                        help="Path to master_ground_truth.csv")
    parser.add_argument("--stage4_output", default=None,
                        help="Path to stage4 anomaly_df CSV "
                             "(auto-derived from --pipeline_output if not set)")
    parser.add_argument("--source_file", default=None,
                        help="source_file value to filter ground truth rows "
                             "(defaults to the pipeline output filename stem)")
    args        = parser.parse_args(argv)
    output_path = Path(args.pipeline_output)
    gt_path     = Path(args.ground_truth)
    source_file = args.source_file or output_path.stem

    # Resolve Stage 4 path — explicit flag takes priority over auto-derived.
    s4_path = Path(args.stage4_output) if args.stage4_output else _derive_s4_path(output_path)

    print()
    _sep("═")
    print("  LOG PIPELINE VALIDATOR")
    print(f"  Pipeline output : {output_path}")
    print(f"  Stage 4 anomaly : {s4_path}"
          + (" ✓" if s4_path.exists() else "  (not found — A9–A17 will be skipped)"))
    print(f"  Ground truth    : {gt_path}")
    _sep("═")

    # ── Load pipeline output ──────────────────────────────────────────
    if not output_path.exists():
        print(f"\n❌  Pipeline output not found: {output_path}")
        sys.exit(1)
    try:
        df = _load_df(
            output_path,
            bool_cols=("is_noise", "timestamp_parsed_ok", "is_merged"),
        )
    except Exception as e:
        print(f"\n❌  Failed to read pipeline output: {e}")
        sys.exit(1)

    print(f"\n  Loaded {len(df):,} rows, {len(df.columns)} columns  "
          f"(pipeline output).\n")

    # ── Stage 3 assertions A1–A8, A18–A20 ───────────────────────────────
    s3_pass = _run_assertions(_ASSERTIONS_S3, df, "ASSERTIONS  A1–A8, A18–A20  (pipeline output)")

    # ── Stage 4 assertions A9–A17, A21–A22 ───────────────────────────────
    s4_pass = True
    if s4_path.exists():
        try:
            s4_df = _load_df(s4_path)
        except Exception as e:
            print(f"\n❌  Failed to read Stage 4 output: {e}")
            s4_pass = False
        else:
            print(f"  Loaded {len(s4_df):,} rows, {len(s4_df.columns)} columns  "
                  f"(Stage 4 anomaly_df).\n")
            s4_pass = _run_assertions(
                _ASSERTIONS_S4, s4_df, "ASSERTIONS  A9–A17, A21–A22  (Stage 4 anomaly_df)"
            )
    else:
        print(f"\n  ℹ  Stage 4 assertions skipped — file not found: {s4_path}")
        print(f"     Re-run the pipeline with the updated test_run.py — it now")
        print(f"     saves stage4_anomaly.csv automatically to the run folder.")
        print(f"     Or pass the path explicitly with --stage4_output <path>")
        print()

    # ── Benchmark ─────────────────────────────────────────────────────
    print("BENCHMARK")
    _sep()
    if not gt_path.exists():
        print(f"  ⚠  Ground truth file not found: {gt_path}")
        print("     Run selection_helper.py to create labeled rows.")
        print()
    else:
        bench = run_benchmark(df, gt_path, source_file)
        if bench.get("status") == "no_ground_truth_for_file":
            print(f"  ⚠  No ground truth rows for source_file='{source_file}'.")
            print("     Label rows with selection_helper.py and append to master file.")
        elif bench.get("status") == "ok":
            n = bench.get("n_labeled_rows", 0)
            print(f"  Compared against {n} labeled rows.\n")

            # Print NaN gap warning before field results so it's impossible to miss
            nan_gap = bench.get("singleton_class_nan_gap")
            if nan_gap:
                print(f"  ⚠  COVERAGE GAP: {nan_gap}\n")

            bench_pass = True
            for field, res in bench["fields"].items():
                acc    = res["accuracy"]
                target = res["target"]
                passed = res["pass"]
                icon   = "✅" if passed else "❌"
                print(f"  {icon}  {field:25s}  {acc*100:5.1f}%  "
                      f"(target ≥ {target*100:.0f}%,  "
                      f"{res['n_correct']}/{res['n_total']} correct)")
                if not passed:
                    bench_pass = False
                    for sample in res.get("mismatch_sample", [])[:5]:
                        ln  = sample.get("line_no", "?")
                        got = sample.get(field, "?")
                        exp = sample.get(_GT_COL_MAP[field], "?")
                        msg = str(sample.get("message", ""))[:60]
                        print(f"         line {ln}: got='{got}' expected='{exp}'  "
                              f"msg='{msg}'")
            _sep()
            if bench_pass:
                print("  ✅  All benchmark targets MET.\n")
            else:
                print("  ❌  One or more targets MISSED. Review mismatches above.\n")

    # ── Final exit code ───────────────────────────────────────────────
    sys.exit(0 if (s3_pass and s4_pass) else 1)


if __name__ == "__main__":
    main()