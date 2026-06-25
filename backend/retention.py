# backend/retention.py
#
# ══════════════════════════════════════════════════════════════════════
# DATA RETENTION POLICY  (S2.1 + S2.2)
# ══════════════════════════════════════════════════════════════════════
#
# Why this file exists
# --------------------
# Run folders accumulate indefinitely.  After 30 days of active use
# (even at one run per day), outputs/runs/ will hold 30+ folders each
# containing 5–7 CSVs.  Without cleanup, disk fills silently and the
# API starts failing with no obvious error.
#
# Public API
# ----------
#   apply_retention_policy() -> dict
#       Reads the runs index, identifies runs that violate either the
#       age limit (RETENTION_DAYS) or the count limit (MAX_STORED_RUNS),
#       and deletes them using pipeline.delete_run().
#
#       Returns a summary dict with keys:
#           deleted_count    : int   — number of runs deleted
#           deleted_run_ids  : list  — run_ids that were removed
#           remaining_count  : int   — runs still in the index after cleanup
#
#       Raises no exceptions — all errors are logged as warnings so a
#       retention failure never crashes the API startup.
#
# Call site
# ---------
#   Called once from the API's lifespan handler on startup:
#
#       from retention import apply_retention_policy
#       apply_retention_policy()
#
# PLACEMENT: backend/retention.py
# ══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List

# Allow imports from the project root (where config.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RETENTION_DAYS, MAX_STORED_RUNS

# pipeline.py provides both the index helpers and delete_run().
# Imported here (not at module top-level) to avoid any circular import
# risk — retention.py is imported by the API lifespan, which also
# imports pipeline.py.
from pipeline import _load_runs_index, delete_run

logger = logging.getLogger("retention")


# ══════════════════════════════════════════════════════════════════════
# MAIN POLICY FUNCTION
# ══════════════════════════════════════════════════════════════════════

def apply_retention_policy(
    retention_days:   int | None = None,
    max_stored_runs:  int | None = None,
) -> Dict:
    """
    Apply the data retention policy to the runs index.

    Two independent eviction criteria are evaluated and combined:

    1. Age limit — runs older than RETENTION_DAYS are deleted.
       The run's "completed_at" field is used; if missing, "started_at"
       is tried; if both are missing the run is treated as aged-out to
       avoid accumulating entries with no timestamp.

    2. Count limit — if more than MAX_STORED_RUNS runs remain after the
       age pass, the oldest excess runs are deleted until the count is
       at or below the limit.  Oldest is determined by "completed_at"
       ascending.

    Parameters
    ----------
    retention_days : int | None
        Override for RETENTION_DAYS from config.py.  If None, uses the
        config value.
    max_stored_runs : int | None
        Override for MAX_STORED_RUNS from config.py.  If None, uses the
        config value.

    Returns
    -------
    dict with keys:
        deleted_count   : int   — total runs deleted in this call
        deleted_run_ids : list  — run_ids that were removed
        remaining_count : int   — runs remaining in the index
    """
    days_limit  = int(retention_days  if retention_days  is not None else RETENTION_DAYS)
    count_limit = int(max_stored_runs if max_stored_runs is not None else MAX_STORED_RUNS)

    deleted_ids: List[str] = []

    try:
        index = _load_runs_index()
    except Exception as exc:
        logger.warning("retention: could not load runs index — %s", exc)
        return {"deleted_count": 0, "deleted_run_ids": [], "remaining_count": 0}

    if not index:
        logger.info("retention: index is empty — nothing to do")
        return {"deleted_count": 0, "deleted_run_ids": [], "remaining_count": 0}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_limit)

    # ── Pass 1: age-based eviction ────────────────────────────────────
    age_candidates: List[str] = []

    for run in index:
        run_id = run.get("run_id")
        if not run_id:
            continue

        # Try completed_at first, fall back to started_at
        ts_str = run.get("completed_at") or run.get("started_at") or run.get("run_timestamp")

        if not ts_str:
            # No timestamp at all — treat as expired to avoid orphan entries
            logger.warning(
                "retention: run_id=%s has no timestamp — treating as expired",
                run_id,
            )
            age_candidates.append(run_id)
            continue

        try:
            # Parse ISO-format timestamp; handle both tz-aware and naive
            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts = ts.astimezone(timezone.utc)
        except Exception:
            logger.warning(
                "retention: run_id=%s has unparseable timestamp '%s' — treating as expired",
                run_id, ts_str,
            )
            age_candidates.append(run_id)
            continue

        if ts < cutoff:
            age_candidates.append(run_id)

    for run_id in age_candidates:
        _safe_delete(run_id, deleted_ids, reason=f"age > {days_limit} days")

    # ── Pass 2: count-based eviction ──────────────────────────────────
    # Reload the index after deletions so the count is accurate
    try:
        index = _load_runs_index()
    except Exception as exc:
        logger.warning("retention: could not reload index after age pass — %s", exc)
        index = []

    excess = len(index) - count_limit
    if excess > 0:
        # Sort oldest-first by completed_at / started_at to evict the
        # oldest excess runs.  Runs without a parseable timestamp sort
        # to the front (treated as oldest).
        def _sort_key(run: dict) -> str:
            return (
                run.get("completed_at")
                or run.get("started_at")
                or run.get("run_timestamp")
                or "0000"
            )

        sorted_runs = sorted(index, key=_sort_key)
        count_candidates = [r.get("run_id") for r in sorted_runs[:excess] if r.get("run_id")]

        for run_id in count_candidates:
            _safe_delete(
                run_id, deleted_ids,
                reason=f"count limit ({count_limit}) exceeded",
            )

    # ── Final state ───────────────────────────────────────────────────
    try:
        remaining = len(_load_runs_index())
    except Exception:
        remaining = 0

    summary = {
        "deleted_count":   len(deleted_ids),
        "deleted_run_ids": deleted_ids,
        "remaining_count": remaining,
    }

    if deleted_ids:
        logger.info(
            "retention: deleted %d run(s): %s — %d run(s) remain",
            len(deleted_ids),
            ", ".join(deleted_ids[:10]) + ("..." if len(deleted_ids) > 10 else ""),
            remaining,
        )
    else:
        logger.info(
            "retention: nothing to delete — %d run(s) within policy "
            "(age <= %dd, count <= %d)",
            remaining, days_limit, count_limit,
        )

    return summary


# ══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════

def _safe_delete(run_id: str, deleted_ids: List[str], reason: str) -> None:
    """
    Call pipeline.delete_run() and record the result.  Never raises —
    a deletion failure is logged as a warning so the rest of the
    retention pass can continue.
    """
    try:
        deleted = delete_run(run_id)
        if deleted:
            deleted_ids.append(run_id)
            logger.info("retention: deleted run_id=%s (%s)", run_id, reason)
        else:
            logger.warning(
                "retention: delete_run(%s) returned False — "
                "run may have already been deleted manually",
                run_id,
            )
    except Exception as exc:
        logger.warning(
            "retention: failed to delete run_id=%s — %s", run_id, exc
        )