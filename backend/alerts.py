# backend/alerts.py
#
# ══════════════════════════════════════════════════════════════════════
# WEBHOOK ALERT HOOK  (S3.1)
# ══════════════════════════════════════════════════════════════════════
#
# Why this file exists
# --------------------
# The pipeline detects CRITICAL incidents and writes them to CSV.
# Without any notification, users must actively check the dashboard.
# Every production log monitoring system notifies on critical findings —
# this is the feature users ask for first.
#
# This module dispatches a single HTTP POST to a configurable webhook
# URL whenever a completed pipeline run contains CRITICAL-severity
# incidents.  It is intentionally minimal:
#
#   • No retry logic — a failed webhook is logged and dropped.
#     (A broken webhook must never fail or delay a pipeline run.)
#   • No queuing — the POST is synchronous and blocking.
#     (It is called from a background pipeline thread, not the event
#     loop, so blocking is fine.)
#   • No authentication beyond what the webhook URL itself provides
#     (e.g. a secret token embedded in the URL query string).
#
# Public API
# ----------
#   send_alert(run_info: dict, incidents_df: pd.DataFrame) -> None
#       Checks for CRITICAL rows in incidents_df.  If any exist and
#       ALERT_WEBHOOK_URL is set, POSTs a JSON payload to the webhook.
#       If ALERT_WEBHOOK_URL is empty, logs a warning and returns.
#       Never raises an exception.
#
# Webhook payload schema
# ----------------------
# {
#   "run_id":        "2026-04-20_143022_app",
#   "log_filename":  "app.log",
#   "n_critical":    3,
#   "incident_ids":  ["INC-001", "INC-003", "INC-007"],
#   "started_at":    "2026-04-20T14:30:22.000000+00:00",
#   "completed_at":  "2026-04-20T14:32:48.000000+00:00",
#   "dashboard_url": ""   // populated if DASHBOARD_BASE_URL is set
# }
#
# Configuration
# -------------
# Set these environment variables (or add to a .env file):
#
#   ALERT_WEBHOOK_URL     — POST target (required to enable alerts)
#   DASHBOARD_BASE_URL    — optional; appended to payload as a deep-link
#                           e.g. "https://myapp.example.com"
#
# PLACEMENT: backend/alerts.py
# ══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import List

import pandas as pd

# Allow imports from the project root (where config.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ALERT_WEBHOOK_URL

logger = logging.getLogger("alerts")

# Optional base URL for the dashboard deep-link included in the payload.
# Not in config.py because it is specific to the deployed environment
# and rarely needs to change.
_DASHBOARD_BASE_URL: str = os.environ.get("DASHBOARD_BASE_URL", "")

# Column names to look for when identifying incident severity and ID.
# Listed in priority order — first match wins.
_SEVERITY_COL_CANDIDATES = [
    "incident_severity",
    "peak_anomaly_level",
    "anomaly_label",
    "severity",
]
_INCIDENT_ID_CANDIDATES = [
    "incident_id",
    "run_id",
    "event_id",
    "cluster_id",
]


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def send_alert(run_info: dict, incidents_df: pd.DataFrame) -> None:
    """
    S3.1 — Dispatch a webhook notification if CRITICAL incidents exist.

    Parameters
    ----------
    run_info : dict
        The completed run's info dict (as written to run_info.json).
        Must contain at minimum "run_id".
    incidents_df : pd.DataFrame
        The Stage 5 incidents DataFrame.  May be empty.

    Behaviour
    ---------
    • If incidents_df contains no CRITICAL rows → returns silently
      (no POST, no log noise).
    • If ALERT_WEBHOOK_URL is empty → logs a one-time warning and
      returns silently.
    • If the POST fails (network error, non-2xx response) → logs the
      error as a warning and returns.  Never raises.
    """
    # ── Guard: nothing to do if no incidents or df is empty ───────────
    if incidents_df is None or incidents_df.empty:
        return

    # ── Identify CRITICAL rows ─────────────────────────────────────────
    sev_col = _resolve_col(incidents_df, _SEVERITY_COL_CANDIDATES)
    if sev_col is None:
        logger.debug(
            "send_alert: no severity column found in incidents_df — "
            "cannot determine CRITICAL rows; skipping alert"
        )
        return

    critical_mask = (
        incidents_df[sev_col]
        .fillna("")
        .astype(str)
        .str.upper()
        == "CRITICAL"
    )
    n_critical = int(critical_mask.sum())

    if n_critical == 0:
        return  # No CRITICAL incidents — nothing to alert on

    # ── Guard: webhook URL must be configured ─────────────────────────
    webhook_url = ALERT_WEBHOOK_URL.strip()
    if not webhook_url:
        logger.warning(
            "send_alert: %d CRITICAL incident(s) detected in run_id=%s "
            "but ALERT_WEBHOOK_URL is not set — no notification sent.  "
            "Set the ALERT_WEBHOOK_URL environment variable to enable alerts.",
            n_critical,
            run_info.get("run_id", "unknown"),
        )
        return

    # ── Collect incident IDs for the payload ──────────────────────────
    id_col = _resolve_col(incidents_df, _INCIDENT_ID_CANDIDATES)
    if id_col is not None:
        incident_ids: List[str] = (
            incidents_df.loc[critical_mask, id_col]
            .dropna()
            .astype(str)
            .tolist()
        )
    else:
        incident_ids = []

    # ── Build payload ──────────────────────────────────────────────────
    run_id = run_info.get("run_id", "unknown")

    dashboard_url = ""
    if _DASHBOARD_BASE_URL:
        dashboard_url = f"{_DASHBOARD_BASE_URL.rstrip('/')}/runs/{run_id}"

    payload = {
        "run_id":        run_id,
        "log_filename":  run_info.get("log_filename", ""),
        "n_critical":    n_critical,
        "incident_ids":  incident_ids,
        "started_at":    run_info.get("started_at", ""),
        "completed_at":  run_info.get("completed_at", ""),
        "dashboard_url": dashboard_url,
    }

    # ── POST to webhook ────────────────────────────────────────────────
    _post_webhook(webhook_url, payload, run_id=run_id)


# ══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════

def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first candidate column name that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _post_webhook(url: str, payload: dict, run_id: str = "") -> None:
    """
    POST JSON payload to url using only the standard library.

    Using urllib instead of requests/httpx keeps this module dependency-
    free — the alerts module should work even before the full API
    dependencies are installed.

    Logs success at INFO level and failure at WARNING level.
    Never raises.
    """
    try:
        body    = json.dumps(payload, default=str).encode("utf-8")
        request = urllib.request.Request(
            url,
            data    = body,
            headers = {
                "Content-Type": "application/json",
                "User-Agent":   "AI-Log-Monitor-Alerts/1.0",
            },
            method  = "POST",
        )

        # 10 second timeout — a slow webhook must not hold up the worker
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
            if 200 <= status < 300:
                logger.info(
                    "send_alert: webhook POST succeeded  "
                    "run_id=%s  n_critical=%d  status=%d  url=%.60s...",
                    run_id, payload.get("n_critical", 0), status, url,
                )
            else:
                logger.warning(
                    "send_alert: webhook returned non-2xx status %d  "
                    "run_id=%s  url=%.60s...",
                    status, run_id, url,
                )

    except urllib.error.URLError as exc:
        logger.warning(
            "send_alert: webhook POST failed (network error)  "
            "run_id=%s  error=%s  url=%.60s...",
            run_id, exc.reason if hasattr(exc, "reason") else exc, url,
        )
    except Exception as exc:
        logger.warning(
            "send_alert: unexpected error during webhook POST  "
            "run_id=%s  error=%s",
            run_id, exc,
        )