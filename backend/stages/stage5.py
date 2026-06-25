"""
stage5.py — Stage 5: Root Cause Analysis
=========================================
v7 — Blueprint BP-S5-1 through BP-S5-8 + Master Spec Upgrades

Public API
----------
    from stages.stage5 import run_stage5

    results = run_stage5(
        stage4_results,
        raw_df=df_stage3,
        cluster_manifest=cluster_manifest,   # strongly recommended
        stage1_stats=stage1_stats_dict,      # optional, for pipeline_metadata
        stage2_stats=stage2_stats_dict,      # optional
        stage3_stats=stage3_stats_dict,      # optional
    )

    df_incidents       = results["incidents_df"]
    df_incident_events = results["incident_events_df"]
    df_timeline        = results["timeline_df"]
    df_unlinked        = results["unlinked_anomalies"]
    incident_map       = results["incident_map"]
    col_map            = results["col_map"]
    pipeline_metadata  = results["pipeline_metadata"]

Return contract (dict keys)
----------------------------
    incidents_df        — one row per incident (cluster_ids, severity, narrative, etc.)
    incident_events_df  — anomalous clusters with cascade chain fields
    timeline_df         — raw log lines that belong to any incident
    unlinked_anomalies  — anomalous clusters not linked to any incident
    incident_map        — dict {incident_id: [cluster_ids]}
    col_map             — resolved column name map
    config_used         — full merged config
    pipeline_metadata   — BP-S5-7 audit block

Blueprint changes in v6
------------------------
BP-S5-1  Manifest-sourced event counts (total_event_count, error_event_count)
BP-S5-2  Explicit cluster_ids field; n_clusters = len(cluster_ids) always
BP-S5-3  Cross-incident deduplication (cluster overlap OR time window overlap)
BP-S5-4  LLM grounding validation on narrative fields
BP-S5-5  Deterministic fallback narrative when grounding fails
BP-S5-6  Cross-stage consistency assertions (5 assertions, logged to metadata)
BP-S5-7  Pipeline metadata block (pipeline_metadata.json written to output_dir)
BP-S5-8  services_affected as explicit list; n_services_affected = len() always

Master Spec Upgrades in v7
---------------------------
S5-ML-1  Anthropic LLM narrative generation (claude-sonnet-4-20250514, temperature=0)
         Fill-in-template prompt with confidence gate (n_clusters>=2, score>=0.60,
         non-empty cascade, domain not excluded).
S5-ML-2  Causal-claim grounding verification — sentences with causal language
         ("caused", "led to", "triggered", "because", "resulted in", "due to")
         must cite a cluster_id or service name. Sentences failing check are stripped.
S5-ML-3  FAISS historical incident similarity search — top-3 similar past incidents
         retrieved and surfaced in recommended_action and similar_past_incidents field.
S5-ML-4  Confidence gate — LLM skipped for weak incidents; deterministic fallback used.
S5-ML-5  FAISS hub detection and suppression — incidents retrieved >20% of queries
         are downranked and suppressed to prevent hubbing.
FIX-UF   Union-Find cap — max candidates = 1,000 before O(n²) loop;
         fallback to timestamp-only grouping when exceeded.
NEW-COL  narrative_source column: 'llm_grounded' | 'llm_fallback_stripped' |
         'deterministic_fallback'
NEW-COL  similar_past_incidents column: JSON list of top-3 FAISS matches

Also retained from v5/v6
------------------------
FIX-3A/B/C  Domain column resolution from multiple candidate sources
FIX-4       Error class bucketing — config-driven patterns
FIX-5A      Success-message hard exclusion before incident grouping
FIX-5B      All-success incident skip
FIX-6A      Cascade simultaneity threshold (configurable, not hardcoded 0 s)
FIX-6B      Full ordered cascade chain always built
FIX-6C      Recurrence keyed on (root_cause_service, error_class) pair
FIX-6D      Tiebreaker sort within tied-timestamp cascade groups
NEW-A       what_happened signature generator
NEW-B       Error trigger extraction
NEW-C       Severity guard (instant CRITICAL → HIGH downgrade without fatal signal)
NEW-D       Recommended action v4 with context-aware remediation map
Fix-6 (v6) Zero-field validation — repair manifest-sourced zeros
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import time as _time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("stage5")

# ─────────────────────────────────────────────────────────────────────
# MODULE-LEVEL ML STATE  (S5-ML-1 / S5-ML-3 / S5-ML-5)
# ─────────────────────────────────────────────────────────────────────

# LLM — lazy Anthropic client handle (None until first successful import)
_ANTHROPIC_CLIENT: Any = None
_ANTHROPIC_AVAILABLE: bool = False

try:
    import anthropic as _anthropic_mod
    _ANTHROPIC_CLIENT = _anthropic_mod.Anthropic()
    _ANTHROPIC_AVAILABLE = True
    logger.info("S5-ML-1: anthropic package available — LLM narrative generation enabled")
except ImportError:
    logger.info("S5-ML-1: anthropic not installed — LLM narrative generation will use deterministic fallback")
except Exception as _e:
    logger.warning("S5-ML-1: anthropic import error (%s) — deterministic fallback active", _e)

# Ollama availability check — attempted once at import time.
# The base URL is read from the environment or defaults to localhost:11434.
# Stage 5 reads the resolved URL from cfg at call time; this check only
# uses the default so the startup log is meaningful.
_OLLAMA_AVAILABLE: bool    = False
_OLLAMA_DEFAULT_URL: str   = "http://localhost:11434"
_OLLAMA_LOADED_MODELS: list = []

try:
    import requests as _requests_mod
    _ollama_probe = _requests_mod.get(
        f"{_OLLAMA_DEFAULT_URL}/api/tags", timeout=3
    )
    if _ollama_probe.status_code == 200:
        _OLLAMA_AVAILABLE = True
        # Extract loaded model names for startup validation
        _tags_data = _ollama_probe.json()
        _OLLAMA_LOADED_MODELS = [
            m.get("name", "") for m in _tags_data.get("models", [])
        ]
        # Check if the configured model (phi4:14b) is actually present
        _expected_model = "phi4:14b"
        _model_present  = any(
            _expected_model in m for m in _OLLAMA_LOADED_MODELS
        )
        if _model_present:
            logger.info(
                "S5-ML-1: Ollama server at %s — model '%s' confirmed ✅ "
                "LLM narrative generation enabled",
                _OLLAMA_DEFAULT_URL, _expected_model,
            )
        else:
            logger.warning(
                "S5-ML-1: Ollama running at %s but '%s' not found in loaded models. "
                "Available: %s. Run 'ollama pull %s' if missing.",
                _OLLAMA_DEFAULT_URL, _expected_model,
                _OLLAMA_LOADED_MODELS, _expected_model,
            )
    else:
        logger.warning(
            "S5-ML-1: Ollama probe returned HTTP %d — LLM narrative will use "
            "deterministic fallback",
            _ollama_probe.status_code,
        )
except Exception as _ollama_probe_exc:
    logger.info(
        "S5-ML-1: Ollama not reachable at %s (%s) — "
        "deterministic fallback active. Start Ollama with 'ollama serve'.",
        _OLLAMA_DEFAULT_URL, _ollama_probe_exc,
    )

# FAISS module-level state (S5-ML-3 / S5-ML-5)
_FAISS_INDEX: Any = None           # faiss.IndexFlatIP when loaded
_FAISS_INCIDENT_STORE: List = []   # parallel list of incident record dicts
_FAISS_RETRIEVAL_COUNTS: Dict[int, int] = {}   # idx → retrieval count
_FAISS_TOTAL_QUERIES: int = 0      # total queries for hub detection

_FAISS_AVAILABLE: bool = False
try:
    import faiss as _faiss_mod   # noqa: F401
    _FAISS_AVAILABLE = True
    logger.info("S5-ML-3: faiss-cpu available — historical incident similarity enabled")
except ImportError:
    logger.info("S5-ML-3: faiss-cpu not installed — similar_past_incidents will be []")

# ─────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────

class PipelineConsistencyError(Exception):
    """Raised when a hard consistency assertion fails and raise_on_error=True."""
    pass


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

STAGE5_CONFIG: Dict[str, Any] = {
    "incident_anomaly_labels"          : {"CRITICAL", "HIGH"},
    "include_burst_medium"             : True,
    "incident_window_fallback_s"       : 300,
    "incident_window_multiplier"       : 2.0,
    "co_domain_requires_either"        : True,
    "min_clusters_per_incident"        : 1,
    "cascade_min_lag_s"                : 1,
    "cascade_simultaneity_threshold_s" : 1,    # FIX-6A
    "severity_rank"                    : {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1},

    # BP-S5-3: cross-incident deduplication thresholds
    "dedup_cluster_overlap_threshold"  : 0.50,
    "dedup_time_overlap_threshold"     : 0.80,

    # S5-WINDOW-CAP: absolute ceiling on the adaptive incident grouping window.
    # The effective cap is min(5% of log span, incident_window_max_s).
    # Increase this for environments where incidents genuinely last > 30 minutes.
    "incident_window_max_s"            : 1800,

    "domain_priority": [
        "audit", "security", "payment", "telemetry", "hardware",
        "campaign", "inventory", "auth", "database", "messaging",
        "connectivity", "api", "storage", "scheduler",
        "infrastructure", "profile", "network", "other",
    ],

    # S5-5B: domain is always named "domain" after Stage 2 consolidation.
    # Fallback names removed — their presence implied alternate column names
    # were valid upstream outputs, which they are not after the fix.
    "domain_col_candidates": ["domain"],

    "success_message_patterns": [
        r"0 error\(s\)",
        r"finished:\s*\d+\s+file\(s\)\s+archived",
        r"archived,\s*0 error",
        r"completed successfully",
        r"✅",
        r"\bno errors?\b",
        r"success(?:fully)?\b",
    ],

    "impossible_patterns": [
        r"does not exist",
        r"fetch failed",
        r"\bENOENT\b",
        r"\bEADDRINUSE\b",
        r"not found after \d+ retr",
        r"relation .+ does not exist",
        r"Local file not found",
    ],

    # FIX-4: error class bucketing — config-driven, generalises to any log file.
    # S5-2 FIX: Removed app-specific patterns (GCS_ERROR, DAEMON_FAIL, BUILD_ERROR,
    # SESSION_SAVE_ERROR, PROCESS_KILL_ERROR). These only matched one application's
    # technology stack and produced GENERIC_ERROR for everything else. Per-deployment
    # configs should extend this list via cfg={"error_class_patterns": [...]}.
    # Patterns kept here are universally applicable across log formats.
    "error_class_patterns": [
        (r"\bEADDRINUSE\b",                              "EADDRINUSE"),
        (r"\bENOENT\b",                                  "ENOENT"),
        (r"\bECONNREFUSED\b",                            "ECONNREFUSED"),
        (r"\bEACCES\b",                                  "EACCES"),
        (r"\bEPIPE\b",                                   "EPIPE"),
        (r"\bETIMEDOUT\b",                               "ETIMEDOUT"),
        (r"\bERESET\b|\bECONNRESET\b",                  "ECONNRESET"),
        (r"FATAL.*terminat|terminat.*connection.*admin|pg.*FATAL", "DB_FATAL"),
        (r"relation\s+\".+\"\s+does not exist",          "DB_RELATION_MISSING"),
        (r"idle client|connection pool.*error",           "DB_POOL"),
        (r"deadlock detected|deadlock found",             "DB_DEADLOCK"),
        (r"out of memory|OOMKill|oom.kill",               "OOM"),
        (r"(?:GET|POST|PUT|DELETE|PATCH)\s+\S+\s+5\d{2}", "HTTP_5xx"),
        (r"(?:GET|POST|PUT|DELETE|PATCH)\s+\S+\s+4\d{2}", "HTTP_4xx"),
        (r"TypeError|ReferenceError|SyntaxError",         "JS_EXCEPTION"),
        (r"NullPointerException|NullReferenceException",  "NULL_POINTER"),
        (r"StackOverflowError|stack overflow",            "STACK_OVERFLOW"),
        (r"certificate.*expir|ssl.*handshake.*fail|tls.*handshake.*fail", "TLS_ERROR"),
        (r"disk.*full|no space left on device",           "DISK_FULL"),
    ],

    # NEW-A: error signature → base phrase
    # S5-1 FIX: All app-specific signatures removed (DaemonClient, ddq, GCS,
    # version-stack, pm2 flush, vite/tsx, etc.). Those only matched one
    # application's technology stack and produced the boilerplate fallback for
    # every other log file. Per-deployment configs should add their own signatures
    # via cfg={"what_happened_signatures": [...]}.
    # Signatures kept here are generic POSIX/HTTP patterns universal to any stack.
    "what_happened_signatures": [
        (r"EADDRINUSE",
         "Server startup crash — port already in use"),
        (r"ECONNREFUSED",
         "Downstream service refused connection"),
        (r"ETIMEDOUT|connection timed out",
         "Connection to downstream service timed out"),
        (r"ECONNRESET",
         "Connection reset by peer — possible upstream restart or network issue"),
        (r"ENOENT|file.*not found|no such file",
         "Required file missing from disk"),
        (r"EACCES|permission denied",
         "Permission denied — check file/socket ownership and access rights"),
        (r"EPIPE|broken pipe",
         "Broken pipe — client disconnected before response was written"),
        (r"terminating connection due to admin",
         "Database connections terminated by admin command"),
        (r"relation .+ does not exist",
         "Missing database table caused API 500 error"),
        (r"idle client|connection pool",
         "Database connection pool error"),
        (r"deadlock detected|deadlock found",
         "Database deadlock — concurrent transactions are blocking each other"),
        (r"out of memory|OOMKill",
         "Process killed by OOM — insufficient memory for workload"),
        (r"certificate.*expir|tls.*handshake.*fail|ssl.*handshake.*fail",
         "TLS/SSL failure — certificate expired or handshake rejected"),
        (r"disk.*full|no space left on device",
         "Disk full — write operations will fail until space is freed"),
        (r"401.*Unauthorized|Unauthorized.*401",
         "Service unauthenticated — 401 on startup"),
        (r"POST .+ 500|GET .+ 500",
         "API endpoint returned internal server error 500"),
        (r"POST .+ 404|GET .+ 404",
         "API endpoint returned route-not-found 404"),
        (r"POST .+ 400|GET .+ 400",
         "API request rejected with bad-request 400"),
        (r"Starting server.*NORMAL|Server is running",
         "Server restarted and came back online"),
    ],

    # NEW-B: error trigger extraction
    # S5-2 FIX: Removed app-specific patterns (DaemonClient.*?error=,
    # outcome=\w+|\detail=\w+). These added regex overhead on every anomaly
    # message and only ever matched one application's log format.
    # Per-deployment configs can add their own via cfg={"error_trigger_patterns": [...]}.
    "error_trigger_patterns": [
        (r"(EADDRINUSE[^\n:]*:[^\n]{0,50})",        1),
        (r"(ENOENT[^\n:]*:[^\n]{0,60})",             1),
        (r"(ECONNREFUSED[^\n:]*:[^\n]{0,50})",       1),
        (r"(EACCES[^\n:]*:[^\n]{0,50})",             1),
        (r"(ETIMEDOUT[^\n:]*:[^\n]{0,50})",          1),
        (r"(ECONNRESET[^\n:]*:[^\n]{0,50})",         1),
        (r"(FATAL[:\s]+[^\n]{0,70})",                1),
        (r"severity:\s*'(ERROR|FATAL)'[^\n]{0,60}",  0),
        (r"((?:GET|POST|PUT|DELETE|PATCH)\s+\S+\s+[45]\d{2}[^\n]{0,40})", 1),
        (r"(TypeError\s*[:\[][^\n]{0,60})",          1),
        (r"(ReferenceError[^\n]{0,60})",              1),
        (r"(SyntaxError[^\n]{0,60})",                 1),
        (r"(NullPointerException[^\n]{0,60})",        1),
        (r"(not found (?:in|after)[^\n]{0,60})",      1),
        (r"(fetch failed[^\n]{0,50})",                1),
        (r"(Timeout:\s*\d+\s*ms[^\n]{0,30})",        1),
        (r"(\d+ms\s*timeout[^\n]{0,30})",             1),
        (r"(relation\s+\"[^\"]+\"\s+does not exist)", 1),
        (r"(address already in use[^\n]{0,50})",      1),
        (r"(out of memory[^\n]{0,50})",               1),
        (r"(no space left on device[^\n]{0,50})",     1),
        (r"(certificate.*expir[^\n]{0,60})",          1),
        (r"[Ee]rror[:\s]+([^\n]{0,60})",              1),
    ],

    # NEW-D: remediation verb phrases
    # S5-1 FIX: All app-specific remediations removed (GCS, DaemonClient, DDQ,
    # vite/tsx, version-stack, pm2 flush, Outlook OAuth, document_stacks table,
    # etc.). These produced meaningless boilerplate on any other log file.
    # Per-deployment configs should extend this list via
    # cfg={"remediation_map": [...]}.
    # Entries kept here are universal POSIX/HTTP remediations valid for any stack.
    "remediation_map": [
        (r"EADDRINUSE|address already in use",
         "Identify and kill the orphaned process holding the port "
         "(lsof -i :<port> / netstat -tulpn) before restarting the service."),
        (r"ENOENT|no such file|file.*not found",
         "Verify that all required files and directories exist before the service "
         "starts; check recent deploys or config changes for path regressions."),
        (r"ECONNREFUSED",
         "Confirm the downstream service is running and listening on the expected "
         "address and port; check firewall rules and service discovery config."),
        (r"ETIMEDOUT|connection timed out",
         "Check network latency and downstream service health; consider increasing "
         "the timeout or adding a circuit breaker for the failing dependency."),
        (r"ECONNRESET|connection reset",
         "Review upstream load balancer keep-alive settings and client retry logic; "
         "check for sudden restarts of the downstream service."),
        (r"EACCES|permission denied",
         "Check file and socket ownership and ACLs; ensure the service runs as the "
         "correct user and has the necessary OS-level permissions."),
        (r"EPIPE|broken pipe",
         "Add error handling around write operations to detect early client "
         "disconnection; check proxy idle timeouts and keep-alive settings."),
        (r"FATAL.*terminat|terminat.*connection.*admin",
         "Check pg_stat_activity for blocked connections; review who issued the "
         "admin terminate command and whether it was intentional."),
        (r"relation.*does not exist",
         "Run pending database migrations; the referenced table is missing from "
         "the schema — check migration history and rollback state."),
        (r"idle client|connection pool",
         "Tune the connection pool min/max settings; add reconnect-on-idle logic "
         "and health checks to the database client configuration."),
        (r"deadlock detected|deadlock found",
         "Review transaction ordering to establish a consistent lock acquisition "
         "order; consider shorter transactions and explicit lock timeouts."),
        (r"out of memory|OOMKill",
         "Profile the service for memory leaks; increase container/VM memory limits "
         "or reduce concurrent workload; check for unbounded cache growth."),
        (r"certificate.*expir|tls.*handshake|ssl.*handshake",
         "Renew the TLS certificate before expiry; verify certificate chain "
         "completeness and that all services trust the CA."),
        (r"disk.*full|no space left",
         "Free disk space immediately (clear old logs, temp files, snapshots); "
         "add disk usage alerting to prevent recurrence."),
        (r"401.*Unauthorized|Unauthorized|Not authenticated",
         "Re-authenticate the affected service integration; verify OAuth/API token "
         "refresh logic runs at startup and before token expiry."),
        (r"5\d{2}",
         "Check the upstream service error logs for the root cause of the 5xx; "
         "verify downstream dependencies are healthy."),
        (r"4\d{2}",
         "Validate the request payload and headers against the API contract; check "
         "for schema changes or breaking updates in the downstream API."),
    ],

    "domain_actions": {
        "audit"          : "Review Linux audit records (ausearch/aureport); identify the syscall, auid, and process involved; check for privilege escalation or unexpected execve chains.",
        "security"       : "Escalate to security team; review access logs for unauthorized activity, WAF block reasons, and intrusion indicators.",
        "payment"        : "Inspect payment gateway responses; verify downstream auth and database dependencies; check for Stripe webhook failures or refund loops.",
        "telemetry"      : "Check MAVLink/TLogWriter connectivity; verify GPS lock, flight-mode state machine, and drone link health.",
        "hardware"       : "Inspect serial port / COM device availability; verify accelerometer, compass, and IMU calibration; check RC link quality.",
        "campaign"       : "Review campaign template processing, DDQ embedding pipeline, and document autofill queue for failures.",
        "inventory"      : "Check ERP sync status, SKU lookup service, warehouse stock-level feeds, and reorder trigger logic.",
        "auth"           : "Investigate authentication service; check credentials, LDAP/AD connectivity, session store, JWT/OAuth token refresh, and rate-limiting rules.",
        "database"       : "Check database health, connection pool exhaustion, slow queries, deadlock frequency, and replication lag.",
        "messaging"      : "Inspect message queue depth, consumer lag, broker health, SMTP delivery status, and HTTP 4xx response patterns.",
        "connectivity"   : "Check WebSocket link state, LDAP reachability, TLS handshake success rate, and serial link health.",
        "api"            : "Check API gateway error rates, upstream service health, rate limit headers, and nginx proxy configuration.",
        "storage"        : "Check disk usage, I/O latency, GCS/S3 bucket availability, and file path integrity.",
        "scheduler"      : "Review job queue depth, worker health, lock contention, and background processing pipeline.",
        "infrastructure" : "Investigate affected service logs; check port availability, process management, container lifecycle, and OOM events.",
        "profile"        : "Check user profile service health, account settings API, and preference store consistency.",
        "network"        : "Review network connectivity, firewall rules, DNS resolution, TLS certificate validity, and rate limit policies.",
        "other"          : "Investigate affected service logs, check upstream dependencies, and review recent changes.",
        "unknown"        : "Investigate affected service logs, check upstream dependencies, and review recent changes.",
    },
    "default_action"    : "Investigate affected service logs, check upstream dependencies, and review recent changes.",
    "output_dir"        : "outputs",
    "incidents_filename": "incidents.csv",
    "suppress_display"  : True,   # no IPython display in production mode

    # S5-3: Configurable entity ID extraction patterns used by _extract_doc_ids.
    # Each entry is a (regex, label) tuple. The regex must have exactly one
    # capture group for the ID value. Ships empty — add per-deployment patterns:
    # e.g. (r'docId=(\d+)', 'docId'), (r'orderId=ORD-(\d+)', 'orderId')
    "id_extraction_patterns": [],

    # BP-S5-6: whether to raise hard exception on consistency failure
    "raise_on_consistency_error": False,

    # ── S5-ML-1: LLM narrative generation ────────────────────────────
    # Set llm_narrative_enabled=False to force deterministic-only mode.
    "llm_narrative_enabled"       : True,
    # Provider: "ollama" (local, default) or "anthropic" (cloud fallback).
    "llm_provider"                : "ollama",
    "ollama_base_url"             : "http://localhost:11434",
    "ollama_model"                : "phi4:14b",
    # Anthropic settings (used when llm_provider="anthropic")
    "llm_model"                   : "claude-sonnet-4-20250514",
    "llm_temperature"             : 0,
    "llm_max_tokens"              : 700,    # applied to both providers
    # Domains excluded from LLM (too generic to yield grounded narratives)
    # S5-3: removed "other" — clusters classified as "other" domain now get LLM narratives;
    # only "unknown" (completely unclassified) is excluded
    "llm_excluded_domains"        : {"unknown"},
    # Number of parallel threads for LLM narrative calls.
    # Set to 1 to disable parallelism (sequential, original behaviour).
    # For Ollama (single GPU): 2–3 is safe; more causes GPU contention.
    # For Anthropic API: 4–8 is safe (rate-limit headroom permitting).
    "llm_max_workers"             : 3,

    # ── S5-ML-4: LLM confidence gate ──────────────────────────────────
    # All conditions must hold for the LLM to be invoked:
    # S5-1: lowered from 2 → 1 so single-cluster incidents get LLM narratives
    "llm_min_clusters"            : 1,       # n_clusters >= this
    "llm_min_anomaly_score"       : 0.60,    # top_cluster_score >= this
    # S5-2: removed cascade requirement — single-service logs never had " → " chain
    "llm_require_cascade"         : False,   # cascade_chain not required

    # ── S5-ML-3: FAISS historical similarity ──────────────────────────
    "faiss_top_k"                 : 3,
    "faiss_min_score"             : 0.50,    # cosine-similarity floor
    # S5-ML-5: hub suppression — incident retrieval count / total queries
    "faiss_hub_fraction"          : 0.20,
    # Path to persist FAISS index across runs (None = in-memory only)
    "faiss_index_path"            : None,

    # ── FIX-UF: Union-Find candidate cap ──────────────────────────────
    # When anomalous candidate count exceeds this, fall back to
    # timestamp-only grouping (no domain/service check) to avoid O(n²).
    "union_find_max_candidates"   : 1000,
}


# ══════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════

def _safe_str(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip()


def _parse_services(val) -> list:
    """Parse services_affected — handles both JSON array and legacy pipe string."""
    if isinstance(val, list):
        return val
    s = str(val).strip() if val is not None else ""
    if not s or s in ("nan", ""):
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Legacy pipe-delimited fallback
    return [x.strip() for x in s.split("|") if x.strip()]


def _coerce_timestamps(ts_series: pd.Series) -> pd.Series:
    # S5-TZ-FIX: pd.to_datetime(..., utc=True) always returns a tz-aware Series,
    # so the old `else` branch was dead code. If ts_series contains tz-aware values
    # in any timezone (e.g. IST UTC+5:30), utc=True correctly converts them to UTC.
    # If ts_series is already tz-naive UTC (as produced by stage4's tz_localize(None)),
    # utc=True re-attaches UTC — also correct. We then strip tz with tz_localize(None)
    # in every case so all callers receive consistent tz-naive UTC Timestamps.
    # This eliminates the phantom 5h30m offset that appeared when IST-aware timestamps
    # were compared against tz-naive UTC values from temporal_df.
    parsed = pd.to_datetime(ts_series, errors="coerce", utc=True)
    return parsed.dt.tz_convert("UTC").dt.tz_localize(None)


def _coerce_burst(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    s = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "✓", "✓ burst", "burst"}
    return s.isin(truthy)


def _is_rising(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.contains("rising", case=False, na=False)


def _parse_cascade_source(val) -> Set[str]:
    if val is None:
        return set()
    s = str(val).strip()
    if s in ("", "—", "nan"):
        return set()
    s = re.sub(r'^[⛓\s]+', '', s).strip()
    return {x.strip() for x in s.split(',') if x.strip()}


def _is_success_message(msg: str, patterns: List[str]) -> bool:
    if not msg:
        return False
    for p in patterns:
        if re.search(p, msg, re.IGNORECASE):
            return True
    return False


def _is_impossible_message(msg: str, patterns: List[str]) -> bool:
    if not msg:
        return False
    for p in patterns:
        if re.search(p, msg, re.IGNORECASE):
            return True
    return False


def _extract_doc_ids(messages: List[str], cfg: Optional[Dict] = None) -> str:
    """
    S5-3 FIX: Extract entity IDs from messages using configurable patterns.
    Previously hardcoded to 'docId=\\d+' (app-specific). Now driven by
    'id_extraction_patterns' in config — a list of (regex, label) tuples.
    Falls back to an empty string when no patterns match, rather than silently
    returning nothing on every non-matching log file.
    """
    if cfg is None:
        cfg = {}
    patterns = cfg.get("id_extraction_patterns", [])
    found: Dict[str, Set[str]] = {}
    for msg in messages:
        if not msg:
            continue
        text = str(msg)
        for pattern, label in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    found.setdefault(label, set()).update(str(m) for m in matches)
            except re.error:
                continue
    if not found:
        return ""
    parts = []
    for label in sorted(found.keys()):
        ids_str = ", ".join(sorted(found[label], key=lambda x: (len(x), x)))
        parts.append(f"{label}=[{ids_str}]" if len(found) > 1 else ids_str)
    return "; ".join(parts)


def _format_lag(seconds: float) -> str:
    if seconds < 60:
        return f"+{seconds:.0f}s"
    return f"+{seconds/60:.0f}m"


def _get_primary_domain(domain_series: pd.Series, priority: List[str]) -> str:
    present = set(domain_series.dropna().astype(str).str.strip().str.lower().unique())
    present.discard("")
    present.discard("nan")
    if not present:
        return "unknown"
    for dom in priority:
        if dom in present:
            return dom
    counts = domain_series.dropna().value_counts()
    if len(counts) > 0:
        return str(counts.index[0])
    return "unknown"


def _adaptive_window(anomaly_ts: pd.Series, cfg: Dict) -> float:
    ts_sorted = anomaly_ts.dropna().sort_values()
    if len(ts_sorted) < 4:
        w = cfg["incident_window_fallback_s"]
        logger.info("Adaptive window: too few anomalous events — using fallback %ds", w)
        return float(w)
    gaps = ts_sorted.diff().dropna().dt.total_seconds()
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        w = cfg["incident_window_fallback_s"]
        logger.info("Adaptive window: all timestamps identical — using fallback %ds", w)
        return float(w)
    median_gap = float(gaps.median())
    window = median_gap * cfg["incident_window_multiplier"]

    # S5-4 FIX: Enforce a minimum window floor so startup bursts (all anomalies
    # within a few seconds) don't shrink the window to near-zero, causing every
    # incident to appear independent. Floor = cascade_min_lag_s × 10 (default 10s).
    min_floor = cfg.get("cascade_min_lag_s", 1) * 10

    # S5-WINDOW-CAP FIX: Enforce a log-span-aware upper cap so that on multi-day
    # logs a large median_gap × multiplier does not produce a window that spans
    # the entire log, causing all clusters to chain into one mega-incident via
    # intermediate union-find links.
    # Cap = min(5% of total log span, 1800s absolute max).
    # For a 2-day (172800s) log: span_cap = 8640s → capped to 1800s. ✓
    # For a 3-hour (10800s) log: span_cap = 540s → allows normal grouping. ✓
    # The configurable key "incident_window_max_s" overrides the 1800s ceiling.
    log_span_s = (ts_sorted.iloc[-1] - ts_sorted.iloc[0]).total_seconds()
    span_cap   = log_span_s * 0.05 if log_span_s > 0 else 1800.0
    abs_max    = float(cfg.get("incident_window_max_s", 1800.0))
    effective_cap = min(span_cap, abs_max)

    window = max(float(min_floor), min(window, effective_cap))

    logger.info(
        "Adaptive window: median_gap=%.1fs × multiplier=%.1f → %.1fs "
        "(floor=%.1fs, span_cap=%.1fs, abs_max=%.1fs) → final=%.1fs",
        median_gap, cfg["incident_window_multiplier"],
        median_gap * cfg["incident_window_multiplier"],
        float(min_floor), span_cap, abs_max, window,
    )
    return window


def _domain_or_service_match(row_a: Dict, row_b: Dict, cfg: Dict) -> bool:
    dom_match = (
        _safe_str(row_a.get("domain")) != "" and
        _safe_str(row_a.get("domain")) == _safe_str(row_b.get("domain"))
    )
    svc_match = (
        _safe_str(row_a.get("_svc")) != "" and
        _safe_str(row_a.get("_svc")) == _safe_str(row_b.get("_svc"))
    )
    if cfg["co_domain_requires_either"]:
        return dom_match or svc_match
    return dom_match and svc_match


# ══════════════════════════════════════════════════════════════════════
# BP-S5-1 — MANIFEST HELPERS
# ══════════════════════════════════════════════════════════════════════

def _manifest_count(cluster_id: str, manifest: Optional[Dict]) -> int:
    """Read verified count from manifest. Returns 0 if absent."""
    if not manifest:
        return 0
    entry = manifest.get("clusters", {}).get(cluster_id, {})
    return int(entry.get("count", 0))


def _manifest_error_count(cluster_id: str, manifest: Optional[Dict]) -> int:
    """Read ERROR+FATAL count from manifest severity_distribution."""
    if not manifest:
        return 0
    entry = manifest.get("clusters", {}).get(cluster_id, {})
    dist = entry.get("severity_distribution", {})
    return int(dist.get("ERROR", 0)) + int(dist.get("FATAL", 0))


def _manifest_services(cluster_id: str, manifest: Optional[Dict]) -> List[str]:
    """Read services list from manifest for a cluster."""
    if not manifest:
        return []
    entry = manifest.get("clusters", {}).get(cluster_id, {})
    return list(entry.get("services", []))


def _incident_counts_from_manifest(
    cluster_ids: List[str],
    manifest: Optional[Dict],
) -> Tuple[int, int]:
    """
    BP-S5-1 — Return (total_event_count, error_event_count) for an
    incident's cluster_ids list, sourced exclusively from the manifest.
    Returns (0, 0) with a warning when manifest is absent.
    """
    if not manifest:
        logger.warning(
            "BP-S5-1: cluster_manifest not provided — event counts will be 0. "
            "Pass cluster_manifest from Stage 2 for accurate counts."
        )
        return 0, 0
    total  = sum(_manifest_count(cid, manifest) for cid in cluster_ids)
    errors = sum(_manifest_error_count(cid, manifest) for cid in cluster_ids)
    return total, errors


def _incident_services_from_manifest(
    cluster_ids: List[str],
    manifest: Optional[Dict],
    fallback_svc_vals: Optional[Set[str]] = None,
) -> List[str]:
    """
    BP-S5-8 — Build explicit services_affected list.
    Primary source: manifest[c]["services"]. Fallback: anomaly_df-derived set.
    """
    svc_set: Set[str] = set()
    if manifest:
        for cid in cluster_ids:
            for svc in _manifest_services(cid, manifest):
                s = svc.strip()
                if s and s not in ("unknown", "", "nan"):
                    svc_set.add(s)
    if not svc_set and fallback_svc_vals:
        svc_set = {s for s in fallback_svc_vals if s and s not in ("unknown", "", "nan")}
    return sorted(svc_set) if svc_set else ["unknown"]


# ══════════════════════════════════════════════════════════════════════
# BP-S5-4 / BP-S5-5 — GROUNDING VALIDATION + DETERMINISTIC FALLBACK
# ══════════════════════════════════════════════════════════════════════

def _validate_llm_narrative(
    narrative_text: str,
    cluster_ids: List[str],
    services_affected: List[str],
) -> bool:
    """
    BP-S5-4 — Narrative is grounded if it cites at least one cluster_id
    OR one non-trivial service name from services_affected.
    """
    if not narrative_text:
        return False
    for cid in cluster_ids:
        if cid and cid in narrative_text:
            return True
    for svc in services_affected:
        svc_clean = svc.strip()
        if svc_clean and svc_clean not in ("unknown", "") and svc_clean in narrative_text:
            return True
    return False


def _deterministic_narrative_fallback(
    incident_id: str,
    cluster_ids: List[str],
    services_affected: List[str],
    peak_anomaly_level: str,
    start_ts: Any,
    end_ts: Any,
    error_trigger: str = "",
    what_happened: str = "",   # accepted but intentionally NOT echoed (S5-5)
) -> Dict[str, str]:
    """
    BP-S5-5 v2 (S5-5) — Guaranteed-grounded fallback that does NOT echo what_happened.

    Produces distinct root_cause_summary and recommended_action text using:
      - service name, anomaly level, temporal span, cluster count
      - error_trigger for the primary signal line
      - severity-appropriate remediation verb

    Used when: (a) LLM gate fails, (b) LLM response is fully stripped by
    the grounding check, or (c) Anthropic package is unavailable.
    """
    svc_str   = ", ".join(services_affected[:3])
    start_str = str(start_ts)[:16] if pd.notna(start_ts) else "unknown time"
    end_str   = str(end_ts)[:16]   if pd.notna(end_ts)   else "unknown time"
    n_cl      = len(cluster_ids)
    trigger   = error_trigger[:80] if error_trigger else ""

    verb = {
        "CRITICAL": "immediately escalate and investigate",
        "HIGH"    : "investigate promptly",
        "MEDIUM"  : "review",
        "LOW"     : "monitor",
    }.get(peak_anomaly_level, "investigate")

    if trigger:
        root_cause_summary = (
            f"Service {svc_str} reported a {peak_anomaly_level}-severity"
            f" condition between {start_str} and {end_str} across {n_cl}"
            f" event cluster(s). The primary signal was: {trigger}."
        )
    else:
        root_cause_summary = (
            f"An elevated error rate was observed in {svc_str} between"
            f" {start_str} and {end_str}. {n_cl} anomalous cluster(s) were"
            f" identified, indicating abnormal behaviour in this service."
        )

    recommended_action = (
        f"On-call engineers should {verb} {svc_str}. Review deployment"
        f" history, dependency health, and error logs for the period"
        f" {start_str} to {end_str}. Check for config changes or upstream"
        f" failures that may have triggered this condition."
    )

    return {
        "root_cause_summary": root_cause_summary,
        "recommended_action": recommended_action,
    }




# ══════════════════════════════════════════════════════════════════════
# S5-ML-3/5 — FAISS HISTORICAL INCIDENT SIMILARITY
# ══════════════════════════════════════════════════════════════════════

def _incident_text_for_embedding(row: Dict) -> str:
    """
    Build a short canonical text representation of an incident for
    embedding. Uses only fields that are always present.
    """
    parts = [
        f"severity:{row.get('incident_severity', '')}",
        f"domain:{row.get('primary_domain', '')}",
        f"root:{row.get('root_cause_service', '')}",
        f"trigger:{str(row.get('error_trigger', ''))[:80]}",
        f"what:{str(row.get('what_happened', ''))[:100]}",
    ]
    return " ".join(p for p in parts if p.split(":", 1)[-1].strip())


def _tfidf_embed(text: str, vocab: Dict[str, int], dim: int = 128) -> np.ndarray:
    """
    Lightweight deterministic TF-IDF-style embedding — used when neither
    sentence-transformers nor faiss is available.  Returns a unit-normed
    float32 vector of length `dim`.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is None:
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            vocab[tok] = idx
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _embed_incident_text(text: str, cfg: Dict) -> Optional[np.ndarray]:
    """
    Embed a short incident text string. Three-tier fallback:
      1. sentence-transformers (all-MiniLM-L6-v2) — if installed.
      2. sklearn TF-IDF hashing — if sklearn installed.
      3. Internal lightweight TF-IDF (_tfidf_embed) — always available.

    Returns a float32 unit-normed 1-D numpy array, or None on hard error.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model_name = "all-MiniLM-L6-v2"
        # Cache the model on the module for reuse
        if not hasattr(_embed_incident_text, "_st_model"):
            _embed_incident_text._st_model = SentenceTransformer(_model_name)
        emb = _embed_incident_text._st_model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        )
        return np.array(emb, dtype=np.float32)
    except ImportError:
        pass
    except Exception as _e:
        logger.debug("_embed_incident_text: SentenceTransformer failed (%s), trying TF-IDF", _e)

    # Tier 2: sklearn HashingVectorizer
    try:
        from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore
        if not hasattr(_embed_incident_text, "_hv"):
            _embed_incident_text._hv = HashingVectorizer(
                n_features=128, norm="l2", alternate_sign=False
            )
        mat = _embed_incident_text._hv.transform([text])
        return np.array(mat.toarray()[0], dtype=np.float32)
    except ImportError:
        pass
    except Exception as _e:
        logger.debug("_embed_incident_text: HashingVectorizer failed (%s), using internal TF-IDF", _e)

    # Tier 3: internal lightweight TF-IDF
    if not hasattr(_embed_incident_text, "_vocab"):
        _embed_incident_text._vocab = {}
    return _tfidf_embed(text, _embed_incident_text._vocab)


def _faiss_index_incident(incident_row: Dict, cfg: Dict) -> None:
    """
    S5-ML-3 — Add one incident to the in-memory FAISS index so that future
    incidents can retrieve it as a similar past event.  Also persists the
    index to faiss_index_path if configured.
    """
    global _FAISS_INDEX, _FAISS_INCIDENT_STORE

    if not _FAISS_AVAILABLE:
        return

    import faiss  # type: ignore

    text = _incident_text_for_embedding(incident_row)
    emb  = _embed_incident_text(text, cfg)
    if emb is None:
        return

    emb = emb.astype(np.float32).reshape(1, -1)
    dim = emb.shape[1]

    if _FAISS_INDEX is None or _FAISS_INDEX.d != dim:
        _FAISS_INDEX = faiss.IndexFlatIP(dim)
        logger.info("S5-ML-3: FAISS IndexFlatIP initialised (dim=%d)", dim)

    faiss.normalize_L2(emb)
    _FAISS_INDEX.add(emb)
    _FAISS_INCIDENT_STORE.append(incident_row)

    # Persist if path configured
    idx_path = cfg.get("faiss_index_path")
    if idx_path:
        try:
            faiss.write_index(_FAISS_INDEX, str(idx_path))
            store_path = str(idx_path) + ".store.pkl"
            with open(store_path, "wb") as fh:
                pickle.dump(_FAISS_INCIDENT_STORE, fh)
        except Exception as _e:
            logger.warning("S5-ML-3: failed to persist FAISS index: %s", _e)


def _faiss_load_index(cfg: Dict) -> None:
    """Load a previously persisted FAISS index from disk if configured."""
    global _FAISS_INDEX, _FAISS_INCIDENT_STORE

    if not _FAISS_AVAILABLE:
        return

    import faiss  # type: ignore

    idx_path = cfg.get("faiss_index_path")
    if not idx_path:
        return
    idx_path = Path(idx_path)
    store_path = Path(str(idx_path) + ".store.pkl")
    if idx_path.exists() and store_path.exists():
        try:
            _FAISS_INDEX = faiss.read_index(str(idx_path))
            with open(store_path, "rb") as fh:
                _FAISS_INCIDENT_STORE = pickle.load(fh)
            logger.info(
                "S5-ML-3: loaded FAISS index from %s (%d entries)",
                idx_path, len(_FAISS_INCIDENT_STORE),
            )
        except Exception as _e:
            logger.warning("S5-ML-3: failed to load FAISS index from %s: %s", idx_path, _e)


def _faiss_search_similar(
    query_row: Dict,
    cfg: Dict,
    exclude_incident_id: Optional[str] = None,
) -> List[Dict]:
    """
    S5-ML-3 / S5-ML-5 — Retrieve top-k similar past incidents from FAISS.

    Hub suppression (S5-ML-5): any stored incident retrieved in >20% of
    all queries so far is down-ranked and excluded.

    Returns a list of dicts with keys:
        incident_id, similarity, primary_domain, root_cause_service,
        what_happened, recommended_action, suppressed (bool)
    """
    global _FAISS_RETRIEVAL_COUNTS, _FAISS_TOTAL_QUERIES

    if not _FAISS_AVAILABLE or _FAISS_INDEX is None or len(_FAISS_INCIDENT_STORE) == 0:
        return []

    import faiss  # type: ignore

    top_k        = int(cfg.get("faiss_top_k", 3))
    min_score    = float(cfg.get("faiss_min_score", 0.50))
    hub_fraction = float(cfg.get("faiss_hub_fraction", 0.20))

    text = _incident_text_for_embedding(query_row)
    emb  = _embed_incident_text(text, cfg)
    if emb is None:
        return []

    emb = emb.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(emb)

    k_search = min(top_k * 3, _FAISS_INDEX.ntotal)  # over-fetch for hub filtering
    scores, indices = _FAISS_INDEX.search(emb, k_search)

    _FAISS_TOTAL_QUERIES += 1
    hub_threshold = max(1, int(_FAISS_TOTAL_QUERIES * hub_fraction))

    results: List[Dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_FAISS_INCIDENT_STORE):
            continue
        if float(score) < min_score:
            continue

        stored = _FAISS_INCIDENT_STORE[idx]
        if exclude_incident_id and stored.get("incident_id") == exclude_incident_id:
            continue

        # Hub suppression (S5-ML-5)
        retrieval_count = _FAISS_RETRIEVAL_COUNTS.get(int(idx), 0)
        is_hub = retrieval_count >= hub_threshold and _FAISS_TOTAL_QUERIES > 5
        if is_hub:
            logger.debug(
                "S5-ML-5: hub suppressed FAISS entry %d (retrieved %d / %d queries)",
                idx, retrieval_count, _FAISS_TOTAL_QUERIES,
            )
            continue

        _FAISS_RETRIEVAL_COUNTS[int(idx)] = retrieval_count + 1
        results.append({
            "incident_id"         : stored.get("incident_id", ""),
            "similarity"          : round(float(score), 4),
            "primary_domain"      : stored.get("primary_domain", ""),
            "root_cause_service"  : stored.get("root_cause_service", ""),
            "what_happened"       : stored.get("what_happened", ""),
            "recommended_action"  : stored.get("recommended_action", ""),
        })
        if len(results) >= top_k:
            break

    return results


# ══════════════════════════════════════════════════════════════════════
# ACTION (Section 4.2) — LLM NARRATIVE RESPONSE CACHE
# ══════════════════════════════════════════════════════════════════════
# Module-level dict — lives for one pipeline run (reset on process restart).
# Identical incidents (same cascade_chain, error_trigger, primary_domain,
# anomaly_label, services) reuse the cached narrative without a second API call.

_NARRATIVE_CACHE: Dict[str, Tuple[str, str, str]] = {}


def _narrative_cache_key(
    incident_row: Dict,
    cluster_ids: List[str],
    services_affected: List[str],
) -> str:
    key_data = {
        "cascade_chain"    : incident_row.get("cascade_chain", ""),
        "error_trigger"    : str(incident_row.get("error_trigger", ""))[:120],
        "primary_domain"   : incident_row.get("primary_domain", ""),
        "services"         : sorted(services_affected[:5]),
        "incident_severity": incident_row.get("incident_severity", ""),  # NOTE: incident_row uses incident_severity, not anomaly_label
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# ACTION (Section 4.1) — EXPONENTIAL BACKOFF FOR LLM RETRIES
# ══════════════════════════════════════════════════════════════════════
# Wraps any LLM callable with up to 2 retries (delays: 2 s, 4 s).
# Only triggers deterministic fallback after all retries are exhausted.



def _call_with_backoff(fn, *args, max_retries: int = 2, base_delay: float = 2.0, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on exception.
    Raises the last exception if all retries are exhausted.
    Delays: 2 s, 4 s (base_delay=2, exponent doubles per attempt).
    """
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as _e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt + 1, max_retries + 1, _e, delay,
            )
            _time.sleep(delay)


# ══════════════════════════════════════════════════════════════════════
# S5-ML-1/2/4 — LLM NARRATIVE GENERATION + CAUSAL GROUNDING VERIFICATION
# ══════════════════════════════════════════════════════════════════════

_CAUSAL_VERB_PAT = re.compile(
    r"\b(caused|led to|triggered|resulted in|due to)\b",
    re.IGNORECASE,
)

# S5-6: Descriptive-only verbs — sentences using these are NOT causal claims
# and must never be stripped even when they lack a cluster/service citation.
_DESCRIPTIVE_ONLY_PAT = re.compile(
    r"\b(experienced|showed|detected|observed|reported|indicated)\b",
    re.IGNORECASE,
)


def _is_ungrounded_causal(
    sentence: str,
    cluster_ids: List[str],
    services_affected: List[str],
) -> bool:
    """
    S5-6 — Return True only when a sentence makes an explicit causal claim
    (contains a causal verb from _CAUSAL_VERB_PAT) AND is not grounded by a
    cluster_id or service name citation.

    Sentences that use only descriptive verbs ("experienced", "observed", etc.)
    are always kept regardless of grounding — they describe state, not causation.
    """
    if not _CAUSAL_VERB_PAT.search(sentence):
        return False   # no causal verb at all — keep unconditionally

    if _DESCRIPTIVE_ONLY_PAT.search(sentence):
        return False   # softened / descriptive language — keep

    # Has an explicit causal verb — require at least one grounding anchor
    for cid in cluster_ids:
        if cid and cid in sentence:
            return False
    for svc in services_affected:
        svc_clean = re.sub(r"[^a-z0-9]", "", str(svc).lower())
        if svc_clean and svc_clean in sentence.lower():
            return False

    return True   # ungrounded explicit causal claim — strip


_CAUSAL_RE = _CAUSAL_VERB_PAT   # kept as module alias for any external callers


def _strip_ungrounded_causal_sentences(
    text: str,
    cluster_ids: List[str],
    services_affected: List[str],
) -> Tuple[str, bool]:
    """
    S5-ML-2 / S5-6 — Causal-claim grounding verification (relaxed).

    Split text into sentences. A sentence is stripped only if:
      (a) it contains an explicit causal verb (caused, led to, triggered,
          resulted in, due to) AND
      (b) it does NOT use a descriptive-only verb (experienced, observed, etc.)
          AND
      (c) it does NOT cite a cluster_id OR non-trivial service name.

    Descriptive sentences that make no causal claim are always retained,
    even when they lack explicit cluster/service citations.

    Returns (cleaned_text, any_stripped).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: List[str] = []
    any_stripped = False

    anchor_terms = set()
    for cid in cluster_ids:
        if cid:
            anchor_terms.add(cid)
    for svc in services_affected:
        svc_c = svc.strip()
        if svc_c and svc_c not in ("unknown", ""):
            anchor_terms.add(svc_c)

    for sentence in sentences:
        if not sentence.strip():
            continue
        if _is_ungrounded_causal(sentence, cluster_ids, services_affected):
            logger.debug("S5-ML-2: stripped ungrounded causal sentence: %s", sentence[:80])
            any_stripped = True
            continue
        kept.append(sentence)

    return " ".join(kept).strip(), any_stripped


def _llm_confidence_gate(row: Dict, cfg: Dict) -> Tuple[bool, str]:
    """
    S5-ML-4 — Evaluate whether an incident meets the bar for LLM narrative.

    Availability check is provider-aware (updated for Ollama integration):
      - provider="ollama"    → gate passes if _OLLAMA_AVAILABLE is True
      - provider="anthropic" → gate passes if _ANTHROPIC_AVAILABLE is True

    Returns (should_call_llm: bool, reason: str).
    """
    if not cfg.get("llm_narrative_enabled", True):
        return False, "llm_narrative_enabled=False"

    # Provider-aware availability check
    provider = cfg.get("llm_provider", "ollama")
    if provider == "ollama":
        if not _OLLAMA_AVAILABLE:
            return False, "ollama_not_available"
    else:
        if not _ANTHROPIC_AVAILABLE or _ANTHROPIC_CLIENT is None:
            return False, "anthropic_not_available"

    min_clusters = int(cfg.get("llm_min_clusters", 2))
    if int(row.get("n_clusters", 0)) < min_clusters:
        return False, f"n_clusters<{min_clusters}"

    min_score = float(cfg.get("llm_min_anomaly_score", 0.60))
    top_score = row.get("top_cluster_score", float("nan"))
    try:
        if float(top_score) < min_score:
            return False, f"top_cluster_score<{min_score}"
    except (TypeError, ValueError):
        return False, "top_cluster_score_missing"

    if cfg.get("llm_require_cascade", True):
        chain = str(row.get("cascade_chain", "")).strip()
        if not chain or " → " not in chain:
            return False, "cascade_chain_empty_or_single_hop"

    excluded_domains = set(cfg.get("llm_excluded_domains", {"other", "unknown"}))
    domain = str(row.get("primary_domain", "other")).strip().lower()
    if domain in excluded_domains:
        return False, f"domain_excluded({domain})"

    return True, "gate_passed"


def _call_llm_narrative(
    incident_row: Dict,
    cluster_ids: List[str],
    services_affected: List[str],
    similar_past: List[Dict],
    cfg: Dict,
) -> Tuple[str, str, str]:
    """
    S5-ML-1 — Generate a grounded root-cause narrative for a single incident.

    Provider-aware: reads cfg['llm_provider'] to choose between Ollama
    (default) and Anthropic (fallback / production path).

    Returns (root_cause_summary, recommended_action, narrative_source).
    narrative_source is one of:
        'llm_grounded'           — LLM output fully grounded
        'llm_fallback_stripped'  — LLM output had ungrounded causal sentences
                                   stripped; remainder kept
        'deterministic_fallback' — LLM failed / unrecoverable; fallback used
    """
    # ── ACTION 4.2: check narrative cache ────────────────────────────
    _cache_key = _narrative_cache_key(incident_row, cluster_ids, services_affected)
    if _cache_key in _NARRATIVE_CACHE:
        logger.info(
            "S5-ML-1: narrative cache hit for %s — reusing cached result",
            incident_row.get("incident_id"),
        )
        return _NARRATIVE_CACHE[_cache_key]

    # Build similar-past context block
    similar_block = ""
    if similar_past:
        lines = []
        for sp in similar_past[:3]:
            lines.append(
                f"  - Past incident {sp['incident_id']} "
                f"(sim={sp['similarity']:.2f}): "
                f"root={sp['root_cause_service']}, "
                f"domain={sp['primary_domain']}, "
                f"what={str(sp['what_happened'])[:80]}"
            )
        similar_block = "\nSimilar past incidents:\n" + "\n".join(lines)

    cluster_sample = ", ".join(cluster_ids[:5]) + ("..." if len(cluster_ids) > 5 else "")
    svc_sample     = ", ".join(services_affected[:5])

    # WARN 2 FIX: root_cause_hint is conditional — if empty, lead with
    # cascade_chain and error_trigger as primary evidence instead.
    # cascade_chain is ALWAYS included in the prompt regardless.
    _root_cause_hint_val = str(incident_row.get('root_cause_hint', '')).strip()
    if _root_cause_hint_val:
        _primary_evidence_instruction = (
            "Use root_cause_hint as your primary evidence. Do NOT restate what_happened."
        )
        _hint_line = f"  root_cause_hint  : {_root_cause_hint_val[:200]}"
    else:
        _primary_evidence_instruction = (
            "Use cascade_chain and error_trigger as your primary evidence. "
            "Do NOT restate what_happened."
        )
        _hint_line = ""  # omit the hint line entirely when empty

    prompt = f"""You are an SRE root-cause analysis assistant. You must respond in EXACTLY the format shown at the end — no preamble, no explanation, no markdown, no extra text.

Rules:
- ROOT_CAUSE: 2-4 sentences. Explain WHY the anomaly occurred and what the operational impact is. Name the specific service or cluster. {_primary_evidence_instruction}
- RECOMMENDED_ACTION: 2-4 sentences. Concrete remediation steps for the on-call engineer. Be specific to the service and error type. Do NOT repeat the root cause.
- Write ONLY the two labelled sections. Stop after RECOMMENDED_ACTION.

Incident data:
  incident_id      : {incident_row.get('incident_id', '')}
  severity         : {incident_row.get('incident_severity', '')}
  primary_domain   : {incident_row.get('primary_domain', '')}
  services_affected: {svc_sample}
  cluster_ids      : {cluster_sample}
  cascade_chain    : {incident_row.get('cascade_chain', '')}
  error_trigger    : {str(incident_row.get('error_trigger', ''))[:120]}
{(_hint_line + chr(10)) if _hint_line else ''}  what_happened    : {str(incident_row.get('what_happened', ''))[:120]}
  top_cluster_score: {incident_row.get('top_cluster_score', '')}
  duration_minutes : {incident_row.get('duration_minutes', '')}
  recurrence       : {incident_row.get('recurrence_flag', False)}
{similar_block}

ROOT_CAUSE: <2-4 sentence root cause>
RECOMMENDED_ACTION: <2-4 sentence remediation>"""

    provider = cfg.get("llm_provider", "ollama")

    # ── Ollama path ───────────────────────────────────────────────────
    if provider == "ollama":
        try:
            import requests as _requests
            ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
            ollama_model    = cfg.get("ollama_model", "phi4:14b")

            # phi4:14b ignores format instructions inside the prompt body when
            # they conflict with its instruction-tuning defaults (it prefers
            # markdown headers and numbered lists). Using the `system` field
            # as a strict mode enforcer is more reliable than repeating rules
            # in the prompt — the system role has higher weight in phi4's RLHF.
            _system_prompt = (
                "You are a terse SRE assistant. "
                "You MUST respond using ONLY these two labelled lines and nothing else. "
                "No markdown. No headers. No numbered lists. No preamble. No explanation. "
                "Output format — respond with exactly this structure:\n"
                "ROOT_CAUSE: <your 2-4 sentence root cause here>\n"
                "RECOMMENDED_ACTION: <your 2-4 sentence action here>\n"
                "Stop immediately after RECOMMENDED_ACTION. Do not write anything else."
            )

            def _ollama_call():
                resp = _requests.post(
                    f"{ollama_base_url}/api/generate",
                    json={
                        "model":  ollama_model,
                        "system": _system_prompt,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": int(cfg.get("llm_max_tokens", 700)),
                            "temperature": float(cfg.get("llm_temperature", 0)),
                            # Stop tokens: prevent phi4 from continuing past the
                            # two structured sections or starting markdown blocks
                            "stop": ["\n\n\n", "###", "---", "Note:", "Additionally,"],
                        },
                    },
                    # phi4:14b cold-start can take 30-90s on first call;
                    # 60s ceiling is sufficient for warm model inference (~5-15s).
                    # If you see frequent timeouts on cold start, raise to 90s.
                    timeout=60,
                )
                resp.raise_for_status()
                resp_json    = resp.json()
                raw          = resp_json.get("response", "").strip()
                done_reason  = resp_json.get("done_reason", "")
                if done_reason == "length":
                    logger.warning(
                        "S5-ML-1 [ollama]: response for %s hit num_predict limit "
                        "(%d tokens) — consider raising llm_max_tokens if truncated",
                        incident_row.get("incident_id"),
                        cfg.get("llm_max_tokens", 700),
                    )
                return raw

            raw_text = _call_with_backoff(_ollama_call, max_retries=1)
        except Exception as _e:
            logger.warning(
                "S5-ML-1 [ollama]: API call failed (all retries exhausted) for %s: %s",
                incident_row.get("incident_id"), _e,
            )
            return "", "", "deterministic_fallback"

    # ── Anthropic path (fallback / production) ────────────────────────
    else:
        try:
            def _anthropic_call():
                response = _ANTHROPIC_CLIENT.messages.create(
                    model       = cfg.get("llm_model", "claude-sonnet-4-20250514"),
                    max_tokens  = int(cfg.get("llm_max_tokens", 700)),
                    temperature = float(cfg.get("llm_temperature", 0)),
                    messages    = [{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip() if response.content else ""

            raw_text = _call_with_backoff(_anthropic_call)
        except Exception as _e:
            logger.warning(
                "S5-ML-1 [anthropic]: API call failed (all retries exhausted) for %s: %s",
                incident_row.get("incident_id"), _e,
            )
            return "", "", "deterministic_fallback"

    # ── Parse structured response (shared by both paths) ─────────────

    # WARN 1 FIX: truncation-detection log — if both markers are absent the
    # response was almost certainly cut off silently at llm_max_tokens.
    _has_rc  = bool(re.search(r"ROOT_CAUSE:",        raw_text, re.IGNORECASE))
    _has_ra  = bool(re.search(r"RECOMMENDED_ACTION:", raw_text, re.IGNORECASE))
    if not _has_rc or not _has_ra:
        logger.warning(
            "S5-ML-1 [%s]: LLM response for %s appears truncated or malformed "
            "(ROOT_CAUSE present=%s, RECOMMENDED_ACTION present=%s). "
            "Consider raising llm_max_tokens if this recurs.",
            provider, incident_row.get("incident_id"), _has_rc, _has_ra,
        )
    root_cause_text  = ""
    recommended_text = ""

    rc_match  = re.search(r"ROOT_CAUSE:\s*(.+?)(?=RECOMMENDED_ACTION:|$)", raw_text, re.DOTALL | re.IGNORECASE)
    ra_match  = re.search(r"RECOMMENDED_ACTION:\s*(.+?)$",                 raw_text, re.DOTALL | re.IGNORECASE)

    if rc_match:
        root_cause_text = rc_match.group(1).strip()
    if ra_match:
        recommended_text = ra_match.group(1).strip()

    if not root_cause_text and not recommended_text:
        logger.warning("S5-ML-1: could not parse LLM response for %s", incident_row.get("incident_id"))
        return "", "", "deterministic_fallback"

    # S5-ML-2: causal-claim grounding verification
    rc_clean,  rc_stripped  = _strip_ungrounded_causal_sentences(root_cause_text,  cluster_ids, services_affected)
    ra_clean,  ra_stripped  = _strip_ungrounded_causal_sentences(recommended_text, cluster_ids, services_affected)

    any_stripped = rc_stripped or ra_stripped
    narrative_source = "llm_fallback_stripped" if any_stripped else "llm_grounded"

    # If stripping leaves the root_cause empty, fall back entirely
    if not rc_clean.strip():
        logger.info("S5-ML-2: root_cause fully stripped for %s — using deterministic fallback", incident_row.get("incident_id"))
        return "", "", "deterministic_fallback"

    logger.info(
        "S5-ML-1 [%s]: LLM narrative generated for %s (source=%s, rc_stripped=%s, ra_stripped=%s)",
        provider, incident_row.get("incident_id"), narrative_source, rc_stripped, ra_stripped,
    )
    # ACTION 4.2: store in narrative cache so duplicate incidents skip the API call
    _NARRATIVE_CACHE[_cache_key] = (rc_clean, ra_clean, narrative_source)
    return rc_clean, ra_clean, narrative_source


# ══════════════════════════════════════════════════════════════════════
# FIX-4 — ERROR CLASS BUCKETING
# ══════════════════════════════════════════════════════════════════════

def _classify_error_class(error_trigger: str, cfg: Dict) -> str:
    if not error_trigger:
        return "GENERIC_ERROR"
    for pattern, label in cfg.get("error_class_patterns", []):
        try:
            if re.search(pattern, error_trigger, re.IGNORECASE):
                return label
        except re.error:
            continue
    return "GENERIC_ERROR"


# ══════════════════════════════════════════════════════════════════════
# FIX-3 — DOMAIN COLUMN RESOLUTION
# ══════════════════════════════════════════════════════════════════════

def _resolve_domain_col(df: pd.DataFrame, col: Dict, cfg: Dict) -> Optional[str]:
    mapped = col.get("domain")
    if mapped and mapped in df.columns:
        return mapped
    for candidate in cfg.get("domain_col_candidates", []):
        if candidate in df.columns:
            return candidate
    # S5-5B: if we reach here, the primary "domain" column was not found.
    # This should not happen after Stage 2 consolidation.
    logger.warning(
        "S5: 'domain' column not found in DataFrame — "
        "Stage 2 may not have run correctly."
    )
    for c in df.columns:
        if "domain" in c.lower():
            return c
    return None


def _ensure_domain_column(
    incident_events_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    col: Dict,
    cfg: Dict,
) -> pd.DataFrame:
    """
    FIX-3C — Guarantee that incident_events_df has a populated 'domain'
    column. After Stage 2 consolidation, Source 1 should always succeed.
      1. domain column already present and non-empty → use it (normal path)
      2. anomaly_df has domain column → map via eid_col (WARNING: pipeline issue)
      3. last resort → fill with 'other' (ERROR: pipeline integrity failure)
    """
    eid_col = col["event_id"]
    dom_col = _resolve_domain_col(incident_events_df, col, cfg)

    if dom_col and dom_col != "domain":
        incident_events_df = incident_events_df.copy()
        incident_events_df["domain"] = incident_events_df[dom_col]
        dom_col = "domain"

    if dom_col == "domain" and "domain" in incident_events_df.columns:
        non_empty = incident_events_df["domain"].dropna().astype(str).str.strip()
        non_empty = non_empty[~non_empty.isin(["", "nan", "unknown"])]
        if len(non_empty) > 0:
            logger.debug(
                "FIX-3: domain column already populated (%d/%d rows non-empty)",
                len(non_empty), len(incident_events_df),
            )
            return incident_events_df

    # Source 2: anomaly_df domain — Source 1 failed, which should not happen
    # after Stage 2 consolidation.
    logger.warning(
        "FIX-3: domain column missing or empty in incident_events_df — "
        "this should not happen after Stage 2 consolidation. "
        "Attempting anomaly_df fallback."
    )
    anom_dom_col = _resolve_domain_col(anomaly_df, col, cfg)
    if anom_dom_col and eid_col in anomaly_df.columns and eid_col in incident_events_df.columns:
        domain_map = (
            anomaly_df[[eid_col, anom_dom_col]]
            .drop_duplicates(subset=eid_col)
            .set_index(eid_col)[anom_dom_col]
        )
        incident_events_df = incident_events_df.copy()
        mapped = incident_events_df[eid_col].map(domain_map)
        filled_count = mapped.notna().sum()
        if filled_count > 0:
            incident_events_df["domain"] = mapped.fillna(
                incident_events_df.get("domain", pd.Series("other", index=incident_events_df.index))
            )
            logger.debug(
                "FIX-3: domain populated from anomaly_df (%d/%d rows filled)",
                filled_count, len(incident_events_df),
            )
            return incident_events_df

    # Last resort: both Source 1 and Source 2 failed — pipeline integrity error.
    logger.error(
        "FIX-3: domain could not be resolved for incident_events_df — "
        "pipeline integrity error. Filling with 'other' as last resort."
    )
    incident_events_df = incident_events_df.copy()
    incident_events_df["domain"] = "other"
    return incident_events_df


# ══════════════════════════════════════════════════════════════════════
# NEW-B — ERROR TRIGGER EXTRACTOR
# ══════════════════════════════════════════════════════════════════════

def _extract_error_trigger(messages: List[str], cfg: Dict) -> str:
    patterns = cfg.get("error_trigger_patterns", [])
    combined = " | ".join(m for m in messages if m and str(m).strip())

    for pattern, group_or_idx in patterns:
        try:
            m = re.search(pattern, combined, re.IGNORECASE | re.DOTALL)
            if not m:
                continue
            if isinstance(group_or_idx, int) and group_or_idx > 0:
                try:
                    raw = m.group(group_or_idx)
                except IndexError:
                    raw = m.group(0)
            else:
                raw = m.group(0)
            raw = re.sub(r"^(severity:\s*'(ERROR|FATAL)'\s*)", '', raw, flags=re.IGNORECASE)
            raw = re.sub(r'\s+', ' ', raw).strip()
            if raw:
                return raw[:60].rstrip(' ,;')
        except re.error:
            continue

    for msg in messages:
        snippet = _safe_str(msg)[:60].strip()
        if snippet:
            return snippet
    return ""


# ══════════════════════════════════════════════════════════════════════
# NEW-A — WHAT HAPPENED GENERATOR
# ══════════════════════════════════════════════════════════════════════

def _generate_what_happened(
    messages: List[str],
    root_svc: str,
    n_services: int,
    any_burst: bool,
    recurrence: bool,
    security: bool,
    n_clusters: int,
    cfg: Dict,
) -> str:
    signatures = cfg.get("what_happened_signatures", [])
    combined   = " | ".join(m for m in messages if m and str(m).strip())

    base_phrase = ""
    for pattern, phrase_template in signatures:
        try:
            if re.search(pattern, combined, re.IGNORECASE | re.DOTALL):
                base_phrase = phrase_template.replace("{svc}", root_svc or "service")
                break
        except re.error:
            continue

    if not base_phrase:
        if n_clusters > 5:
            base_phrase = f"Multiple anomalies detected in {root_svc or 'service'}"
        else:
            base_phrase = f"Anomalous behaviour in {root_svc or 'unknown service'}"

    qualifier = ""
    if n_services > 3:
        qualifier = f"cascading across {n_services} services"
    elif any_burst:
        qualifier = "during traffic surge"
    elif security:
        qualifier = "auth failure detected"

    summary = f"{base_phrase} — {qualifier}" if qualifier else base_phrase

    if recurrence:
        summary = f"[Recurring] {summary}"

    words = summary.split()
    if len(words) > 16:
        summary = " ".join(words[:16]) + "…"

    return summary


# ══════════════════════════════════════════════════════════════════════
# NEW-D — RECOMMENDED ACTION (v4)
# ══════════════════════════════════════════════════════════════════════

def _recommended_action_v4(row: pd.Series, cfg: Dict) -> str:
    remediation_map = cfg.get("remediation_map", [])
    default_action  = cfg.get("default_action", "Investigate affected services.")

    severity    = _safe_str(row.get("incident_severity", "HIGH"))
    root_svc    = _safe_str(row.get("root_cause_service", ""))
    cascade     = _safe_str(row.get("cascade_chain", ""))
    burst_any   = bool(row.get("any_burst", False))
    rising_any  = bool(row.get("any_rising_trend", False))
    impossible  = bool(row.get("has_impossible_attempt", False))
    security    = bool(row.get("has_security_signal", False))
    n_services  = int(row.get("n_services_affected", 1))
    recurrence  = bool(row.get("recurrence_flag", False))
    doc_ids     = _safe_str(row.get("affected_document_ids", ""))
    duration    = row.get("duration_minutes", None)
    error_trig  = _safe_str(row.get("error_trigger", ""))
    what        = _safe_str(row.get("what_happened", ""))

    try:
        dur_val = float(duration)
    except (TypeError, ValueError):
        dur_val = None

    self_healed = (dur_val is not None and dur_val < 1.0 and not impossible)

    if severity == "CRITICAL" and not self_healed:
        tier = "P1 — page on-call immediately."
    elif severity == "CRITICAL" and self_healed:
        tier = "P2 — self-resolved in under a minute; verify no secondary impact."
    elif severity == "HIGH":
        tier = "P2 — investigate within the hour."
    else:
        tier = "P3 — schedule investigation; monitor for recurrence."

    start_clause = (
        f"Start at '{root_svc}'."
        if root_svc and root_svc not in ("unknown", "co-originating", "")
        else ""
    )
    part1 = " ".join(filter(None, [tier, start_clause]))

    lookup_text = " ".join(filter(None, [error_trig, what]))
    part2 = default_action
    for pattern, remedy in remediation_map:
        try:
            if re.search(pattern, lookup_text, re.IGNORECASE):
                part2 = remedy
                break
        except re.error:
            continue

    context_parts: List[str] = []
    seen: Set[str] = set()

    def _add(sentence: str) -> None:
        key = sentence[:40]
        if key not in seen:
            seen.add(key)
            context_parts.append(sentence)

    if n_services > 1:
        _add(f"This incident spans {n_services} services — a cascading failure is likely.")
    if cascade and " → " in cascade:
        _add(f"Observed propagation path: {cascade}.")
    if burst_any:
        _add("At least one event showed a traffic burst — check for sudden load spikes or retry storms.")
    if rising_any:
        _add("Error rate was still rising at end of the log window — the incident may not be resolved.")
    if impossible:
        _add("Impossible-operation patterns detected — review for missing resources, misconfiguration, or stuck retry loops.")
    if security:
        _add("Security/authentication signals present — verify no unauthorized access occurred and check session integrity.")
    if recurrence:
        _add(f"⚠️ Recurrence: '{root_svc}' has been the root cause in earlier incidents this window — treat as a systemic issue, not a one-off.")
    if doc_ids:
        _add(f"Affected document ID(s): {doc_ids} — cross-reference document processing logs.")

    part3 = " ".join(context_parts)
    return " ".join(filter(None, [part1, part2, part3]))


# ══════════════════════════════════════════════════════════════════════
# BP-S5-3 — CROSS-INCIDENT DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════

def _compute_time_overlap_fraction(
    start_a: Any, end_a: Any, start_b: Any, end_b: Any,
) -> float:
    """Fraction of the shorter window that overlaps with the other window."""
    try:
        if any(pd.isna(v) for v in [start_a, end_a, start_b, end_b]):
            return 0.0
        overlap_start = max(start_a, start_b)
        overlap_end   = min(end_a, end_b)
        if overlap_end <= overlap_start:
            return 0.0
        overlap_s = (overlap_end - overlap_start).total_seconds()
        dur_a     = (end_a - start_a).total_seconds()
        dur_b     = (end_b - start_b).total_seconds()
        shorter   = min(dur_a, dur_b)
        if shorter <= 0:
            return 0.0
        return overlap_s / shorter
    except Exception:
        return 0.0


def _dedup_incidents(
    incidents_df: pd.DataFrame,
    manifest: Optional[Dict],
    cfg: Dict,
) -> pd.DataFrame:
    """
    BP-S5-3 — One-pass cross-incident deduplication.
    Merges pairs where cluster overlap OR time window overlap exceeds threshold.
    Processed in timestamp order; no cascading.
    """
    cluster_overlap_thresh = cfg.get("dedup_cluster_overlap_threshold", 0.50)
    time_overlap_thresh    = cfg.get("dedup_time_overlap_threshold", 0.80)

    if len(incidents_df) <= 1:
        return incidents_df

    df = incidents_df.copy().reset_index(drop=True)

    def _parse_cids(v) -> List[str]:
        if isinstance(v, list):
            return v
        s = _safe_str(v)
        return [x.strip() for x in s.split("|") if x.strip()] if s else []

    merged_into: Dict[int, int] = {}
    n = len(df)

    for i in range(n):
        if i in merged_into:
            continue
        for j in range(i + 1, n):
            if j in merged_into:
                continue

            cids_i = set(_parse_cids(df.at[i, "cluster_ids"]))
            cids_j = set(_parse_cids(df.at[j, "cluster_ids"]))
            if not cids_i or not cids_j:
                continue

            inter_size   = len(cids_i & cids_j)
            min_size     = min(len(cids_i), len(cids_j))
            overlap_frac = inter_size / max(min_size, 1)

            time_frac = _compute_time_overlap_fraction(
                df.at[i, "incident_start"], df.at[i, "incident_end"],
                df.at[j, "incident_start"], df.at[j, "incident_end"],
            )

            if not (overlap_frac > cluster_overlap_thresh or time_frac > time_overlap_thresh):
                continue

            # Merge j into i (i is survivor)
            merged_cids     = sorted(cids_i | cids_j)
            merged_cids_str = "|".join(merged_cids)

            valid_starts = [v for v in [df.at[i, "incident_start"], df.at[j, "incident_start"]] if pd.notna(v)]
            valid_ends   = [v for v in [df.at[i, "incident_end"],   df.at[j, "incident_end"]]   if pd.notna(v)]
            new_start = min(valid_starts) if valid_starts else pd.NaT
            new_end   = max(valid_ends)   if valid_ends   else pd.NaT

            try:
                new_duration = round((new_end - new_start).total_seconds() / 60, 1) \
                    if pd.notna(new_start) and pd.notna(new_end) else float("nan")
            except Exception:
                new_duration = float("nan")

            total_ev, error_ev = _incident_counts_from_manifest(merged_cids, manifest)

            svc_set: Set[str] = set()
            if manifest:
                for cid in merged_cids:
                    svc_set.update(_manifest_services(cid, manifest))
            svc_set.discard("unknown"); svc_set.discard("")
            merged_svcs_str = "|".join(sorted(svc_set)) if svc_set else \
                "|".join(filter(None, set(
                    _parse_services(df.at[i, "services_affected"]) +
                    _parse_services(df.at[j, "services_affected"])
                )))

            sev_rank   = cfg.get("severity_rank", {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1})
            sev_i      = sev_rank.get(df.at[i, "incident_severity"], 0)
            sev_j      = sev_rank.get(df.at[j, "incident_severity"], 0)
            merged_sev = df.at[i, "incident_severity"] if sev_i >= sev_j else df.at[j, "incident_severity"]

            df.at[i, "cluster_ids"]         = merged_cids_str
            df.at[i, "n_clusters"]          = len(merged_cids)
            df.at[i, "services_affected"]   = merged_svcs_str
            df.at[i, "n_services_affected"] = max(len([s for s in _parse_services(merged_svcs_str) if s not in ("unknown", "")]), 1)
            df.at[i, "incident_start"]      = new_start
            df.at[i, "incident_end"]        = new_end
            df.at[i, "duration_minutes"]    = new_duration
            df.at[i, "incident_severity"]   = merged_sev
            if total_ev > 0:
                df.at[i, "total_event_count"] = total_ev
                df.at[i, "error_event_count"] = error_ev
            df.at[i, "any_burst"]              = bool(df.at[i, "any_burst"]) or bool(df.at[j, "any_burst"])
            df.at[i, "any_rising_trend"]       = bool(df.at[i, "any_rising_trend"]) or bool(df.at[j, "any_rising_trend"])
            df.at[i, "has_impossible_attempt"] = bool(df.at[i, "has_impossible_attempt"]) or bool(df.at[j, "has_impossible_attempt"])
            df.at[i, "has_security_signal"]    = bool(df.at[i, "has_security_signal"]) or bool(df.at[j, "has_security_signal"])

            merged_into[j] = i
            logger.info(
                "BP-S5-3 Dedup: merged %s → %s (cluster_overlap=%.2f, time_overlap=%.2f)",
                df.at[j, "incident_id"], df.at[i, "incident_id"],
                overlap_frac, time_frac,
            )

    survivors = [i for i in range(n) if i not in merged_into]
    result    = df.iloc[survivors].copy().reset_index(drop=True)

    merged_n = n - len(result)
    if merged_n > 0:
        logger.info("BP-S5-3 Dedup: %d → %d incidents (%d merged)", n, len(result), merged_n)
    else:
        logger.info("BP-S5-3 Dedup: no incidents merged (all %d distinct)", n)

    return result


# ══════════════════════════════════════════════════════════════════════
# BP-S5-6 — CROSS-STAGE CONSISTENCY ASSERTIONS
# ══════════════════════════════════════════════════════════════════════

def _run_consistency_checks(
    incidents_df: pd.DataFrame,
    manifest: Optional[Dict],
    cfg: Dict,
) -> Tuple[bool, List[str]]:
    """
    BP-S5-6 — Run 5 cross-stage consistency assertions.
    Returns (all_passed, failures_list).
    """
    failures: List[str] = []
    raise_on_error = cfg.get("raise_on_consistency_error", False)

    def _fail(msg: str) -> None:
        failures.append(msg)
        logger.error("BP-S5-6 CONSISTENCY FAIL: %s", msg)
        if raise_on_error:
            raise PipelineConsistencyError(msg)

    def _parse_cids(v) -> List[str]:
        if isinstance(v, list):
            return v
        s = _safe_str(v)
        return [x.strip() for x in s.split("|") if x.strip()] if s else []

    def _parse_svcs(v) -> List[str]:
        return _parse_services(v)

    if len(incidents_df) == 0:
        logger.info("BP-S5-6: no incidents — consistency checks skipped")
        return True, []

    # Assert 1: all cluster_ids exist in manifest
    if manifest:
        manifest_cids = set(manifest.get("clusters", {}).keys())
        all_incident_cids: Set[str] = set()
        for _, row in incidents_df.iterrows():
            all_incident_cids.update(_parse_cids(row.get("cluster_ids", "")))
        unknown_cids = all_incident_cids - manifest_cids
        if unknown_cids:
            _fail(
                f"Assert 1: {len(unknown_cids)} cluster_id(s) in incidents not in manifest: "
                f"{sorted(unknown_cids)[:5]}"
            )
        else:
            logger.info(
                "BP-S5-6 Assert 1 PASS: all %d cluster_id(s) verified in manifest",
                len(all_incident_cids),
            )
    else:
        logger.info("BP-S5-6 Assert 1: skipped (no manifest)")

    # Assert 2: n_clusters == len(cluster_ids)
    a2_failures = []
    for _, row in incidents_df.iterrows():
        cids = _parse_cids(row.get("cluster_ids", ""))
        n    = int(row.get("n_clusters", 0))
        if n != len(cids):
            a2_failures.append(
                f"{row['incident_id']}: n_clusters={n} but len(cluster_ids)={len(cids)}"
            )
    if a2_failures:
        _fail(f"Assert 2: n_clusters mismatch in {len(a2_failures)} incident(s): {a2_failures[:3]}")
    else:
        logger.info("BP-S5-6 Assert 2 PASS: n_clusters == len(cluster_ids) for all %d incidents", len(incidents_df))

    # Assert 3: n_services_affected == len(non-unknown services)
    a3_failures = []
    for _, row in incidents_df.iterrows():
        svcs             = _parse_svcs(row.get("services_affected", ""))
        n                = int(row.get("n_services_affected", 0))
        non_unknown_svcs = [s for s in svcs if s and s not in ("unknown", "")]
        expected         = max(len(non_unknown_svcs), 1)
        if n != expected:
            a3_failures.append(
                f"{row['incident_id']}: n_services_affected={n} but derived={expected}"
            )
    if a3_failures:
        _fail(f"Assert 3: n_services_affected mismatch in {len(a3_failures)} incident(s): {a3_failures[:3]}")
    else:
        logger.info("BP-S5-6 Assert 3 PASS: n_services_affected consistent for all incidents")

    # Assert 4: incident_start <= incident_end
    a4_failures = []
    for _, row in incidents_df.iterrows():
        s = row.get("incident_start")
        e = row.get("incident_end")
        if pd.notna(s) and pd.notna(e) and s > e:
            a4_failures.append(f"{row['incident_id']}: start={s} > end={e}")
    if a4_failures:
        _fail(f"Assert 4: {len(a4_failures)} incident(s) have start > end: {a4_failures[:3]}")
    else:
        logger.info("BP-S5-6 Assert 4 PASS: incident_start <= incident_end for all incidents")

    # Assert 5: total_event_count matches manifest sums
    if manifest:
        a5_failures = []
        for _, row in incidents_df.iterrows():
            cids = _parse_cids(row.get("cluster_ids", ""))
            if not cids:
                continue
            expected_total, _ = _incident_counts_from_manifest(cids, manifest)
            stored_total = row.get("total_event_count", None)
            if pd.notna(stored_total) and expected_total > 0:
                stored_int = int(stored_total)
                if stored_int != expected_total:
                    a5_failures.append(
                        f"{row['incident_id']}: stored={stored_int} but manifest_sum={expected_total}"
                    )
        if a5_failures:
            _fail(f"Assert 5: total_event_count mismatch in {len(a5_failures)} incident(s): {a5_failures[:3]}")
        else:
            logger.info("BP-S5-6 Assert 5 PASS: total_event_count matches manifest for all incidents")
    else:
        logger.info("BP-S5-6 Assert 5: skipped (no manifest)")

    all_passed = len(failures) == 0
    if all_passed:
        logger.info("BP-S5-6: ✅ All consistency checks passed")
    else:
        logger.warning("BP-S5-6: ⚠️ %d consistency check(s) failed", len(failures))

    return all_passed, failures


# ══════════════════════════════════════════════════════════════════════
# BP-S5-7 — PIPELINE METADATA BLOCK
# ══════════════════════════════════════════════════════════════════════

def _build_pipeline_metadata(
    stage1_stats: Optional[Dict],
    stage2_stats: Optional[Dict],
    stage3_stats: Optional[Dict],
    stage4_results: Optional[Dict],
    incidents_df: pd.DataFrame,
    manifest: Optional[Dict],
    consistency_passed: bool,
    consistency_failures: List[str],
    suppressed_anomalies: List[Dict],
    validation_failures: List[Dict],
    cfg: Dict,
) -> Dict:
    """BP-S5-7 — Assemble the complete pipeline_metadata audit block."""
    s1 = stage1_stats or {}
    lines_total       = int(s1.get("lines_total", s1.get("total", 0)))
    lines_parsed_ok   = int(s1.get("lines_parsed_ok", s1.get("parsed_ok", 0)))
    lines_noise       = int(s1.get("lines_noise_stripped", s1.get("noise", 0)))
    lines_failed      = int(s1.get("lines_parse_failed", 0))
    lines_quarantined = int(s1.get("lines_quarantined", 0))
    detected_encoding = str(s1.get("detected_encoding", "unknown"))
    detected_format   = str(s1.get("format_probe", s1.get("detected_format", "unknown")))

    s2 = stage2_stats or {}
    unique_templates = int(s2.get("unique_templates", s2.get("n_templates", 0)))
    if not unique_templates and manifest:
        unique_templates = len(manifest.get("clusters", {}))
    total_clusters = int(s2.get("total_clusters", s2.get("n_clusters", unique_templates)))

    s3 = stage3_stats or {}
    header_cluster_count    = int(s3.get("header_cluster_count", total_clusters))
    s3_consistency_failures = list(s3.get("consistency_failures", []))

    s4 = stage4_results or {}
    s4_anomaly_df     = s4.get("anomaly_df", pd.DataFrame())
    n_anomalies_total = len(s4_anomaly_df) if not s4_anomaly_df.empty else 0

    n_incidents   = len(incidents_df)
    sev_breakdown = {}
    if n_incidents > 0 and "incident_severity" in incidents_df.columns:
        sev_breakdown = incidents_df["incident_severity"].value_counts().to_dict()

    return {
        "pipeline_version"           : "v7",
        "run_timestamp"              : datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage_versions"             : {
            "stage1": "BP-A through BP-E",
            "stage2": "BP-M1 through BP-M4",
            "stage3": "ACCURACY-FIX-B1 through B8",
            "stage4": "ACC-5, ACC-6, FIX-L through FIX-O",
            "stage5": "v7 / BP-S5-1 through BP-S5-8 + S5-ML-1..5 + FIX-UF",
        },
        # Stage 1 audit
        "lines_total"                : lines_total,
        "lines_parsed_ok"            : lines_parsed_ok,
        "lines_noise_stripped"       : lines_noise,
        "lines_parse_failed"         : lines_failed,
        "lines_quarantined"          : lines_quarantined,
        "detected_encoding"          : detected_encoding,
        "detected_format"            : detected_format,
        # Stage 2 audit
        "unique_templates"           : unique_templates,
        "total_clusters"             : total_clusters,
        # Stage 3 audit
        "header_cluster_count"       : header_cluster_count,
        "stage3_consistency_failures": s3_consistency_failures,
        # Stage 4 audit
        "total_anomalies_detected"   : n_anomalies_total,
        # Stage 5 audit
        "incidents_formed"           : n_incidents,
        "incidents_by_severity"      : sev_breakdown,
        # S5-ML audit
        "llm_narrative_enabled"      : cfg.get("llm_narrative_enabled", True) and (
            _OLLAMA_AVAILABLE if cfg.get("llm_provider", "ollama") == "ollama" else _ANTHROPIC_AVAILABLE
        ),
        "llm_provider"               : cfg.get("llm_provider", "ollama"),
        "llm_model"                  : (
            cfg.get("ollama_model", "phi4:14b")
            if cfg.get("llm_provider", "ollama") == "ollama"
            else cfg.get("llm_model", "claude-sonnet-4-20250514")
        ),
        "faiss_enabled"              : _FAISS_AVAILABLE,
        "faiss_index_size"           : len(_FAISS_INCIDENT_STORE),
        "faiss_total_queries"        : _FAISS_TOTAL_QUERIES,
        # Suppression + validation
        "suppressed_anomalies"       : suppressed_anomalies,
        "validation_failures"        : validation_failures,
        # Consistency
        "consistency_checks_passed"  : consistency_passed,
        "consistency_failures"       : consistency_failures,
    }


# ══════════════════════════════════════════════════════════════════════
# FIX-6 (v6) — ZERO-FIELD VALIDATION
# ══════════════════════════════════════════════════════════════════════

def _validate_zero_fields(
    incidents_df: pd.DataFrame,
    manifest: Optional[Dict],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fix-6 (scoped) — Flag and repair mathematically impossible zero fields.
    Only repairs total_event_count and error_event_count when manifest
    confirms they should be non-zero.
    """
    if incidents_df.empty:
        return incidents_df, []

    repairs: List[str] = []
    df = incidents_df.copy()

    def _parse_cids(v) -> List[str]:
        s = _safe_str(v)
        return [x.strip() for x in s.split("|") if x.strip()] if s else []

    for idx, row in df.iterrows():
        cids       = _parse_cids(row.get("cluster_ids", ""))
        n_clusters = int(row.get("n_clusters", 0))
        total_ev   = row.get("total_event_count", 0)
        inc_id     = row.get("incident_id", "?")

        try:
            total_is_zero = pd.notna(total_ev) and int(total_ev) == 0
        except (TypeError, ValueError):
            total_is_zero = True

        if total_is_zero and n_clusters > 0 and manifest and cids:
            manifest_total, manifest_errors = _incident_counts_from_manifest(cids, manifest)
            if manifest_total > 0:
                df.at[idx, "total_event_count"] = manifest_total
                df.at[idx, "error_event_count"] = manifest_errors
                repairs.append(
                    f"{inc_id}: total_event_count repaired 0 → {manifest_total} "
                    f"(error_event_count: 0 → {manifest_errors}) from manifest"
                )
                continue

        # Check if error_event_count is suspiciously zero when manifest says otherwise
        error_ev = row.get("error_event_count", 0)
        try:
            error_is_zero  = pd.notna(error_ev) and int(error_ev) == 0
            total_nonzero  = pd.notna(total_ev) and int(total_ev) > 0
        except (TypeError, ValueError):
            continue

        if error_is_zero and total_nonzero and manifest and cids:
            has_error_in_manifest = False
            for cid in cids:
                entry = manifest.get("clusters", {}).get(cid, {})
                dist  = entry.get("severity_distribution", {})
                if int(dist.get("ERROR", 0)) + int(dist.get("FATAL", 0)) > 0:
                    has_error_in_manifest = True
                    break
            if has_error_in_manifest:
                _, manifest_errors = _incident_counts_from_manifest(cids, manifest)
                df.at[idx, "error_event_count"] = manifest_errors
                repairs.append(
                    f"{inc_id}: error_event_count repaired 0 → {manifest_errors} "
                    f"(manifest confirms ERROR/FATAL events exist)"
                )

    return df, repairs


# ══════════════════════════════════════════════════════════════════════
# 5a — INCIDENT GROUPING
# ══════════════════════════════════════════════════════════════════════

def group_incidents(
    anomaly_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    col: Dict,
    cfg: Dict,
) -> Tuple[pd.DataFrame, Dict[str, List]]:
    """
    FIX-5A — Two-pass success handling before grouping.
    Clusters are grouped into incidents by time window + domain/service match
    using a Union-Find algorithm.
    Returns (incident_events_df, incident_map).
    """
    eid_col              = col["event_id"]
    svc_col              = col["service"]
    anomaly_labels       = cfg["incident_anomaly_labels"]
    include_burst_medium = cfg.get("include_burst_medium", True)
    success_patterns     = cfg.get("success_message_patterns", [])

    working_df = anomaly_df.copy()

    # FIX-5A: success-message hard exclusion
    if success_patterns and "sample_message" in working_df.columns:
        success_mask = working_df["sample_message"].apply(
            lambda m: _is_success_message(_safe_str(m), success_patterns)
        )
        downgrade_count = int(success_mask.sum())
        if downgrade_count > 0:
            working_df.loc[success_mask, "anomaly_label"]    = "INFO"
            working_df.loc[success_mask, "_success_excluded"] = True
            logger.info(
                "5a: Success-message hard-exclusion: %d cluster(s) removed from candidacy",
                downgrade_count,
            )
        if "_success_excluded" in working_df.columns:
            working_df = working_df[working_df["_success_excluded"].isna()].copy()
            working_df = working_df.drop(columns=["_success_excluded"], errors="ignore")

    is_anomalous = working_df["anomaly_label"].isin(anomaly_labels)
    if include_burst_medium and "burst_detected" in working_df.columns:
        burst_bool   = _coerce_burst(working_df["burst_detected"])
        is_anomalous = is_anomalous | ((working_df["anomaly_label"] == "MEDIUM") & burst_bool)

    candidates = working_df[is_anomalous].copy()
    logger.info("5a: Candidate anomalous clusters: %d", len(candidates))

    if len(candidates) == 0:
        logger.info("5a: No anomalous clusters — no incidents to form")
        return pd.DataFrame(), {}

    # Map timestamps onto candidates
    if temporal_df is not None and len(temporal_df) > 0 and "first_seen" in temporal_df.columns:
        ts_map = (
            temporal_df[[eid_col, "first_seen", "last_seen"]]
            .drop_duplicates(subset=eid_col)
            .set_index(eid_col)
        )
        candidates["_first_seen"] = candidates[eid_col].map(ts_map["first_seen"])
        candidates["_last_seen"]  = candidates[eid_col].map(ts_map["last_seen"])
    else:
        candidates["_first_seen"] = pd.NaT
        candidates["_last_seen"]  = pd.NaT

    candidates["_first_seen"] = _coerce_timestamps(candidates["_first_seen"])
    candidates["_last_seen"]  = _coerce_timestamps(candidates["_last_seen"])

    if svc_col and svc_col in candidates.columns:
        candidates["_svc"] = candidates[svc_col].fillna("unknown")
    elif "top_source" in candidates.columns:
        candidates["_svc"] = candidates["top_source"].fillna("unknown")
    else:
        candidates["_svc"] = "unknown"

    ts_valid = candidates["_first_seen"].dropna()
    window_s = _adaptive_window(ts_valid, cfg)

    # FIX-UF: cap candidates to avoid O(n²) blowup on large log files.
    # When count exceeds union_find_max_candidates, fall back to
    # timestamp-only grouping (no domain/service check).
    uf_max   = int(cfg.get("union_find_max_candidates", 1000))
    n        = len(candidates)
    use_ts_only = (n > uf_max)
    if use_ts_only:
        logger.warning(
            "FIX-UF: %d candidates exceeds cap=%d — using timestamp-only grouping "
            "(domain/service check disabled for this run). "
            "Raise union_find_max_candidates in cfg to restore full matching.",
            n, uf_max,
        )

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    cands_reset = candidates.reset_index(drop=True)
    rows        = cands_reset.to_dict("records")

    for i in range(n):
        for j in range(i + 1, n):
            a, b   = rows[i], rows[j]
            ts_a   = a["_first_seen"]
            ts_b   = b["_first_seen"]

            if pd.isna(ts_a) or pd.isna(ts_b):
                time_ok = True
            else:
                time_ok = abs((ts_a - ts_b).total_seconds()) <= window_s

            if not time_ok:
                continue
            # FIX-UF: skip domain/service check when over cap
            if use_ts_only or _domain_or_service_match(a, b, cfg):
                union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    group_list = list(groups.values())

    def _group_min_ts(indices: List[int]):
        ts_vals = [rows[i]["_first_seen"] for i in indices]
        valid   = [t for t in ts_vals if not pd.isna(t)]
        return min(valid) if valid else pd.Timestamp.max

    group_list.sort(key=_group_min_ts)

    incident_map: Dict[str, List] = {}
    event_to_incident: Dict = {}

    for rank, indices in enumerate(group_list, start=1):
        inc_id = f"INC-{rank:04d}"
        eids   = [rows[i][eid_col] for i in indices]
        incident_map[inc_id] = eids
        for eid in eids:
            event_to_incident[eid] = inc_id

    cands_reset["incident_id"] = cands_reset[eid_col].map(event_to_incident)
    cands_reset = cands_reset.drop(columns=["_svc"], errors="ignore")

    logger.info(
        "5a: Formed %d incidents from %d anomalous clusters (window=%.1fs)",
        len(incident_map), len(candidates), window_s,
    )
    return cands_reset, incident_map


# ══════════════════════════════════════════════════════════════════════
# 5b — CASCADE CHAIN RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════

def reconstruct_cascade_chains(
    incident_events_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    col: Dict,
    cfg: Dict,
) -> pd.DataFrame:
    """
    FIX-6A/B/D — Reconstruct full ordered cascade chains for each incident.
    Ties broken by anomaly_score (FIX-6D). Always builds full chain (FIX-6B).
    Simultaneity threshold is configurable, not hardcoded (FIX-6A).
    """
    if len(incident_events_df) == 0:
        return incident_events_df

    eid_col      = col["event_id"]
    svc_col      = col["service"]
    simul_thresh = cfg.get("cascade_simultaneity_threshold_s", 1)

    df = incident_events_df.copy()

    if svc_col and svc_col in df.columns:
        df["_svc"] = df[svc_col].fillna("unknown")
    elif "top_source" in df.columns:
        df["_svc"] = df["top_source"].fillna("unknown")
    else:
        df["_svc"] = "unknown"

    if "_first_seen" not in df.columns:
        df["_first_seen"] = pd.NaT
    df["_first_seen"] = _coerce_timestamps(df["_first_seen"])

    if temporal_df is not None and len(temporal_df) > 0 and "first_seen" in temporal_df.columns:
        ts_map = (
            temporal_df[[eid_col, "first_seen", "last_seen"]]
            .drop_duplicates(subset=eid_col)
            .set_index(eid_col)
        )
        mapped_first = df[eid_col].map(ts_map["first_seen"])
        mapped_last  = df[eid_col].map(ts_map["last_seen"])
        df["_first_seen"] = _coerce_timestamps(mapped_first.combine_first(df["_first_seen"]))
        if "_last_seen" not in df.columns:
            df["_last_seen"] = pd.NaT
        # S5-TZ-FIX: coerce each side independently first, then combine_first,
        # then one final coerce on the merged result. The old nested pattern
        # passed a tz-aware intermediate into combine_first alongside a tz-naive
        # series, producing silent wrong values or a mixed-tz error.
        _last_coerced  = _coerce_timestamps(mapped_last)
        _last_fallback = _coerce_timestamps(df["_last_seen"])
        df["_last_seen"] = _coerce_timestamps(_last_coerced.combine_first(_last_fallback))

    cascade_position   = pd.Series(1,     index=df.index, dtype=int)
    root_cause_service = pd.Series("",    index=df.index, dtype=str)
    cascade_chain_str  = pd.Series("",    index=df.index, dtype=str)
    is_root_cause      = pd.Series(False, index=df.index, dtype=bool)

    for inc_id, grp in df.groupby("incident_id"):
        # Build per-service score map for tiebreaking (FIX-6D)
        svc_score_map: Dict[str, float] = {}
        if "anomaly_score" in grp.columns:
            for _, row in grp.iterrows():
                svc   = _safe_str(row.get("_svc", "unknown"))
                score = float(row.get("anomaly_score", 0) or 0)
                if svc not in svc_score_map or score > svc_score_map[svc]:
                    svc_score_map[svc] = score

        svc_times = grp.groupby("_svc")["_first_seen"].min().dropna()

        if len(svc_times) == 0:
            svcs_present = grp["_svc"].unique().tolist()
            chain_str    = " → ".join(svcs_present) if svcs_present else ""
            root_svc     = svcs_present[0] if svcs_present else "unknown"
            for row_idx in grp.index:
                root_cause_service[row_idx] = root_svc
                cascade_chain_str[row_idx]  = chain_str
                is_root_cause[row_idx]      = (df.at[row_idx, "_svc"] == root_svc)
            continue

        svc_times_sorted  = svc_times.sort_values()
        ordered_svcs_raw  = list(svc_times_sorted.index)

        # FIX-6D: tiebreaker within tied-timestamp groups
        ordered_svcs: List[str] = []
        i = 0
        while i < len(ordered_svcs_raw):
            group_ts = svc_times_sorted.iloc[i]
            j = i
            while j < len(ordered_svcs_raw) and svc_times_sorted.iloc[j] == group_ts:
                j += 1
            tied_svcs = ordered_svcs_raw[i:j]
            tied_svcs.sort(key=lambda s: -svc_score_map.get(s, 0.0))
            ordered_svcs.extend(tied_svcs)
            i = j

        t0 = svc_times_sorted.iloc[0]
        simultaneous = (
            len(svc_times_sorted) >= 2 and
            (svc_times_sorted.iloc[1] - t0).total_seconds() < simul_thresh
        )
        root_svc = ordered_svcs[0] if not simultaneous else "co-originating"

        # FIX-6B: always build full ordered chain
        chain_parts = []
        for svc in ordered_svcs:
            lag_s = (svc_times[svc] - t0).total_seconds()
            chain_parts.append(f"{svc}({_format_lag(lag_s)})")
        chain_str = " → ".join(chain_parts)

        for row_idx in grp.index:
            row_svc  = df.at[row_idx, "_svc"]
            position = ordered_svcs.index(row_svc) + 1 if row_svc in ordered_svcs else 1
            cascade_position[row_idx]   = position
            root_cause_service[row_idx] = root_svc
            cascade_chain_str[row_idx]  = chain_str
            is_root_cause[row_idx]      = (row_svc == root_svc and not simultaneous)

    df["cascade_position"]   = cascade_position
    df["root_cause_service"] = root_cause_service
    df["cascade_chain"]      = cascade_chain_str
    df["is_root_cause"]      = is_root_cause
    df = df.drop(columns=["_svc"], errors="ignore")

    logger.info("5b: Cascade chains built. Root causes identified: %d", is_root_cause.sum())
    return df


# ══════════════════════════════════════════════════════════════════════
# 5c — TIMELINE BUILDING
# ══════════════════════════════════════════════════════════════════════

def build_incident_timelines(
    raw_df: Optional[pd.DataFrame],
    incident_events_df: pd.DataFrame,
    col: Dict,
    cfg: Dict,
) -> pd.DataFrame:
    """Build the raw-line timeline for all incidents."""
    if len(incident_events_df) == 0 or raw_df is None or len(raw_df) == 0:
        return pd.DataFrame()

    eid_col = col["event_id"]
    svc_col = col["service"]
    ts_col  = col["timestamp"]
    msg_col = col["message"]
    sev_col = col["severity"]

    if eid_col not in raw_df.columns:
        logger.warning("5c: event_id column not found in raw_df — timeline skipped")
        return pd.DataFrame()

    lookup = (
        incident_events_df[[eid_col, "incident_id", "anomaly_label", "is_root_cause"]]
        .drop_duplicates(subset=eid_col)
        .set_index(eid_col)
    )

    incident_eids = set(lookup.index)
    mask          = raw_df[eid_col].isin(incident_eids)
    tl            = raw_df[mask].copy()

    if len(tl) == 0:
        logger.info("5c: No raw lines matched incident clusters")
        return pd.DataFrame()

    tl["incident_id"]   = tl[eid_col].map(lookup["incident_id"])
    tl["anomaly_label"] = tl[eid_col].map(lookup["anomaly_label"])
    tl["is_root_cause"] = tl[eid_col].map(lookup["is_root_cause"]).fillna(False)

    if ts_col and ts_col in tl.columns:
        tl["_ts"] = _coerce_timestamps(tl[ts_col])
    else:
        tl["_ts"] = pd.NaT

    keep = {"incident_id", "_ts", eid_col, "anomaly_label", "is_root_cause"}
    for c in [svc_col, sev_col, msg_col, "domain"]:
        if c and c in tl.columns:
            keep.add(c)

    tl = tl[[c for c in keep if c in tl.columns]].copy()
    tl = tl.sort_values(["incident_id", "_ts"], na_position="last").reset_index(drop=True)

    logger.info(
        "5c: Timeline built: %d log lines across %d incidents",
        len(tl), tl["incident_id"].nunique(),
    )
    return tl


# ══════════════════════════════════════════════════════════════════════
# 5d — INCIDENT SUMMARIES
# ══════════════════════════════════════════════════════════════════════

def build_incident_summaries(
    incident_events_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    col: Dict,
    cfg: Dict,
    anomaly_df: Optional[pd.DataFrame] = None,
    cluster_manifest: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Build one summary row per incident.
    Returns (incidents_df, validation_failures_list).

    BP-S5-1: counts from manifest only.
    BP-S5-2: explicit cluster_ids; n_clusters = len(cluster_ids).
    BP-S5-4: grounding validation on narrative text.
    BP-S5-5: deterministic fallback if grounding fails.
    BP-S5-8: services_affected as explicit list.
    FIX-3C:  domain column guaranteed present.
    FIX-4:   recurrence keyed on (root_cause_service, error_class).
    FIX-5B:  all-success incidents skipped.
    NEW-A/B/C/D: narrative generators.
    """
    if len(incident_events_df) == 0:
        return pd.DataFrame(), []

    # FIX-3C: guarantee domain column
    ref_df = anomaly_df if anomaly_df is not None else incident_events_df
    incident_events_df = _ensure_domain_column(incident_events_df, ref_df, col, cfg)

    eid_col         = col["event_id"]
    svc_col         = col["service"]
    sev_rank        = cfg["severity_rank"]
    domain_priority = cfg.get("domain_priority", [])
    impossible_pats = cfg.get("impossible_patterns", [])
    success_pats    = cfg.get("success_message_patterns", [])

    # Fatal-signal patterns for NEW-C severity guard
    fatal_patterns = [
        r"EADDRINUSE", r"FATAL", r"fetch failed", r"not found after \d+ retr",
        r"relation .+ does not exist", r"ENOENT", r"Local file not found",
    ]

    def _has_fatal_signal(trigger: str, hint: str) -> bool:
        combined = f"{trigger} {hint}"
        return any(re.search(p, combined, re.IGNORECASE) for p in fatal_patterns)

    validation_failures: List[Dict] = []
    rows: List[Dict] = []
    seen_root_error_pairs: Set[Tuple] = set()   # for FIX-4 recurrence detection (populated after loop)

    for inc_id, grp in incident_events_df.groupby("incident_id"):

        # BP-S5-2: explicit cluster_ids
        cluster_ids: List[str] = sorted(grp[eid_col].astype(str).tolist())
        n_clusters: int = len(cluster_ids)

        label_ranks = grp["anomaly_label"].map(sev_rank).fillna(0)
        max_rank    = int(label_ranks.max())
        inc_sev     = {v: k for k, v in sev_rank.items()}.get(max_rank, "LOW")

        # BP-S5-8: services_affected — manifest primary, df fallback
        fallback_svc_vals: Set[str] = set()
        if svc_col and svc_col in grp.columns:
            fallback_svc_vals.update(grp[svc_col].dropna().astype(str).unique().tolist())
        if "top_source" in grp.columns:
            fallback_svc_vals.update(grp["top_source"].dropna().astype(str).unique().tolist())
        if "cascade_source" in grp.columns:
            for cs_val in grp["cascade_source"].dropna():
                fallback_svc_vals.update(_parse_cascade_source(cs_val))
        fallback_svc_vals.discard("unknown"); fallback_svc_vals.discard(""); fallback_svc_vals.discard("nan")

        services_affected_list: List[str] = _incident_services_from_manifest(
            cluster_ids, cluster_manifest, fallback_svc_vals
        )
        n_services: int = max(
            len([s for s in services_affected_list if s not in ("unknown", "")]), 1
        )
        services_affected_str: str = json.dumps(services_affected_list)

        # FIX-3: primary_domain
        primary_dom = _get_primary_domain(grp["domain"], domain_priority)

        root_svc_vals = grp.get("root_cause_service", pd.Series(dtype=str)).dropna().unique()
        root_svc      = root_svc_vals[0] if len(root_svc_vals) > 0 else "unknown"

        chain_vals       = grp.get("cascade_chain", pd.Series(dtype=str)).dropna().unique()
        non_empty_chains = [c for c in chain_vals if c and str(c).strip()]
        multi_hop        = [c for c in non_empty_chains if " → " in c]
        cascade_chain    = multi_hop[0] if multi_hop else (non_empty_chains[0] if non_empty_chains else "")

        # Temporal span
        first_seen_vals = _coerce_timestamps(grp.get("_first_seen", pd.Series(dtype=object)))
        last_seen_vals  = _coerce_timestamps(grp.get("_last_seen",  pd.Series(dtype=object)))
        inc_start = first_seen_vals.min() if first_seen_vals.notna().any() else pd.NaT
        inc_end   = last_seen_vals.max()  if last_seen_vals.notna().any()  else pd.NaT
        if pd.isna(inc_end) and pd.notna(inc_start):
            inc_end = inc_start
        duration_min = (
            round((inc_end - inc_start).total_seconds() / 60, 1)
            if pd.notna(inc_start) and pd.notna(inc_end) else float("nan")
        )

        # BP-S5-1: manifest-sourced event counts
        total_event_count, error_event_count = _incident_counts_from_manifest(
            cluster_ids, cluster_manifest
        )

        # Signals
        burst_col  = grp.get("burst_detected", pd.Series(dtype=object))
        any_burst  = bool(_coerce_burst(burst_col).any())
        trend_col  = grp.get("trend_direction", pd.Series(dtype=str))
        any_rising = bool(_is_rising(trend_col).any())

        sample_msgs = grp.get("sample_message", pd.Series(dtype=str)).fillna("").tolist()

        # FIX-5B: skip all-success incidents
        all_success = (
            len(sample_msgs) > 0 and
            all(_is_success_message(m, success_pats) for m in sample_msgs if m.strip())
        )
        if all_success:
            logger.info("FIX-5B: Skipping all-success incident %s: %s", inc_id, sample_msgs[0][:80])
            continue

        has_impossible = any(_is_impossible_message(m, impossible_pats) for m in sample_msgs)

        # Security signal detection
        has_security = False
        if "domain" in grp.columns:
            if grp["domain"].astype(str).str.lower().eq("auth").any():
                has_security = True
        if not has_security and "domain" in grp.columns and "sample_message" in grp.columns:
            msg_mask = grp["domain"].astype(str).str.lower().eq("messaging")
            if msg_mask.any():
                msg_texts = grp.loc[msg_mask, "sample_message"].fillna("").astype(str)
                if msg_texts.str.contains(r'\b4\d{2}\b', regex=True).any():
                    has_security = True
        if not has_security and "singleton_class" in grp.columns:
            if grp["singleton_class"].isin({"impossible_attempt_count"}).any():
                has_security = True

        error_trigger = _extract_error_trigger(sample_msgs, cfg)

        # Root cause hint from highest-scoring cluster
        root_cause_hint = ""
        if "anomaly_score" in grp.columns and "sample_message" in grp.columns:
            scores   = pd.to_numeric(grp["anomaly_score"], errors="coerce")
            best_idx = scores.idxmax() if scores.notna().any() else None
            if best_idx is not None:
                root_cause_hint = _safe_str(grp.at[best_idx, "sample_message"])[:120]

        top_cluster_score = float("nan")
        if "anomaly_score" in grp.columns:
            scores = pd.to_numeric(grp["anomaly_score"], errors="coerce")
            if scores.notna().any():
                top_cluster_score = round(float(scores.max()), 4)

        singleton_cluster_count = 0
        if "singleton_class" in grp.columns:
            singleton_cluster_count = int(
                (grp["singleton_class"].astype(str).str.strip() == "true_anomaly").sum()
            )

        affected_doc_ids = _extract_doc_ids(sample_msgs, cfg=cfg)

        rows.append({
            # BP-S5-2
            "cluster_ids"            : "|".join(cluster_ids),
            "incident_id"            : inc_id,
            "incident_severity"      : inc_sev,
            "n_clusters"             : n_clusters,
            "n_services_affected"    : n_services,
            "services_affected"      : services_affected_str,
            "primary_domain"         : primary_dom,
            "root_cause_service"     : root_svc,
            "cascade_chain"          : cascade_chain,
            "incident_start"         : inc_start,
            "incident_end"           : inc_end,
            "duration_minutes"       : duration_min,
            # BP-S5-1
            "total_event_count"      : total_event_count,
            "error_event_count"      : error_event_count,
            "any_burst"              : any_burst,
            "any_rising_trend"       : any_rising,
            "has_impossible_attempt" : has_impossible,
            "has_security_signal"    : has_security,
            "root_cause_hint"        : root_cause_hint,
            "error_trigger"          : error_trigger,
            "top_cluster_score"      : top_cluster_score,
            "singleton_cluster_count": singleton_cluster_count,
            "recurrence_flag"        : False,   # filled below (FIX-4)
            "affected_document_ids"  : affected_doc_ids,
            "member_event_ids"       : "|".join(str(e) for e in grp[eid_col].tolist()),
            "narrative_grounded"     : True,    # updated after validation
            "narrative"              : "",       # alias filled after what_happened is set
            # S5-ML new columns
            "narrative_source"       : "deterministic_fallback",  # overwritten if LLM succeeds
            "similar_past_incidents" : "[]",    # JSON list; overwritten after FAISS search
        })

    if not rows:
        return pd.DataFrame(), validation_failures

    incidents_df = pd.DataFrame(rows)

    # FIX-4: recurrence keyed on (root_cause_service, error_class) pair
    recurrence_flags: List[bool] = []
    seen_root_error_pairs_local: Set[Tuple] = set()
    for _, row in incidents_df.iterrows():
        rc        = _safe_str(row["root_cause_service"])
        err_class = _classify_error_class(_safe_str(row.get("error_trigger", "")), cfg)
        pair      = (rc, err_class)
        is_recur  = pair in seen_root_error_pairs_local and rc not in ("", "unknown", "co-originating")
        recurrence_flags.append(is_recur)
        if rc and rc not in ("unknown", "co-originating"):
            seen_root_error_pairs_local.add(pair)
    incidents_df["recurrence_flag"] = recurrence_flags

    # what_happened generation (NEW-A)
    what_happened_list: List[str] = []
    for _, row in incidents_df.iterrows():
        grp  = incident_events_df[incident_events_df["incident_id"] == row["incident_id"]]
        msgs = grp.get("sample_message", pd.Series(dtype=str)).fillna("").tolist()
        what = _generate_what_happened(
            messages   = msgs,
            root_svc   = _safe_str(row["root_cause_service"]),
            n_services = int(row["n_services_affected"]),
            any_burst  = bool(row["any_burst"]),
            recurrence = bool(row["recurrence_flag"]),
            security   = bool(row["has_security_signal"]),
            n_clusters = int(row["n_clusters"]),
            cfg        = cfg,
        )
        what_happened_list.append(what)
    incidents_df["what_happened"] = what_happened_list

    # NEW-C: severity guard (instant CRITICAL → HIGH without fatal signal)
    def _apply_severity_guard(row: pd.Series) -> str:
        if row["incident_severity"] != "CRITICAL":
            return row["incident_severity"]
        try:
            dur = float(row["duration_minutes"])
        except (TypeError, ValueError):
            dur = None
        if dur == 0.0 and not _has_fatal_signal(
            _safe_str(row.get("error_trigger", "")),
            _safe_str(row.get("root_cause_hint", "")),
        ):
            return "HIGH"
        return "CRITICAL"

    incidents_df["incident_severity"] = incidents_df.apply(_apply_severity_guard, axis=1)

    # NEW-D: recommended action
    incidents_df["recommended_action"] = incidents_df.apply(
        lambda row: _recommended_action_v4(row, cfg), axis=1
    )

    # ── S5-ML-1/3/4: FAISS similarity search + LLM narrative ──────────
    # Load persisted FAISS index once per run (no-op if already in memory)
    _faiss_load_index(cfg)

    _provider_available = (
        _OLLAMA_AVAILABLE if cfg.get("llm_provider", "ollama") == "ollama"
        else _ANTHROPIC_AVAILABLE
    )
    llm_enabled = cfg.get("llm_narrative_enabled", True) and _provider_available

    # ── Ollama warm-up: send a minimal prompt before parallel calls ────
    # phi4:14b can take 30-90s to load from cold. Warming up once here
    # means all parallel workers see a warm model (~5-15s per call).
    if llm_enabled and cfg.get("llm_provider", "ollama") == "ollama" and _OLLAMA_AVAILABLE:
        try:
            import requests as _rq
            _warmup_resp = _rq.post(
                f"{cfg.get('ollama_base_url', 'http://localhost:11434')}/api/generate",
                json={
                    "model" : cfg.get("ollama_model", "phi4:14b"),
                    "prompt": "ping",
                    "stream": False,
                    "options": {"num_predict": 1, "temperature": 0},
                },
                timeout=90,   # allow full cold-start time on the warm-up ping
            )
            logger.info("S5-ML-1: Ollama warm-up complete (status=%d)", _warmup_resp.status_code)
        except Exception as _wu_exc:
            logger.warning("S5-ML-1: Ollama warm-up ping failed (%s) — proceeding anyway", _wu_exc)

    # ── Pre-compute per-incident inputs (cheap, sequential) ────────────
    incident_inputs: List[Dict] = []
    for idx, row in incidents_df.iterrows():
        row_dict = row.to_dict()
        cids     = [x.strip() for x in _safe_str(row.get("cluster_ids", "")).split("|") if x.strip()]
        svcs     = _parse_services(row.get("services_affected"))
        similar  = _faiss_search_similar(row_dict, cfg, exclude_incident_id=_safe_str(row.get("incident_id")))
        incidents_df.at[idx, "similar_past_incidents"] = json.dumps(similar)
        incident_inputs.append({
            "idx"     : idx,
            "row_dict": row_dict,
            "cids"    : cids,
            "svcs"    : svcs,
            "similar" : similar,
        })

    # ── Parallel LLM narrative generation ──────────────────────────────
    # LLM calls are I/O-bound (HTTP to Ollama/Anthropic), so threading
    # gives near-linear speedup. max_workers=1 restores sequential behaviour.
    max_workers = int(cfg.get("llm_max_workers", 3)) if llm_enabled else 1

    def _process_one_incident(inp: Dict):
        """Worker: gate check + LLM call for a single incident. Thread-safe."""
        row_dict = inp["row_dict"]
        cids     = inp["cids"]
        svcs     = inp["svcs"]
        similar  = inp["similar"]
        should_call_llm, gate_reason = _llm_confidence_gate(row_dict, cfg)
        if should_call_llm:
            rc_text, ra_text, narr_src = _call_llm_narrative(
                incident_row      = row_dict,
                cluster_ids       = cids,
                services_affected = svcs,
                similar_past      = similar,
                cfg               = cfg,
            )
        else:
            rc_text, ra_text, narr_src = "", "", "deterministic_fallback"
        return inp["idx"], rc_text, ra_text, narr_src, gate_reason

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one_incident, inp): inp for inp in incident_inputs}
        for future in as_completed(futures):
            try:
                idx, rc_text, ra_text, narr_src, gate_reason = future.result()
                row = incidents_df.loc[idx]
                if narr_src != "deterministic_fallback" and rc_text:
                    incidents_df.at[idx, "what_happened"]      = rc_text
                    incidents_df.at[idx, "recommended_action"] = ra_text if ra_text else row.get("recommended_action", "")
                    incidents_df.at[idx, "narrative_source"]   = narr_src
                    logger.info(
                        "S5-ML-1: applied LLM narrative for %s (source=%s)",
                        row.get("incident_id"), narr_src,
                    )
                else:
                    incidents_df.at[idx, "narrative_source"] = "deterministic_fallback"
                    logger.info(
                        "S5-ML-1: LLM fallback for %s (gate=%s, llm_result=%s)",
                        row.get("incident_id"), gate_reason, narr_src,
                    )
                # S5-ML-3: index this incident for future similarity retrieval
                _faiss_index_incident(incidents_df.loc[idx].to_dict(), cfg)
            except Exception as _fe:
                logger.warning("S5-ML-1: worker exception for incident: %s", _fe)

    # Add similar incidents context to recommended_action where present
    for idx, row in incidents_df.iterrows():
        similar_raw = row.get("similar_past_incidents", "[]")
        try:
            similar_list = json.loads(similar_raw) if isinstance(similar_raw, str) else similar_raw
        except Exception:
            similar_list = []
        if similar_list:
            context_lines = []
            for sp in similar_list[:2]:
                context_lines.append(
                    f"Similar past: {sp.get('incident_id','')} "
                    f"(root={sp.get('root_cause_service','')}, "
                    f"sim={sp.get('similarity',0):.2f}) — "
                    f"{str(sp.get('recommended_action',''))[:60]}..."
                )
            if context_lines:
                existing = _safe_str(row.get("recommended_action", ""))
                incidents_df.at[idx, "recommended_action"] = (
                    existing + " | " + " | ".join(context_lines)
                ).strip(" |")

    # BP-S5-4: grounding validation + BP-S5-5: deterministic fallback
    for idx, row in incidents_df.iterrows():
        cids     = [x.strip() for x in _safe_str(row.get("cluster_ids", "")).split("|") if x.strip()]
        svcs     = _parse_services(row.get("services_affected"))
        what_txt = _safe_str(row.get("what_happened", ""))
        act_txt  = _safe_str(row.get("recommended_action", ""))

        what_grounded = _validate_llm_narrative(what_txt, cids, svcs)
        if not what_grounded:
            fallback = _deterministic_narrative_fallback(
                incident_id       = _safe_str(row["incident_id"]),
                cluster_ids       = cids,
                services_affected = svcs,
                peak_anomaly_level= _safe_str(row.get("incident_severity", "HIGH")),
                start_ts          = row.get("incident_start"),
                end_ts            = row.get("incident_end"),
                error_trigger     = _safe_str(row.get("error_trigger", "")),
            )
            incidents_df.at[idx, "what_happened"] = fallback["root_cause_summary"]
            validation_failures.append({
                "stage"      : "stage5_what_happened",
                "incident_id": _safe_str(row["incident_id"]),
                "assertion"  : "narrative_grounding",
                "resolution" : "deterministic_fallback_applied",
            })
            logger.info("BP-S5-4: what_happened ungrounded for %s — fallback applied", row["incident_id"])

        act_grounded = _validate_llm_narrative(act_txt, cids, svcs)
        if not act_grounded:
            fallback = _deterministic_narrative_fallback(
                incident_id       = _safe_str(row["incident_id"]),
                cluster_ids       = cids,
                services_affected = svcs,
                peak_anomaly_level= _safe_str(row.get("incident_severity", "HIGH")),
                start_ts          = row.get("incident_start"),
                end_ts            = row.get("incident_end"),
                error_trigger     = _safe_str(row.get("error_trigger", "")),
            )
            incidents_df.at[idx, "recommended_action"] = fallback["recommended_action"]
            validation_failures.append({
                "stage"      : "stage5_recommended_action",
                "incident_id": _safe_str(row["incident_id"]),
                "assertion"  : "narrative_grounding",
                "resolution" : "deterministic_fallback_applied",
            })
            logger.info("BP-S5-4: recommended_action ungrounded for %s — fallback applied", row["incident_id"])

        incidents_df.at[idx, "narrative_grounded"] = what_grounded and act_grounded

    # Sync narrative field from what_happened (narrative is the canonical alias used by the frontend)
    incidents_df["narrative"] = incidents_df["what_happened"]

    # Sort by severity then start time
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    incidents_df["_sev_rank"] = incidents_df["incident_severity"].map(sev_order).fillna(9)
    incidents_df = (
        incidents_df
        .sort_values(["_sev_rank", "incident_start"])
        .drop(columns=["_sev_rank"])
        .reset_index(drop=True)
    )

    logger.info("5d: Incident summaries built: %d incidents", len(incidents_df))
    sev_counts = incidents_df["incident_severity"].value_counts()
    for sev, cnt in sev_counts.items():
        logger.info("  %s: %d", sev, cnt)

    return incidents_df, validation_failures


# ══════════════════════════════════════════════════════════════════════
# COLUMN RESOLUTION (shared with Stage 4 pattern)
# ══════════════════════════════════════════════════════════════════════

def _resolve_col(df: pd.DataFrame, name: str, fallbacks: Tuple = ()) -> Optional[str]:
    if name in df.columns:
        return name
    for fb in fallbacks:
        if fb in df.columns:
            logger.debug("col_resolve: '%s' not found — using '%s'", name, fb)
            return fb
    return None


def resolve_columns(df: pd.DataFrame, cfg: Dict) -> Dict[str, Optional[str]]:
    return {
        "service"       : _resolve_col(df, cfg.get("col_service", "service"),
                                        ("source", "svc", "host")),
        "event_label"   : _resolve_col(df, cfg.get("col_event_label", "cluster_label"),
                                        ("label", "cluster_name")),
        "event_id"      : _resolve_col(df, cfg.get("col_event_id", "event_id"),
                                        ("semantic_cluster_id", "cluster_id", "template_id")),
        "severity"      : _resolve_col(df, cfg.get("col_severity", "severity"),
                                        ("level", "log_level")),
        "timestamp"     : _resolve_col(df, cfg.get("col_timestamp", "timestamp_parsed"),
                                        ("ts", "time", "datetime", "log_time", "timestamp")),
        "message"       : _resolve_col(df, cfg.get("col_message", "message"),
                                        ("msg", "raw_line", "normalized_message")),
        "is_noise"      : _resolve_col(df, cfg.get("col_is_noise", "is_noise")),
        "event_template": _resolve_col(df, cfg.get("col_event_template", "event_template"),
                                        ("template",)),
        "singleton_class": _resolve_col(df, cfg.get("col_singleton_class", "singleton_class")),
        "domain"        : _resolve_col(df, cfg.get("col_domain", "domain"),
                                        ("cluster_label", "event_label")),
        "cluster_id"    : _resolve_col(df, cfg.get("col_cluster_id", "cluster_id"),
                                        ("semantic_cluster_id", "event_id")),
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN RUNNER — run_stage5()
# ══════════════════════════════════════════════════════════════════════

def run_stage5(
    stage4_results: Dict,
    raw_df: Optional[pd.DataFrame] = None,
    cfg: Optional[Dict] = None,
    cluster_manifest: Optional[Dict] = None,
    stage1_stats: Optional[Dict] = None,
    stage2_stats: Optional[Dict] = None,
    stage3_stats: Optional[Dict] = None,
) -> Dict:
    """
    Stage 5 — Root Cause Analysis  (v6 — Blueprint BP-S5-1 through BP-S5-8)

    Parameters
    ----------
    stage4_results   : dict returned by run_stage4()
    raw_df           : original normalised log DataFrame (for timeline building)
    cfg              : optional config overrides for STAGE5_CONFIG
    cluster_manifest : dict from Stage 2 build_manifest() — strongly recommended;
                       when provided all event counts are manifest-sourced, never estimated
    stage1_stats     : stats dict from Stage 1 (for pipeline_metadata)
    stage2_stats     : stats dict from Stage 2 (for pipeline_metadata)
    stage3_stats     : stats dict from Stage 3 (for pipeline_metadata)

    Returns
    -------
    dict with keys:
        incidents_df        — one row per incident
        incident_events_df  — anomalous clusters with cascade chain
        timeline_df         — raw log lines that belong to any incident
        unlinked_anomalies  — anomalous clusters not linked to any incident
        incident_map        — {incident_id: [cluster_ids]}
        col_map             — resolved column name map
        config_used         — full merged config
        pipeline_metadata   — BP-S5-7 audit block
    """
    if cfg is None:
        cfg = {}
    full_cfg = {**STAGE5_CONFIG, **cfg}

    print("=" * 60)
    print("Stage 5 — Root Cause Analysis  (v7 / BP-S5-1..8 + S5-ML-1..5)")
    print("=" * 60)

    if cluster_manifest:
        n_manifest_clusters = len(cluster_manifest.get("clusters", {}))
        print(f"  Manifest: {n_manifest_clusters} clusters (counts are ground truth)")
    else:
        print("  ⚠  No cluster_manifest provided — event counts will be 0.")
        print("     Pass cluster_manifest from Stage 2 for accurate counts.")

    _llm_provider_up = (
        _OLLAMA_AVAILABLE if full_cfg.get("llm_provider", "ollama") == "ollama"
        else _ANTHROPIC_AVAILABLE
    )
    _llm_model_label = (
        full_cfg.get("ollama_model", "phi4:14b")
        if full_cfg.get("llm_provider", "ollama") == "ollama"
        else full_cfg.get("llm_model", "claude-sonnet-4-20250514")
    )
    llm_status = (
        f"enabled [{full_cfg.get('llm_provider', 'ollama')}] (model={_llm_model_label})"
        if (_llm_provider_up and full_cfg.get("llm_narrative_enabled", True))
        else f"disabled ({full_cfg.get('llm_provider', 'ollama')} not available or llm_narrative_enabled=False)"
    )
    faiss_status = (
        f"enabled ({len(_FAISS_INCIDENT_STORE)} past incidents indexed)"
        if _FAISS_AVAILABLE
        else "disabled (faiss-cpu not installed)"
    )
    print(f"  LLM narrative  : {llm_status}")
    print(f"  FAISS similarity: {faiss_status}")

    # ── Unpack Stage 4 outputs ─────────────────────────────────────────
    anomaly_df  = stage4_results["anomaly_df"].copy()
    temporal_df = stage4_results.get("temporal_df", pd.DataFrame())

    # Resolve column names from the anomaly_df schema
    col_cfg = {
        "col_service"        : "service",
        "col_event_label"    : "cluster_label",
        "col_event_id"       : "event_id",
        "col_severity"       : "severity",
        "col_timestamp"      : "timestamp_parsed",
        "col_message"        : "message",
        "col_is_noise"       : "is_noise",
        "col_event_template" : "event_template",
        "col_singleton_class": "singleton_class",
        "col_domain"         : "domain",
        "col_cluster_id"     : "cluster_id",
        **{k: v for k, v in full_cfg.items() if k.startswith("col_")},
    }
    col_full_cfg = {**full_cfg, **col_cfg}
    col          = resolve_columns(anomaly_df, col_full_cfg)
    eid_col      = col["event_id"]

    if eid_col is None:
        raise RuntimeError(
            f"\n\n⚠  Stage 5: cannot resolve event_id column.\n"
            f"   Available columns: {list(anomaly_df.columns)}"
        )

    # Attach temporal timestamps to anomaly_df for incident grouping
    if temporal_df is not None and len(temporal_df) > 0 and "first_seen" in temporal_df.columns:
        ts_map = (
            temporal_df[[eid_col, "first_seen", "last_seen"]]
            .drop_duplicates(subset=eid_col)
            .set_index(eid_col)
        )
        if "_first_seen" not in anomaly_df.columns:
            anomaly_df["_first_seen"] = anomaly_df[eid_col].map(ts_map["first_seen"])
        if "_last_seen" not in anomaly_df.columns:
            anomaly_df["_last_seen"] = anomaly_df[eid_col].map(ts_map["last_seen"])
    else:
        if "_first_seen" not in anomaly_df.columns:
            anomaly_df["_first_seen"] = pd.NaT
        if "_last_seen" not in anomaly_df.columns:
            anomaly_df["_last_seen"] = pd.NaT

    # MEDIUM-14 FIX: ensure domain column is present in anomaly_df before
    # grouping. Stage 4's col_resolve may have renamed it or it may be missing.
    if "domain" not in anomaly_df.columns:
        dom_col = col.get("domain")
        if dom_col and dom_col in anomaly_df.columns and dom_col != "domain":
            anomaly_df = anomaly_df.copy()
            anomaly_df["domain"] = anomaly_df[dom_col]
        elif "cluster_label" in anomaly_df.columns:
            # Infer from the label prefix (e.g. "storage:upload_failed" → "storage")
            anomaly_df = anomaly_df.copy()
            anomaly_df["domain"] = (
                anomaly_df["cluster_label"]
                .fillna("other")
                .str.split(":")
                .str[0]
                .str.strip()
            )

    # Fix 7: Carry domain from raw_df (Stage 3 output, df3) into anomaly_df.
    # anomaly_df is keyed on eid_col (semantic_cluster_id / cluster_id); raw_df
    # is a per-line DataFrame with one domain value per row.  We aggregate raw_df
    # to the cluster level by taking the modal (most frequent) domain per cluster,
    # then left-join onto anomaly_df.  This populates domain correctly even when
    # Stage 4's anomaly_df has a missing, null, or numeric domain column.
    #
    # The join only fires when:
    #   (a) raw_df is provided and has both a cluster/template_id column and a
    #       non-empty "domain" column, AND
    #   (b) anomaly_df's existing "domain" column is missing or mostly null/numeric.
    _raw_df_domain_applied = False
    if raw_df is not None and len(raw_df) > 0 and "domain" in raw_df.columns:
        # Determine which column in raw_df links rows to clusters
        _raw_link_candidates = [eid_col, "semantic_cluster_id", "cluster_id", "template_id"]
        _raw_link_col = next(
            (c for c in _raw_link_candidates if c in raw_df.columns), None
        )
        if _raw_link_col is not None:
            # Check whether anomaly_df's domain column is usable.
            # Unusable when: absent, numeric dtype (domain_confidence leaked in),
            # or ≤10% of rows contain a real string domain value.
            _dom_usable = False
            if "domain" in anomaly_df.columns:
                _dom_series = anomaly_df["domain"]
                if pd.api.types.is_numeric_dtype(_dom_series):
                    # Numeric dtype → domain_confidence column, not domain labels
                    _dom_usable = False
                else:
                    import re as _re_fix7
                    _RE_NUMERIC_FIX7 = _re_fix7.compile(r'^\d+(\.\d+)?$')
                    _non_empty = (
                        _dom_series
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .pipe(lambda s: s[~s.isin(["", "nan", "unknown", "other", "0", "0.0"])])
                        .pipe(lambda s: s[~s.apply(lambda v: bool(_RE_NUMERIC_FIX7.match(v)))])
                    )
                    # Consider domain usable only if >10% of rows have a real domain value
                    _dom_usable = len(_non_empty) > 0.10 * len(anomaly_df)

            if not _dom_usable:
                # Aggregate raw_df domain to cluster level using mode
                _raw_domain_series = (
                    raw_df[[_raw_link_col, "domain"]]
                    .dropna(subset=["domain"])
                    .assign(domain=lambda d: d["domain"].astype(str).str.strip())
                    .pipe(lambda d: d[~d["domain"].isin(["", "nan", "unknown"])])
                    .groupby(_raw_link_col)["domain"]
                    .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "other")
                )
                if len(_raw_domain_series) > 0:
                    anomaly_df = anomaly_df.copy()
                    _mapped = anomaly_df[eid_col].map(_raw_domain_series)
                    _filled = _mapped.notna().sum()
                    anomaly_df["domain"] = _mapped.fillna(
                        anomaly_df.get("domain", pd.Series("other", index=anomaly_df.index))
                    )
                    _raw_df_domain_applied = True
                    logger.info(
                        "Fix 7: domain joined from raw_df (df3) into anomaly_df "
                        "(%d/%d clusters mapped via '%s')",
                        _filled, len(anomaly_df), _raw_link_col,
                    )
                    # Post-join validation: if domain is still missing, log columns
                    _post_non_empty = (
                        anomaly_df["domain"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .pipe(lambda s: s[~s.isin(["", "nan", "unknown"])])
                    )
                    if len(_post_non_empty) == 0:
                        logger.error(
                            "Fix 7: domain still missing after raw_df join. "
                            "anomaly_df columns: %s | raw_df columns: %s",
                            list(anomaly_df.columns),
                            list(raw_df.columns),
                        )
                else:
                    logger.warning(
                        "Fix 7: raw_df has 'domain' column but no usable values after "
                        "filtering. domain will fall back to _ensure_domain_column."
                    )
            else:
                logger.debug(
                    "Fix 7: anomaly_df domain already populated — raw_df join skipped."
                )

    # ── Step 5a: Incident grouping ─────────────────────────────────────
    print("\n[Step 5a] Incident grouping (success-message exclusion active)...")
    incident_events_df, incident_map = group_incidents(
        anomaly_df, temporal_df, col, full_cfg
    )

    incident_eids = (
        set(incident_events_df[eid_col].tolist())
        if len(incident_events_df) > 0
        else set()
    )

    # Identify unlinked anomalies
    anomaly_labels  = full_cfg["incident_anomaly_labels"]
    is_anomalous    = anomaly_df["anomaly_label"].isin(anomaly_labels)
    if full_cfg.get("include_burst_medium", True) and "burst_detected" in anomaly_df.columns:
        burst_bool  = _coerce_burst(anomaly_df["burst_detected"])
        is_anomalous = is_anomalous | ((anomaly_df["anomaly_label"] == "MEDIUM") & burst_bool)
    unlinked_anomalies = anomaly_df[
        is_anomalous & ~anomaly_df[eid_col].isin(incident_eids)
    ].copy()
    print(f"  Unlinked anomalies: {len(unlinked_anomalies)}")

    # ── Step 5b: Cascade chain reconstruction ─────────────────────────
    print("\n[Step 5b] Cascade chain reconstruction...")
    if len(incident_events_df) > 0:
        incident_events_df = reconstruct_cascade_chains(
            incident_events_df, temporal_df, col, full_cfg
        )
    else:
        print("  [5b] No incidents to process")

    # ── Step 5c: Timeline building ─────────────────────────────────────
    print("\n[Step 5c] Timeline building...")
    timeline_df = build_incident_timelines(raw_df, incident_events_df, col, full_cfg)

    # ── Step 5d: Incident summaries ────────────────────────────────────
    print("\n[Step 5d] Building incident summaries...")
    incidents_df, validation_failures = build_incident_summaries(
        incident_events_df,
        temporal_df,
        col,
        full_cfg,
        anomaly_df       = anomaly_df,
        cluster_manifest = cluster_manifest,
    )

    # ── Step 5d-fix6: Zero-field validation ───────────────────────────
    print("\n[Step 5d-fix6] Zero-field validation...")
    incidents_df, fix6_repairs = _validate_zero_fields(incidents_df, cluster_manifest)
    for r in fix6_repairs:
        logger.info("Fix-6: %s", r)
        print(f"  [Fix-6] ✓ {r}")
    if not fix6_repairs:
        print("  [Fix-6] No impossible zero fields detected")

    # ── Step 5e: Cross-incident deduplication ─────────────────────────
    print("\n[Step 5e] Cross-incident deduplication...")
    if len(incidents_df) > 1:
        incidents_df = _dedup_incidents(incidents_df, cluster_manifest, full_cfg)
    else:
        print(f"  [BP-S5-3] {len(incidents_df)} incident(s) — nothing to deduplicate")

    # ── Step 5f: Consistency checks ────────────────────────────────────
    print("\n[Step 5f] Cross-stage consistency checks...")
    consistency_passed, consistency_failures = _run_consistency_checks(
        incidents_df, cluster_manifest, full_cfg
    )

    # ── S5-7: Write confirmed anomaly pool for Stage 4 feedback loop ──
    # Collects sample_message strings from HIGH/CRITICAL incidents and writes
    # them to models/confirmed_anomaly_pool.json so that run_stage4() can
    # read the file on its next execution (S4-9) and exclude these templates
    # from the normal training pool, reducing false-positive re-labelling.
    _pool_path = Path(full_cfg.get(
        "confirmed_anomaly_pool_path", "models/confirmed_anomaly_pool.json"
    ))
    try:
        _pool_path.parent.mkdir(parents=True, exist_ok=True)
        _anomaly_templates: List[str] = []
        if not incidents_df.empty and "member_event_ids" in incidents_df.columns:
            # Build a quick event_id → sample_message lookup from anomaly_df
            sample_msg_map_s5: Dict[str, str] = {}
            if "sample_message" in anomaly_df.columns and eid_col in anomaly_df.columns:
                sample_msg_map_s5 = (
                    anomaly_df[[eid_col, "sample_message"]]
                    .drop_duplicates(subset=eid_col)
                    .set_index(eid_col)["sample_message"]
                    .dropna()
                    .to_dict()
                )
            for _, inc_row in incidents_df.iterrows():
                if inc_row.get("incident_severity") in ("HIGH", "CRITICAL"):
                    for eid in str(inc_row.get("member_event_ids", "")).split("|"):
                        msg = sample_msg_map_s5.get(eid.strip(), "")
                        if msg and len(str(msg)) > 10:
                            _anomaly_templates.append(str(msg)[:120])
        with open(_pool_path, "w", encoding="utf-8") as _pf:
            json.dump(
                {
                    "anomaly_templates": list(set(_anomaly_templates)),
                    "generated_at"     : datetime.now(timezone.utc).isoformat(),
                },
                _pf,
                indent=2,
            )
        logger.info(
            "S5-7: confirmed anomaly pool written to %s (%d templates)",
            _pool_path, len(set(_anomaly_templates)),
        )
        print(f"  ✓ confirmed_anomaly_pool.json written: {_pool_path} "
              f"({len(set(_anomaly_templates))} templates)")
    except Exception as _pool_exc:
        logger.warning("S5-7: failed to write confirmed_anomaly_pool.json: %s", _pool_exc)

    # ── Step 5g: Pipeline metadata ─────────────────────────────────────
    print("\n[Step 5g] Assembling pipeline metadata...")
    suppressed_anomalies: List[Dict] = []
    if "anomaly_label" in anomaly_df.columns and "sample_message" in anomaly_df.columns:
        for _, arow in anomaly_df.iterrows():
            if _is_success_message(
                _safe_str(arow.get("sample_message", "")),
                full_cfg.get("success_message_patterns", []),
            ):
                suppressed_anomalies.append({
                    "cluster_id": _safe_str(arow.get(eid_col, "")),
                    "reason"    : "success_message_pattern_match",
                })

    pipeline_metadata = _build_pipeline_metadata(
        stage1_stats         = stage1_stats,
        stage2_stats         = stage2_stats,
        stage3_stats         = stage3_stats,
        stage4_results       = stage4_results,
        incidents_df         = incidents_df,
        manifest             = cluster_manifest,
        consistency_passed   = consistency_passed,
        consistency_failures = consistency_failures,
        suppressed_anomalies = suppressed_anomalies,
        validation_failures  = validation_failures,
        cfg                  = full_cfg,
    )

    # ── Save outputs ───────────────────────────────────────────────────
    output_dir = Path(full_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    incidents_path = output_dir / full_cfg.get("incidents_filename", "incidents.csv")
    if len(incidents_df) > 0:
        incidents_df.to_csv(incidents_path, index=False)
        print(f"\n  ✓ incidents.csv written: {incidents_path}")
    else:
        print("\n  No incidents to write.")

    meta_path = output_dir / "pipeline_metadata.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(pipeline_metadata, fh, indent=2, default=str)
        print(f"  ✓ pipeline_metadata.json written: {meta_path}")
    except Exception as exc:
        logger.error("Failed to write pipeline_metadata.json: %s", exc)

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Stage 5 complete  (v7)")
    print(f"  Incidents formed          : {len(incidents_df)}")
    if len(incidents_df) > 0:
        icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
        sev_counts = incidents_df["incident_severity"].value_counts()
        for sev, cnt in sev_counts.items():
            print(f"  {icons.get(sev, '  ')} {sev:<12}: {cnt}")
    print(f"  Incident event clusters   : {len(incident_events_df)}")
    print(f"  Timeline lines            : {len(timeline_df)}")
    print(f"  Unlinked anomalies        : {len(unlinked_anomalies)}")
    print(f"  Consistency checks passed : {'✅' if consistency_passed else '⚠️  FAILURES — see pipeline_metadata.json'}")
    print(f"  Narrative grounding issues: {len(validation_failures)}")
    print(f"  Fix-6 zero-field repairs  : {len(fix6_repairs)}")
    if len(incidents_df) > 0 and "narrative_source" in incidents_df.columns:
        src_counts = incidents_df["narrative_source"].value_counts().to_dict()
        for src, cnt in src_counts.items():
            print(f"  Narrative source [{src}]: {cnt}")
    print(f"  FAISS index size          : {len(_FAISS_INCIDENT_STORE)} incidents")
    print(f"  FAISS total queries       : {_FAISS_TOTAL_QUERIES}")
    print(f"{'='*60}")

    # Top incidents preview
    if len(incidents_df) > 0:
        print("\nTop incidents preview:")
        for _, row in incidents_df.head(10).iterrows():
            start_str  = str(row.get("incident_start", ""))[:19]
            dur_str    = (f"{row['duration_minutes']:.1f}min"
                          if pd.notna(row.get("duration_minutes")) else "?min")
            sec_flag   = "🔐" if row.get("has_security_signal") else "  "
            rec_flag   = "↻" if row.get("recurrence_flag") else " "
            dom_str    = str(row.get("primary_domain", "?"))[:12]
            cids_str   = _safe_str(row.get("cluster_ids", ""))
            cids_list  = [x.strip() for x in cids_str.split("|") if x.strip()]
            cids_disp  = (
                f"{cids_list[0]}…(+{len(cids_list)-1})" if len(cids_list) > 1
                else (cids_list[0] if cids_list else "?")
            )
            total_ev   = row.get("total_event_count", 0)
            print(
                f"  {row['incident_id']}  "
                f"{row['incident_severity']:<9}  "
                f"root={str(row['root_cause_service']):<20}  "
                f"domain={dom_str:<14}  "
                f"start={start_str}  dur={dur_str}  {sec_flag}{rec_flag}"
            )
            print(f"           clusters : {cids_disp}  n={row['n_clusters']}")
            svcs_str = ", ".join(_parse_services(row.get("services_affected", "")))[:40]
            print(f"           services : {svcs_str}  n={row['n_services_affected']}")
            if cluster_manifest and int(total_ev or 0) > 0:
                print(
                    f"           events   : {int(total_ev):,} total  "
                    f"{int(row.get('error_event_count', 0)):,} errors  (manifest-sourced)"
                )
            if row.get("what_happened"):
                print(f"           what     : {row['what_happened']}")
            if row.get("error_trigger"):
                print(f"           trigger  : {row['error_trigger']}")
            if row.get("cascade_chain"):
                print(f"           chain    : {row['cascade_chain']}")
            if row.get("affected_document_ids"):
                print(f"           docIds   : {row['affected_document_ids']}")

    return {
        "incidents_df"       : incidents_df,
        "incident_events_df" : incident_events_df,
        "timeline_df"        : timeline_df,
        "unlinked_anomalies" : unlinked_anomalies,
        "incident_map"       : incident_map,
        "col_map"            : col,
        "config_used"        : full_cfg,
        "pipeline_metadata"  : pipeline_metadata,
    }


# ══════════════════════════════════════════════════════════════════════
# STANDALONE __main__ BLOCK
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Standalone test: python stage5.py <stage4_anomaly.csv> [cluster_summary.csv] [manifest.json]

    Examples
    --------
    # Minimal (no manifest — counts will be 0):
    python stage5.py outputs/stage4_anomaly.csv

    # With cluster summary and manifest (recommended):
    python stage5.py outputs/stage4_anomaly.csv outputs/cluster_summary.csv outputs/manifest.json

    # With just manifest:
    python stage5.py outputs/stage4_anomaly.csv - outputs/manifest.json
    """
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Stage 5 — Root Cause Analysis (standalone)",
    )
    parser.add_argument("anomaly_csv",       help="Path to Stage 4 anomaly CSV")
    parser.add_argument("cluster_summary",   nargs="?", default=None,
                        help="Path to cluster_summary CSV from Stage 3 (optional)")
    parser.add_argument("manifest_json",     nargs="?", default=None,
                        help="Path to manifest.json from Stage 2 (optional, recommended)")
    parser.add_argument("--raw-df",          default=None,
                        help="Path to Stage 3 per-line CSV (for timeline building)")
    parser.add_argument("--output-dir",      default="outputs",
                        help="Output directory (default: outputs/)")
    args = parser.parse_args()

    # ── Load anomaly_df ────────────────────────────────────────────────
    anomaly_path = Path(args.anomaly_csv)
    if not anomaly_path.exists():
        print(f"ERROR: anomaly CSV not found: {anomaly_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading anomaly CSV: {anomaly_path}")
    anomaly_df = pd.read_csv(anomaly_path)
    print(f"  Loaded {len(anomaly_df):,} rows, {len(anomaly_df.columns)} columns")

    # ── Load cluster_summary (optional) ───────────────────────────────
    cluster_summary_df: Optional[pd.DataFrame] = None
    if args.cluster_summary and args.cluster_summary != "-":
        cs_path = Path(args.cluster_summary)
        if cs_path.exists():
            cluster_summary_df = pd.read_csv(cs_path)
            print(f"Loaded cluster_summary: {cs_path} ({len(cluster_summary_df)} rows)")
        else:
            print(f"WARNING: cluster_summary not found: {cs_path}")

    # ── Load manifest (optional) ───────────────────────────────────────
    cluster_manifest: Optional[Dict] = None
    manifest_arg = args.manifest_json
    if not manifest_arg and args.cluster_summary == "-":
        # Positional 3rd arg was manifest when cluster_summary is "-"
        pass
    if manifest_arg and manifest_arg != "-":
        mf_path = Path(manifest_arg)
        if mf_path.exists():
            with open(mf_path, encoding="utf-8") as fh:
                cluster_manifest = json.load(fh)
            print(f"Loaded manifest: {mf_path} ({len(cluster_manifest.get('clusters', {}))} clusters)")
        else:
            print(f"WARNING: manifest not found: {mf_path}")

    # ── Load raw_df (optional) ─────────────────────────────────────────
    raw_df: Optional[pd.DataFrame] = None
    if args.raw_df:
        raw_path = Path(args.raw_df)
        if raw_path.exists():
            raw_df = pd.read_csv(raw_path)
            print(f"Loaded raw_df: {raw_path} ({len(raw_df):,} rows)")
        else:
            print(f"WARNING: raw_df not found: {raw_path}")

    # ── Build a minimal stage4_results dict ───────────────────────────
    # Stage 4's run_stage4() returns a rich dict; for standalone use we
    # reconstruct the minimum that Stage 5 needs.
    # Column resolution mirrors Stage 4's defaults.
    col_candidates_eid = ["semantic_cluster_id", "cluster_id", "event_id", "template_id"]
    eid_col_resolved   = next(
        (c for c in col_candidates_eid if c in anomaly_df.columns), None
    )
    if eid_col_resolved is None:
        print(
            "ERROR: cannot find an event_id column in anomaly_df.\n"
            f"Available columns: {list(anomaly_df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Rename to 'event_id' for consistent resolution inside Stage 5
    if eid_col_resolved != "event_id":
        anomaly_df = anomaly_df.copy()
        anomaly_df["event_id"] = anomaly_df[eid_col_resolved]

    # Attach sample_message if cluster_summary has it (needed for narrative)
    if cluster_summary_df is not None and "sample_template" in cluster_summary_df.columns:
        if "cluster_id" in cluster_summary_df.columns:
            sm_map = (
                cluster_summary_df[["cluster_id", "sample_template"]]
                .drop_duplicates(subset=["cluster_id"])
                .set_index("cluster_id")["sample_template"]
            )
            anomaly_df["sample_message"] = anomaly_df["event_id"].map(sm_map).fillna("")

    stage4_results_stub: Dict = {
        "anomaly_df" : anomaly_df,
        "temporal_df": pd.DataFrame(),   # no temporal data in standalone mode
        "col_map"    : {
            "event_id"      : "event_id",
            "service"       : "service"        if "service"        in anomaly_df.columns else None,
            "severity"      : "severity"       if "severity"       in anomaly_df.columns else None,
            "timestamp"     : "timestamp_parsed" if "timestamp_parsed" in anomaly_df.columns else None,
            "message"       : "message"        if "message"        in anomaly_df.columns else None,
            "is_noise"      : "is_noise"       if "is_noise"       in anomaly_df.columns else None,
            "event_template": "event_template" if "event_template" in anomaly_df.columns else None,
            "singleton_class": "singleton_class" if "singleton_class" in anomaly_df.columns else None,
            "domain"        : "domain"         if "domain"         in anomaly_df.columns else None,
            "event_label"   : "cluster_label"  if "cluster_label"  in anomaly_df.columns else None,
            "cluster_id"    : "cluster_id"     if "cluster_id"     in anomaly_df.columns else None,
        },
        "config_used"   : {},
        "sample_msg_map": {},
    }

    # ── Run Stage 5 ────────────────────────────────────────────────────
    results = run_stage5(
        stage4_results   = stage4_results_stub,
        raw_df           = raw_df,
        cluster_manifest = cluster_manifest,
        cfg              = {"output_dir": args.output_dir},
    )

    # ── Print summary ──────────────────────────────────────────────────
    df_inc = results["incidents_df"]
    print(f"\n{'='*60}")
    print(f"Standalone Stage 5 complete.")
    print(f"  Incidents       : {len(df_inc)}")
    print(f"  Unlinked        : {len(results['unlinked_anomalies'])}")
    print(f"  Timeline lines  : {len(results['timeline_df'])}")
    if len(df_inc) > 0:
        out_path = Path(args.output_dir) / "incidents.csv"
        print(f"  Output CSV      : {out_path}")
    meta_path_out = Path(args.output_dir) / "pipeline_metadata.json"
    print(f"  Metadata JSON   : {meta_path_out}")
    print(f"{'='*60}")