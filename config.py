"""
config.py  (project root)
==========================
Single source of truth for all paths, settings, and constants used by
the AI Log Monitoring & Anomaly Detection pipeline.

Usage
-----
    from config import (
        LOG_FILE_PATH, OUTPUT_DIR, RUNS_DIR,
        STAGE1_SETTINGS, STAGE2_SETTINGS,
        validate_paths,
    )

Rules
-----
- All paths are pathlib.Path objects (never bare strings).
- No stage file should import from another stage — only from config.py.
- Run this file directly to verify your environment:
      python config.py
"""

from pathlib import Path
import os

# ══════════════════════════════════════════════════════════════════════
# PROJECT ROOT
# ══════════════════════════════════════════════════════════════════════
# config.py lives at the project root. All other paths are derived from
# here so the project can be moved without breaking anything.

PROJECT_ROOT = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════
# LOG FILE
# ══════════════════════════════════════════════════════════════════════
# Default log file to process when no path is passed to run_pipeline().
# Override at runtime by passing a path to run_pipeline(log_path=...).
# Can also be overridden via the LOG_FILE_PATH environment variable
# so that deployment config never requires editing this file.

LOG_FILE_PATH = Path(
    os.environ.get("LOG_FILE_PATH", str(PROJECT_ROOT / "backend" / "klares-app-7.log"))
)

# Alternate log file (used in manual testing)
# LOG_FILE_PATH = PROJECT_ROOT / "klares-app-7.log"


# ══════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORIES
# ══════════════════════════════════════════════════════════════════════
# Both OUTPUT_DIR and UPLOAD_TEMP_DIR are overridable via environment
# variables so that server deployments never need to edit this file.

# Top-level outputs folder — created automatically by validate_paths()
OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
)

# Each pipeline run gets its own subfolder under RUNS_DIR.
# The folder name is the run_id, e.g.:
#     outputs/runs/2026-04-20_143022_campaign-template-generator-6/
RUNS_DIR = OUTPUT_DIR / "runs"

# Master index of all runs — the Phase 2 API reads this to list runs
# without scanning every subfolder.
RUNS_INDEX_PATH = OUTPUT_DIR / "runs_index.json"

# Ground truth SQLite database — stores user TP/FP feedback per cluster ref.
# Created automatically on first API feedback call.
# False-positive template_ids are fed into Stage 3's known_normal_tids
# safelist on every subsequent pipeline run to suppress repeat false positives.
GROUND_TRUTH_DB_PATH = OUTPUT_DIR / "ground_truth.db"


# ══════════════════════════════════════════════════════════════════════
# API UPLOAD CONSTANTS                                        P6 P7 P8
# ══════════════════════════════════════════════════════════════════════
# These are imported by api.py so that upload validation logic lives
# in exactly one place. To change any limit or add a new extension,
# edit here only — no changes needed elsewhere.
# All numeric limits are overridable via environment variables.

# P6: Maximum accepted upload size. Requests larger than this are
# rejected with HTTP 413 before any file I/O occurs.
# 100 MB expressed in bytes — change the leading number to adjust.
MAX_UPLOAD_SIZE_BYTES: int = int(
    os.environ.get("MAX_UPLOAD_SIZE_BYTES", 100 * 1024 * 1024)  # 100 MB
)

# P7: Temporary landing directory for uploaded log files.
# The API writes the incoming file here; pipeline.py reads from here
# and deletes the file when the run completes (or fails).
# Derived from OUTPUT_DIR so it moves with the project automatically.
# Created by validate_paths() on startup — never needs manual mkdir.
UPLOAD_TEMP_DIR: Path = Path(
    os.environ.get("UPLOAD_TEMP_DIR", str(OUTPUT_DIR / "uploads"))
)

# P8: File extensions accepted by the upload endpoint.
# Validated against the uploaded filename's suffix (lowercased) before
# any file is written to disk. Add '.gz' here if compressed log support
# is added in future — no other file needs to change.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".log", ".txt"})


# ══════════════════════════════════════════════════════════════════════
# JOB QUEUE CONSTANTS                                             S1.2
# ══════════════════════════════════════════════════════════════════════
# Controls how many pipeline runs can be queued and executing at once.
# The API's upload endpoint raises QueueFullError if MAX_QUEUE_DEPTH is
# already reached. MAX_CONCURRENT_RUNS is reserved for future use when
# multiple workers are supported; currently the queue enforces 1.

MAX_QUEUE_DEPTH: int = int(os.environ.get("MAX_QUEUE_DEPTH", 5))
MAX_CONCURRENT_RUNS: int = int(os.environ.get("MAX_CONCURRENT_RUNS", 1))


# ══════════════════════════════════════════════════════════════════════
# PRE-FLIGHT VALIDATION CONSTANTS                          S1.3  S1.4
# ══════════════════════════════════════════════════════════════════════
# Used by preflight.py before any pipeline stage runs.
# Reject thresholds — files failing these checks are returned to the
# caller with a descriptive reason string rather than being processed.

# S1.3 — Content quality thresholds
# Minimum number of non-empty lines required to proceed.
PREFLIGHT_MIN_LINES: int = int(os.environ.get("PREFLIGHT_MIN_LINES", 10))

# Minimum fraction of lines that must parse as recognisable log format.
PREFLIGHT_MIN_PARSE_RATE: float = float(
    os.environ.get("PREFLIGHT_MIN_PARSE_RATE", 0.15)
)

# Maximum fraction of lines that may contain binary noise.
PREFLIGHT_MAX_BINARY_RATE: float = float(
    os.environ.get("PREFLIGHT_MAX_BINARY_RATE", 0.60)
)

# S1.4 — Size / line-count gate
# Files exceeding this line count are rejected before Stage 1 runs.
# At ~100 bytes/line, 2M lines ≈ 200 MB of uncompressed text.
MAX_LOG_LINES: int = int(os.environ.get("MAX_LOG_LINES", 2_000_000))


# ══════════════════════════════════════════════════════════════════════
# DATA RETENTION CONSTANTS                                 S2.1  S2.2
# ══════════════════════════════════════════════════════════════════════
# Used by retention.py to expire old run folders automatically.
# apply_retention_policy() is called once on API startup.

# Runs older than this many days are deleted automatically.
RETENTION_DAYS: int = int(os.environ.get("RETENTION_DAYS", 30))

# If the total number of stored runs exceeds this, oldest runs are
# deleted even if they are within the RETENTION_DAYS window.
MAX_STORED_RUNS: int = int(os.environ.get("MAX_STORED_RUNS", 100))


# ══════════════════════════════════════════════════════════════════════
# ALERT / NOTIFICATION CONSTANTS                                  S3.1
# ══════════════════════════════════════════════════════════════════════
# Used by alerts.py to dispatch webhook notifications on CRITICAL runs.
# If ALERT_WEBHOOK_URL is empty, alerts.py logs a warning and returns
# silently — no exception is raised.

ALERT_WEBHOOK_URL: str = os.environ.get("ALERT_WEBHOOK_URL", "")


# ══════════════════════════════════════════════════════════════════════
# LLM PROVIDER SETTINGS
# ══════════════════════════════════════════════════════════════════════
# Controls which LLM backend stages 4 and 5 use for:
#   - Stage 4 ACC-5: borderline severity-context classification
#   - Stage 5 S5-ML-1: root-cause narrative generation
#
# Switch providers with ZERO code changes — env vars only:
#
#   # Use Ollama (default — local / dev):
#   export LLM_PROVIDER=ollama          # or leave unset
#   export OLLAMA_MODEL=phi4:14b        # or leave unset for default (phi4:14b)
#   export OLLAMA_BASE_URL=http://localhost:11434   # or leave unset
#
#   # Switch to Anthropic (production deployment):
#   export LLM_PROVIDER=anthropic
#   export ANTHROPIC_API_KEY=sk-ant-...
#
# Ollama must be running as a separate process before the pipeline starts.
# Start it with:  ollama serve

LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "ollama")
"""Accepted values: "ollama" (default) | "anthropic"."""

OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "phi4:14b")
"""Ollama model tag to use. Override via OLLAMA_MODEL env var."""

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
"""Ollama server base URL. Override via OLLAMA_BASE_URL env var."""

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
"""Only used when LLM_PROVIDER="anthropic". Override via ANTHROPIC_API_KEY env var."""


# ══════════════════════════════════════════════════════════════════════
# STAGE SETTINGS
# ══════════════════════════════════════════════════════════════════════
# These dicts are unpacked with ** when calling each stage function.
# Keep only the settings that differ from each stage's internal defaults
# — stages have safe defaults for everything not listed here.

STAGE1_SETTINGS = {
    "batch_size"    : 10_000,
    "encoding"      : "utf-8",
    "errors"        : "replace",
    "default_tz"    : "UTC",
    "allow_relaxed" : True,
}

STAGE2_SETTINGS = {
    "drain_similarity": 0.5,
}

# Stage 3, 4, 5 accept optional cfg= dicts — leave empty to use their
# internal defaults, or add overrides here as needed.
STAGE3_SETTINGS: dict = {}

# LLM provider keys are injected into both STAGE4_SETTINGS and
# STAGE5_SETTINGS so each stage can read them via cfg.get("llm_provider"),
# cfg.get("ollama_model"), etc.  Stages fall back to their own internal
# defaults for any key not present here, so existing behaviour is
# preserved when these values match the stage defaults.
STAGE4_SETTINGS: dict = {
    "llm_provider"      : LLM_PROVIDER,
    "ollama_model"      : OLLAMA_MODEL,
    "ollama_base_url"   : OLLAMA_BASE_URL,
    "anthropic_api_key" : ANTHROPIC_API_KEY,
    # Stage 4 ACC-5 mirrors — picked up by _llm_classify_borderline()
    "acc5_ollama_model"    : OLLAMA_MODEL,
    "acc5_ollama_base_url" : OLLAMA_BASE_URL,
}

STAGE5_SETTINGS: dict = {
    "llm_provider"      : LLM_PROVIDER,
    "ollama_model"      : OLLAMA_MODEL,
    "ollama_base_url"   : OLLAMA_BASE_URL,
    "anthropic_api_key" : ANTHROPIC_API_KEY,
}


# ══════════════════════════════════════════════════════════════════════
# PER-RUN OUTPUT PATHS  (generated dynamically inside pipeline.py)
# ══════════════════════════════════════════════════════════════════════
# These helpers build paths inside a run's dedicated folder.
# pipeline.py calls them after creating the run folder.

def run_output_dir(run_id: str) -> Path:
    """Return the dedicated output folder for a specific run."""
    return RUNS_DIR / run_id


def stage_csv_path(run_id: str, stage: int) -> Path:
    """Return the CSV output path for a given stage inside a run folder."""
    names = {
        1: "stage1_parsed.csv",
        2: "stage2_templates.csv",
        3: "stage3_classified.csv",
        4: "stage4_anomaly.csv",
        5: "stage5_incidents.csv",
    }
    return run_output_dir(run_id) / names[stage]


def manifest_path(run_id: str) -> Path:
    """Return the manifest.json path for a specific run."""
    return run_output_dir(run_id) / "manifest.json"


def pipeline_metadata_path(run_id: str) -> Path:
    """Return the pipeline_metadata.json path for a specific run."""
    return run_output_dir(run_id) / "pipeline_metadata.json"


def run_info_path(run_id: str) -> Path:
    """Return the run_info.json path (dashboard summary card) for a run."""
    return run_output_dir(run_id) / "run_info.json"


# ══════════════════════════════════════════════════════════════════════
# LEGACY FIXED OUTPUT PATHS
# ══════════════════════════════════════════════════════════════════════
# These were used before per-run folders were introduced.
# Still exported so that any old code referencing them doesn't break,
# but new code should use run_output_dir() + stage_csv_path() instead.

STAGE1_OUTPUT_PATH   = OUTPUT_DIR / "stage1_parsed.csv"
STAGE2_OUTPUT_PATH   = OUTPUT_DIR / "stage2_templates.csv"
STAGE3_OUTPUT_PATH   = OUTPUT_DIR / "stage3_classified.csv"
STAGE4_OUTPUT_PATH   = OUTPUT_DIR / "stage4_anomaly.csv"
STAGE5_OUTPUT_PATH   = OUTPUT_DIR / "stage5_incidents.csv"
MANIFEST_OUTPUT_PATH = OUTPUT_DIR / "manifest.json"


# ══════════════════════════════════════════════════════════════════════
# VALIDATE PATHS
# ══════════════════════════════════════════════════════════════════════

def validate_paths() -> None:
    """
    Create all output directories and verify the log file exists.
    Call this at the start of every pipeline run.

    Creates
    -------
    - OUTPUT_DIR and all parents
    - RUNS_DIR
    - UPLOAD_TEMP_DIR  [P7] — so the API can write uploads immediately
      on startup without a separate mkdir call.

    Raises
    ------
    FileNotFoundError
        If LOG_FILE_PATH does not point to an existing file.
    """
    # Create output directories (safe if they already exist)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)   # P7

    if not LOG_FILE_PATH.exists():
        raise FileNotFoundError(
            f"\n\n⚠  Log file not found: {LOG_FILE_PATH}\n"
            f"   Update LOG_FILE_PATH in config.py to point to your log file.\n"
            f"   Available .log files in project root:\n"
            + "\n".join(
                f"     {p.name}"
                for p in PROJECT_ROOT.glob("*.log")
            ) or "     (none found)"
        )


def validate_paths_for_run(run_id: str) -> Path:
    """
    Create the dedicated folder for a specific run and return its path.
    Also calls validate_paths() to ensure all top-level dirs exist.

    Parameters
    ----------
    run_id : str
        The run identifier, e.g. '2026-04-20_143022_campaign-template-generator-6'

    Returns
    -------
    Path
        The newly created (or already existing) run folder.
    """
    validate_paths()
    folder = run_output_dir(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ══════════════════════════════════════════════════════════════════════
# SELF-TEST  —  python config.py
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  config.py — environment check")
    print("=" * 60)

    print(f"\n  PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"  LOG_FILE_PATH : {LOG_FILE_PATH}")

    log_ok = LOG_FILE_PATH.exists()
    print(f"  Log file      : {'✅  found' if log_ok else '❌  NOT FOUND — update LOG_FILE_PATH'}")

    print(f"\n  OUTPUT_DIR    : {OUTPUT_DIR}")
    print(f"  RUNS_DIR      : {RUNS_DIR}")
    print(f"  RUNS_INDEX    : {RUNS_INDEX_PATH}")

    # P6 P7 P8: upload constants
    print(f"\n  ── Upload Constants ──────────────────────────────────")
    print(f"  MAX_UPLOAD_SIZE_BYTES : {MAX_UPLOAD_SIZE_BYTES:,}  ({MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB)")
    print(f"  UPLOAD_TEMP_DIR       : {UPLOAD_TEMP_DIR}")
    print(f"  ALLOWED_EXTENSIONS    : {sorted(ALLOWED_EXTENSIONS)}")

    # S1.2: queue constants
    print(f"\n  ── Job Queue Constants ───────────────────────────────")
    print(f"  MAX_QUEUE_DEPTH    : {MAX_QUEUE_DEPTH}")
    print(f"  MAX_CONCURRENT_RUNS: {MAX_CONCURRENT_RUNS}")

    # S1.3 / S1.4: pre-flight constants
    print(f"\n  ── Pre-flight Validation Constants ───────────────────")
    print(f"  PREFLIGHT_MIN_LINES      : {PREFLIGHT_MIN_LINES}")
    print(f"  PREFLIGHT_MIN_PARSE_RATE : {PREFLIGHT_MIN_PARSE_RATE}")
    print(f"  PREFLIGHT_MAX_BINARY_RATE: {PREFLIGHT_MAX_BINARY_RATE}")
    print(f"  MAX_LOG_LINES            : {MAX_LOG_LINES:,}")

    # S2.1 / S2.2: retention constants
    print(f"\n  ── Data Retention Constants ──────────────────────────")
    print(f"  RETENTION_DAYS  : {RETENTION_DAYS}")
    print(f"  MAX_STORED_RUNS : {MAX_STORED_RUNS}")

    # S3.1: alert constants
    print(f"\n  ── Alert / Notification Constants ────────────────────")
    print(f"  ALERT_WEBHOOK_URL : {'(set)' if ALERT_WEBHOOK_URL else '(not set — alerts disabled)'}")

    # LLM provider settings
    print(f"\n  ── LLM Provider Settings ─────────────────────────────")
    print(f"  LLM_PROVIDER      : {LLM_PROVIDER}")
    if LLM_PROVIDER == "ollama":
        print(f"  OLLAMA_MODEL      : {OLLAMA_MODEL}")
        print(f"  OLLAMA_BASE_URL   : {OLLAMA_BASE_URL}")
        # Quick liveness probe so the operator can see at a glance
        try:
            import urllib.request as _urllib_req
            _probe = _urllib_req.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            print(f"  Ollama server     : ✅  reachable (HTTP {_probe.status})")
        except Exception as _oe:
            print(f"  Ollama server     : ❌  NOT reachable ({_oe})")
            print(f"                       Run:  ollama serve")
    else:
        print(f"  ANTHROPIC_API_KEY : {'(set)' if ANTHROPIC_API_KEY else '(not set — set ANTHROPIC_API_KEY env var)'}")

    # Stage settings
    print(f"\n  ── Stage Settings ────────────────────────────────────")
    print(f"  STAGE1_SETTINGS : {STAGE1_SETTINGS}")
    print(f"  STAGE2_SETTINGS : {STAGE2_SETTINGS}")

    # Try creating output dirs
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n  ✅  Output directories created / verified")
    except Exception as e:
        print(f"\n  ❌  Could not create output directories: {e}")

    # Run full validation (will raise if log file missing)
    print()
    try:
        validate_paths()
        print("  ✅  validate_paths() passed — ready to run pipeline")
    except FileNotFoundError as e:
        print(f"  ⚠  {e}")

    # Show available log files
    log_files = list(PROJECT_ROOT.glob("*.log"))
    if log_files:
        print(f"\n  Available log files in project root:")
        for lf in log_files:
            size_mb = lf.stat().st_size / (1024 * 1024)
            print(f"    {lf.name}  ({size_mb:.1f} MB)")
    else:
        print("\n  No .log files found in project root")

    print(f"\n  Run from project root with:  python config.py")
    print("=" * 60)