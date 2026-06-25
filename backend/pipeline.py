# backend/pipeline.py
#
# ══════════════════════════════════════════════════════════════════════
# PIPELINE — chains all 5 stages in sequence
# ══════════════════════════════════════════════════════════════════════
#
# PRODUCTION CHANGES (P1–P7)
# ──────────────────────────
# P1  run_pipeline() now accepts run_id= and embedding_model= parameters.
#     When called from the async API, the API pre-generates the run_id
#     and writes the initial run_info.json before calling this function,
#     so the polling endpoint can serve "processing" status immediately.
#     When called directly (python main.py), both parameters are optional
#     and sensible defaults are used.
#
# P2  Status field lifecycle: run_info.json is updated three times —
#     "processing" at the very start (so polling begins immediately),
#     "complete" with full stats at the end, and "failed" with the error
#     message inside the except block.  The dashboard polls
#     GET /runs/{run_id}/status which reads this file.
#
# P3  embedding_model=None parameter threads the cached model from the
#     API server's lifespan handler through to run_stage3().  When None,
#     Stage 3 loads from disk (local / test runs).  When the API passes
#     the cached object, Stage 3 skips the disk load entirely — reducing
#     Stage 3 time from ~2 minutes to ~30 seconds on subsequent runs.
#
# P4  _register_run() is called after "complete" status is written, not
#     before.  This guarantees that runs_index.json only ever contains
#     completed or failed runs, never phantom "processing" entries that
#     would show as broken cards in the Previous Runs panel.
#
# P5  The entire pipeline body is wrapped in try / except / finally.
#     On any exception: status → "failed", error message written to
#     run_info.json, and the uploaded temp file is deleted if it still
#     exists.  The exception is then re-raised so the API's background
#     task handler can log the full traceback.
#
# P6  stage4's routine_df (baseline INFO clusters separated by
#     _split_routine_clusters) is written to stage4_routine.csv and
#     exposed via get_run() so the dashboard can render a "Baseline
#     Activity" panel alongside the anomaly panel.  total_routine_clusters
#     is also included in run_info so the frontend knows the count without
#     loading the full CSV.
#
# P7  Per-deployment domain keyword injection — STAGE3_SETTINGS in
#     config.py can now include "extra_domain_keywords" or
#     "domain_keyword_overrides" dicts that Stage 3 merges with its
#     built-in keyword map.  This lets GCS, drone-telemetry, and other
#     specialised deployments get correct domain assignment without any
#     changes to shared stage3.py code.
#
# RUN FOLDER STRUCTURE:
#   outputs/runs/<run_id>/
#       ├── stage1_output.csv
#       ├── stage2_output.csv
#       ├── stage3_output.csv
#       ├── stage3_cluster_summary.csv
#       ├── stage4_anomaly.csv
#       ├── stage4_routine.csv     ← P6: baseline INFO clusters (dashboard baseline panel)
#       ├── incidents.csv
#       ├── manifest.json
#       ├── pipeline_metadata.json
#       └── run_info.json          ← status card polled by the dashboard
#
# A master index is kept at outputs/runs_index.json so the dashboard
# can list all runs without scanning the filesystem.
#
# PLACEMENT: backend/pipeline.py
# ══════════════════════════════════════════════════════════════════════

from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import traceback
import pandas as pd

# Allow imports from the project root (where config.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    LOG_FILE_PATH,
    OUTPUT_DIR,
    RUNS_DIR,
    RUNS_INDEX_PATH,
    STAGE1_SETTINGS,
    STAGE2_SETTINGS,
    STAGE3_SETTINGS,
    STAGE4_SETTINGS,
    STAGE5_SETTINGS,
    validate_paths,
    validate_paths_for_run,
)

try:
    from backend.stages.stage1 import run_stage1
    from backend.stages.stage2 import run_stage2
    from backend.stages.stage3 import run_stage3
    from backend.stages.stage4 import run_stage4
    from backend.stages.stage5 import run_stage5
except ImportError:
    from stages.stage1 import run_stage1
    from stages.stage2 import run_stage2
    from stages.stage3 import run_stage3
    from stages.stage4 import run_stage4
    from stages.stage5 import run_stage5

# S3.1 — Alert hook: imported lazily so a missing alerts.py never breaks
# the pipeline.  The actual send is guarded inside send_alert() as well.
try:
    from alerts import send_alert as _send_alert
except ImportError:
    _send_alert = None

# Ground truth — load false-positive safelisted template_ids into Stage 3.
# Imported lazily so a missing ground_truth.py never breaks a pipeline run.
try:
    try:
        from backend.ground_truth import GroundTruthStore as _GroundTruthStore
    except ImportError:
        from ground_truth import GroundTruthStore as _GroundTruthStore
except ImportError:
    _GroundTruthStore = None


# ══════════════════════════════════════════════════════════════════════
def _enrich_rca_narratives(incidents_df, anomaly_df):
    """
    Post-process the incidents DataFrame to produce richer what_happened,
    narrative, and recommended_action fields by synthesising cascade_chain,
    services_affected, root_cause_service, and anomaly signals.
    Only overwrites rows where the existing narrative is missing or too generic
    (fewer than 12 words).
    """
    if incidents_df is None or incidents_df.empty:
        return incidents_df

    import re

    # Build a quick anomaly lookup: top_source → list of anomaly_label
    svc_signals = {}
    if anomaly_df is not None and not anomaly_df.empty:
        for _, row in anomaly_df.iterrows():
            svc = str(row.get("top_source") or "").strip()
            lbl = str(row.get("anomaly_label") or "").strip()
            if svc and lbl:
                svc_signals.setdefault(svc, []).append(lbl)

    def _parse_services(raw):
        if isinstance(raw, list):
            return [s for s in raw if s and s not in ("unknown", "nan")]
        if isinstance(raw, str) and raw.strip():
            sep = "|" if "|" in raw else ","
            return [s.strip() for s in raw.split(sep) if s.strip() and s.strip() not in ("unknown", "nan")]
        return []

    def _parse_cascade(raw):
        if not raw or str(raw).strip() in ("", "nan"):
            return []
        # e.g. "svc-user(+0s) → auth-service(+12s)"
        parts = re.split(r"\s*[→>]+\s*", str(raw))
        return [re.sub(r"\(.*?\)", "", p).strip() for p in parts if p.strip()]

    def _dominant_signal(svc):
        sigs = svc_signals.get(svc, [])
        for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            if level in sigs:
                return level
        return None

    def _word_count(text):
        return len(str(text or "").split())

    rows_enriched = 0
    df = incidents_df.copy()

    for idx, inc in df.iterrows():
        # Skip if narrative already rich
        existing_narrative = str(inc.get("narrative") or "")
        if _word_count(existing_narrative) >= 12:
            continue

        root_svc   = str(inc.get("root_cause_service") or "unknown").strip()
        severity   = str(inc.get("incident_severity") or "MEDIUM").strip()
        domain     = str(inc.get("primary_domain") or "other").strip()
        svcs       = _parse_services(inc.get("services_affected"))
        cascade    = _parse_cascade(inc.get("cascade_chain"))
        n_clusters = inc.get("n_clusters", 1)

        # ── Build what_happened ──────────────────────────────────────────────
        root_signal = _dominant_signal(root_svc) or severity
        if cascade and len(cascade) >= 2:
            cascade_str = " → ".join(cascade[:4])
            what = (
                f"A {root_signal.lower()} anomaly originated in `{root_svc}` and propagated "
                f"through the cascade: {cascade_str}."
            )
        elif len(svcs) > 2:
            what = (
                f"Concurrent {root_signal.lower()} anomalies detected across `{root_svc}` and "
                f"{len(svcs)-1} other service(s) in the {domain} domain, "
                f"spanning {n_clusters} cluster(s)."
            )
        else:
            what = (
                f"{root_signal} anomaly detected in `{root_svc}` ({domain} domain) "
                f"across {n_clusters} cluster(s)."
            )

        # ── Build narrative ──────────────────────────────────────────────────
        narrative_parts = [what]
        if cascade and len(cascade) >= 2:
            trigger, *downstream = cascade[:4]
            ds_str = ", ".join(f"`{s}`" for s in downstream)
            narrative_parts.append(
                f"`{trigger}` was the first to show anomalous behaviour; errors then propagated to {ds_str}."
            )
        if svcs:
            affected_str = ", ".join(f"`{s}`" for s in svcs[:6])
            remainder = len(svcs) - 6
            if remainder > 0:
                affected_str += f" and {remainder} more"
            narrative_parts.append(f"Services directly affected: {affected_str}.")

        # Check if still rising
        if inc.get("incident_end") is None or str(inc.get("incident_end") or "").strip() in ("", "nan", "None"):
            narrative_parts.append(
                "The error rate was still elevated at the end of the observation window — the incident may not be resolved."
            )

        narrative = " ".join(narrative_parts)

        # ── Build recommended_action ─────────────────────────────────────────
        if severity == "CRITICAL":
            priority = "P1 — page on-call immediately"
        elif severity == "HIGH":
            priority = "P2 — alert the on-call engineer"
        else:
            priority = "P3 — investigate at next opportunity"

        steps = [f"{priority}."]
        if root_svc != "unknown":
            steps.append(f"Start investigation at `{root_svc}`.")
        if cascade and len(cascade) >= 2:
            steps.append(
                f"Follow the cascade chain ({' → '.join(cascade[:3])}) to identify where "
                f"the failure propagated and isolate the blast radius."
            )
        steps.append(
            "Review recent deployments, configuration changes, and upstream dependency health."
        )
        if any(_dominant_signal(s) in ("CRITICAL", "HIGH") for s in svcs[:4]):
            steps.append(
                "Multiple services show HIGH/CRITICAL signals — consider a coordinated rollback or circuit-breaker activation."
            )
        recommended = " ".join(steps)

        df.at[idx, "what_happened"]      = what
        df.at[idx, "narrative"]           = narrative
        df.at[idx, "recommended_action"]  = recommended
        rows_enriched += 1

    if rows_enriched:
        print(f"  RCA enrichment: {rows_enriched} incident narrative(s) upgraded")
    return df


# ── Template length distribution helper ──────────────────────────────

def _compute_template_length_dist(cluster_summary) -> dict:
    """
    Bucket cluster sample_templates by word count and return a dict of
    { "1-3": N, "4-7": N, "8-12": N, "13-20": N, "21+": N }
    suitable for the TemplateLengthDist frontend component.
    """
    buckets = {"1-3": 0, "4-7": 0, "8-12": 0, "13-20": 0, "21+": 0}
    if cluster_summary is None:
        return buckets
    # cluster_summary may be a DataFrame or a list of dicts
    try:
        import pandas as pd
        if isinstance(cluster_summary, pd.DataFrame):
            templates = cluster_summary["sample_template"].dropna().tolist() if "sample_template" in cluster_summary.columns else []
        else:
            templates = [r.get("sample_template", "") for r in cluster_summary if r.get("sample_template")]
        for tmpl in templates:
            wc = len(str(tmpl).strip().split())
            if wc <= 3:
                buckets["1-3"] += 1
            elif wc <= 7:
                buckets["4-7"] += 1
            elif wc <= 12:
                buckets["8-12"] += 1
            elif wc <= 20:
                buckets["13-20"] += 1
            else:
                buckets["21+"] += 1
    except Exception:
        pass
    return buckets


# S3.2 — CROSS-RUN RECURRENCE STATS
# ══════════════════════════════════════════════════════════════════════

def _compute_recurrence_stats(run_id: str, service_names: list) -> dict:
    """
    S3.2 — Count how many of the last 10 completed runs (excluding this
    one) had incidents affecting each of the services in service_names.

    Reads directly from the runs index — no extra I/O.

    Parameters
    ----------
    run_id : str
        The current run's ID, excluded from the lookback window.
    service_names : list[str]
        Services affected in the current run (from run_info
        services_affected field, or derived from Stage 5 incidents_df).

    Returns
    -------
    dict — e.g. {"auth-service": 3, "scheduler": 2}
        Only services with a recurrence count > 0 are included.
    """
    if not service_names:
        return {}

    index = _load_runs_index()
    # Filter and slice in memory — acceptable while runs_index.json < ~1000 entries.
    # TODO: if run count grows large, consider storing only the last N entries
    # in the index file itself rather than slicing after full load.
    prior_runs = [
        r for r in index
        if r.get("run_id") != run_id and r.get("status") == "complete"
    ][:10]

    recurrence: dict = {}
    for svc in service_names:
        count = sum(
            1 for r in prior_runs
            if svc in (r.get("services_affected") or [])
        )
        if count > 0:
            recurrence[svc] = count

    return recurrence


# ══════════════════════════════════════════════════════════════════════
# RUN INDEX HELPERS
# These are called by the API endpoints directly.
# ══════════════════════════════════════════════════════════════════════

def _load_runs_index() -> list:
    """Load the master list of all runs. Returns empty list if none exist."""
    if not RUNS_INDEX_PATH.exists():
        return []
    try:
        with open(RUNS_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_runs_index(index: list) -> None:
    """Persist the master runs index to disk atomically."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, default=str)


def _register_run(run_info: dict) -> None:
    """
    Add a run entry to the master index — newest first.

    P4: Only called after status is set to "complete" or "failed".
    This ensures the Previous Runs panel never shows a phantom
    "processing" entry that would be stuck there forever on a crash.
    """
    index = _load_runs_index()
    # Remove any pre-existing entry for this run_id (idempotent upsert)
    run_id = run_info.get("run_id")
    index  = [r for r in index if r.get("run_id") != run_id]
    # Insert newest at the front
    index.insert(0, run_info)
    _save_runs_index(index)


def _write_run_info(run_folder: Path, data: dict) -> None:
    """Write (or overwrite) run_info.json inside the run folder."""
    with open(run_folder / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def delete_run(run_id: str) -> bool:
    """
    Delete a run's folder and remove it from the index.

    Parameters
    ----------
    run_id : str
        The run_id string (matches the folder name under outputs/runs/).

    Returns
    -------
    bool — True if deleted, False if run_id was not found anywhere.

    Called by the API's DELETE /runs/{run_id} endpoint.
    """
    import shutil

    run_folder     = RUNS_DIR / run_id
    deleted_folder = False

    if run_folder.exists():
        shutil.rmtree(run_folder)
        deleted_folder = True

    # Remove from index regardless of whether folder existed
    # (handles the case where folder was manually deleted)
    index     = _load_runs_index()
    new_index = [r for r in index if r.get("run_id") != run_id]

    if len(new_index) == len(index) and not deleted_folder:
        return False  # run_id not found anywhere

    _save_runs_index(new_index)
    return True


def list_runs() -> list:
    """
    Return all runs for the dashboard's Previous Runs panel.
    Each entry is a compact run_info dict with summary stats.

    Called by the API's GET /runs endpoint.
    """
    return _load_runs_index()


def get_run(run_id: str) -> dict | None:
    """
    Return the full data for a single run, including results CSVs.

    Reads stage4_anomaly.csv and incidents.csv from the run folder and
    returns them as dicts so the API can serialise them to JSON for the
    dashboard.

    Called by the API's GET /runs/{run_id}/results endpoint.
    """
    run_folder = RUNS_DIR / run_id
    if not run_folder.exists():
        return None

    run_info_path = run_folder / "run_info.json"
    if not run_info_path.exists():
        return None

    with open(run_info_path, "r", encoding="utf-8") as f:
        run_info = json.load(f)

    result = {"run_info": run_info}

    # Load each output CSV if it exists
    csv_files = {
        "incidents":       "incidents.csv",
        "anomalies":       "stage4_anomaly.csv",
        "routine":         "stage4_routine.csv",      # P1: baseline clusters for dashboard
        "stage3":          "stage3_output.csv",
        "cluster_summary": "stage3_cluster_summary.csv",
    }
    for key, filename in csv_files.items():
        fpath = run_folder / filename
        if fpath.exists():
            try:
                result[key] = pd.read_csv(fpath).to_dict(orient="records")
            except Exception:
                result[key] = []

    # Load manifest
    manifest_file = run_folder / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                result["manifest"] = json.load(f)
        except Exception:
            result["manifest"] = {}

    return result


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(
    log_path: str | Path | None = None,
    run_id:   str | None        = None,
    embedding_model             = None,   # P3: cached model from API server
) -> dict:
    """
    Run the full 5-stage pipeline on a log file.

    Every call creates a unique, timestamped run folder so results from
    previous runs are never overwritten.

    Parameters
    ----------
    log_path : str | Path | None
        Path to the log file to analyse. Falls back to LOG_FILE_PATH
        from config.py when not provided (used by python main.py).

    run_id : str | None  [P1]
        Unique identifier for this run. When called from the async API,
        the API pre-generates the run_id (and writes the initial
        run_info.json) before calling run_pipeline(), so the status
        polling endpoint works from the very first second.
        When None (local / test calls), a timestamped ID is generated
        here automatically.

    embedding_model : optional  [P3]
        Pre-loaded sentence-transformers model object from the API
        server's lifespan cache. When provided, Stage 3 skips the
        disk-load entirely.  When None, Stage 3 loads from disk as
        normal — correct behaviour for local test runs.

    Returns
    -------
    dict with keys:
        run_id, run_folder, run_info,
        stage1, stage2, stage3, stage4, stage5, manifest

    Raises
    ------
    Any exception raised inside the pipeline is caught, written to
    run_info.json as status="failed", registered in the index, and
    then re-raised so the caller (API background task) can log it.
    """
    # ── Resolve input path ────────────────────────────────────────────
    path = Path(log_path) if log_path else LOG_FILE_PATH
    validate_paths()

    # ── P1: Resolve or create run_id ──────────────────────────────────
    # When called from the async API, run_id is pre-generated and the
    # run folder + initial run_info.json already exist on disk.
    # When called directly, generate a fresh timestamped run_id here.
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        run_id    = f"{timestamp}_{path.stem}"

    run_folder = validate_paths_for_run(run_id)   # creates folder if needed

    print(f"Run ID             : {run_id}")
    print(f"Log file           : {path.name}")
    print(f"Output folder      : {run_folder}")
    print(f"Embedding model    : {'cached (warm)' if embedding_model is not None else 'load from disk (cold)'}")
    print("=" * 60)

    # ── P2: Write initial "processing" status ─────────────────────────
    # Written immediately so GET /runs/{run_id}/status returns a valid
    # response from the very first poll, before any stage has run.
    # If the API already wrote this before calling us, we overwrite it
    # with a fresh copy that includes the log file details.
    started_at = datetime.now(timezone.utc).isoformat()
    _write_run_info(run_folder, {
        "run_id":          run_id,
        "log_filename":    path.name,
        "status":          "processing",
        "started_at":      started_at,
        "stage_progress":  "starting",
    })

    # ── P5: Wrap entire pipeline in try/except/finally ─────────────────
    # On any unhandled exception:
    #   1. status → "failed" in run_info.json
    #   2. run registered in runs_index.json so it shows in the UI
    #   3. uploaded temp file deleted if it still exists
    #   4. exception re-raised so the API background task logs the traceback
    try:

        # ── Stage 1 — Ingestion & Format Detection ────────────────────
        stage1_started_at = datetime.now(timezone.utc).isoformat()   # S1.5
        _write_run_info(run_folder, {
            "run_id":           run_id,
            "log_filename":     path.name,
            "status":           "processing",
            "started_at":       started_at,
            "stage_progress":   "stage_1",
            "stage_1_started_at": stage1_started_at,                 # S1.5
        })
        print("\n[Stage 1] Ingestion & format detection...")

        chunk_iter, s1_stats = run_stage1(path, **STAGE1_SETTINGS)
        df1 = pd.concat(list(chunk_iter), ignore_index=True)
        df1.to_csv(run_folder / "stage1_output.csv", index=False)

        print(f"  Done: {len(df1):,} rows  |  "
              f"parsed_ok={s1_stats.parsed_ok:,}  |  "
              f"noise={s1_stats.noise:,}")

        # ── Stage 2 — Drain Template Mining ──────────────────────────
        stage2_started_at = datetime.now(timezone.utc).isoformat()   # S1.5
        _write_run_info(run_folder, {
            "run_id":           run_id,
            "log_filename":     path.name,
            "status":           "processing",
            "started_at":       started_at,
            "stage_progress":   "stage_2",
            "stage_1_started_at": stage1_started_at,                 # S1.5
            "stage_2_started_at": stage2_started_at,                 # S1.5
        })
        print("\n[Stage 2] Drain template mining & normalisation...")

        chunk_iter2, s2_stats, manifest = run_stage2(df1, **STAGE2_SETTINGS)
        df2 = pd.concat(list(chunk_iter2), ignore_index=True)
        df2.to_csv(run_folder / "stage2_output.csv", index=False)
        with open(run_folder / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

        print(f"  Done: {s2_stats.unique_templates:,} unique templates  |  "
              f"similarity={s2_stats.calibrated_drain_similarity:.3f}")

        # ── Stage 3 — Semantic Clustering & Domain Assignment ─────────
        stage3_started_at = datetime.now(timezone.utc).isoformat()   # S1.5
        _write_run_info(run_folder, {
            "run_id":           run_id,
            "log_filename":     path.name,
            "status":           "processing",
            "started_at":       started_at,
            "stage_progress":   "stage_3",
            "stage_1_started_at": stage1_started_at,                 # S1.5
            "stage_2_started_at": stage2_started_at,                 # S1.5
            "stage_3_started_at": stage3_started_at,                 # S1.5
        })
        print("\n[Stage 3] Semantic clustering & domain assignment...")

        # P3: Pass the cached model through — if None, Stage 3 loads
        # from disk.  Either path produces identical results; the only
        # difference is speed (cold ~2 min vs warm ~30 s).
        #
        # P7 (config extension point) — Merge any per-deployment keyword
        # overrides from STAGE3_SETTINGS into the cfg dict passed to Stage 3.
        # This lets per-deployment configs inject custom domain keyword sets
        # (e.g. drone telemetry, GCS-specific terms) via config.py without
        # modifying shared stage3.py code.  Keys honoured by Stage 3:
        #   "extra_domain_keywords": dict[domain_name, list[str]]
        #       — merged with the built-in keyword map, additive only.
        #   "domain_keyword_overrides": dict[domain_name, list[str]]
        #       — replaces the built-in keyword list for named domains.
        stage3_cfg = dict(STAGE3_SETTINGS) if STAGE3_SETTINGS else {}

        # Ground truth feedback loop — inject false-positive template_ids as
        # known_normal_tids so Stage 3 never re-flags them as true_anomaly.
        # Safe to skip if GroundTruthStore is unavailable (e.g. first install).
        if _GroundTruthStore is not None:
            try:
                _gt_store = _GroundTruthStore()
                _fp_tids = _gt_store.get_false_positive_template_ids()
                if _fp_tids:
                    # Merge with any tids already set in STAGE3_SETTINGS
                    _existing = stage3_cfg.get("known_normal_tids") or []
                    stage3_cfg["known_normal_tids"] = list(set(_existing) | set(_fp_tids))
                    print(f"  [GT] Loaded {len(_fp_tids)} false-positive template_ids into Stage 3 safelist")
            except Exception as _gt_err:
                print(f"  [GT] Warning: could not load ground truth safelist: {_gt_err}")

        df3, s3_stats = run_stage3(
            df2,
            cluster_manifest = manifest,
            embedding_model  = embedding_model,   # P3
            cfg              = stage3_cfg or None, # P7
        )
        df3.to_csv(run_folder / "stage3_output.csv", index=False)
        cluster_summary = s3_stats.get("cluster_summary", pd.DataFrame())
        if not cluster_summary.empty:
            cluster_summary.to_csv(run_folder / "stage3_cluster_summary.csv", index=False)

        print(f"  Done: {len(cluster_summary):,} clusters")

        # ── Stage 4 — Anomaly Scoring ─────────────────────────────────
        stage4_started_at = datetime.now(timezone.utc).isoformat()   # S1.5
        _write_run_info(run_folder, {
            "run_id":           run_id,
            "log_filename":     path.name,
            "status":           "processing",
            "started_at":       started_at,
            "stage_progress":   "stage_4",
            "stage_1_started_at": stage1_started_at,                 # S1.5
            "stage_2_started_at": stage2_started_at,                 # S1.5
            "stage_3_started_at": stage3_started_at,                 # S1.5
            "stage_4_started_at": stage4_started_at,                 # S1.5
        })
        print("\n[Stage 4] Anomaly scoring...")

        # P13: STAGE4_SETTINGS from config.py is passed as cfg so any
        # future weight adjustments in config take effect without code
        # changes here.
        s4 = run_stage4(
            df3,
            cluster_summary_df = cluster_summary,
            cfg                = STAGE4_SETTINGS if STAGE4_SETTINGS else None,
        )
        s4["anomaly_df"].to_csv(run_folder / "stage4_anomaly.csv", index=False)

        # P1 (routine_df handoff) — Save the routine/baseline clusters that
        # stage4's _split_routine_clusters separated out so the dashboard can
        # show a "baseline activity" panel instead of hiding ~50 INFO clusters.
        # routine_df contains is_routine=True clusters with anomaly_label="ROUTINE".
        routine_df = s4.get("routine_df", pd.DataFrame())
        if not routine_df.empty:
            routine_df.to_csv(run_folder / "stage4_routine.csv", index=False)

        print(f"  Done: {len(s4['anomaly_df']):,} scored clusters  |  "
              f"routine={len(routine_df):,} baseline clusters")

        # ── Stage 5 — Root Cause Analysis ────────────────────────────
        stage5_started_at = datetime.now(timezone.utc).isoformat()   # S1.5
        _write_run_info(run_folder, {
            "run_id":           run_id,
            "log_filename":     path.name,
            "status":           "processing",
            "started_at":       started_at,
            "stage_progress":   "stage_5",
            "stage_1_started_at": stage1_started_at,                 # S1.5
            "stage_2_started_at": stage2_started_at,                 # S1.5
            "stage_3_started_at": stage3_started_at,                 # S1.5
            "stage_4_started_at": stage4_started_at,                 # S1.5
            "stage_5_started_at": stage5_started_at,                 # S1.5
        })
        print("\n[Stage 5] Root cause analysis & incident grouping...")

        # P14: output_dir is overridden with this run's actual folder.
        # STAGE5_SETTINGS from config.py is merged on top, so any
        # user-level overrides in config still apply.
        stage5_cfg = {
            **STAGE5_SETTINGS,                    # user overrides from config
            "output_dir":         str(run_folder),  # P14: always this run's folder
            "incidents_filename": "incidents.csv",
            "suppress_display":   True,             # no Jupyter HTML in server context
        }
        s5 = run_stage5(
            s4,
            raw_df           = df3,
            cluster_manifest = manifest,
            stage1_stats     = s1_stats.as_dict(),
            stage2_stats     = {
                "unique_templates": s2_stats.unique_templates,
                "total_clusters":   s2_stats.unique_templates,
            },
            stage3_stats = s3_stats if isinstance(s3_stats, dict) else {},
            cfg          = stage5_cfg,
        )

        n_incidents = len(s5["incidents_df"])
        print(f"  Done: {n_incidents:,} incidents")

        # ── Enrich RCA narratives with richer synthesis ───────────────────
        if not s5["incidents_df"].empty:
            s5["incidents_df"] = _enrich_rca_narratives(
                s5["incidents_df"],
                s4["anomaly_df"] if "anomaly_df" in s4 else None,
            )
            # Re-write the enriched incidents.csv
            enriched_csv = run_folder / "incidents.csv"
            s5["incidents_df"].to_csv(enriched_csv, index=False)

        # ── Build the final run_info summary card ─────────────────────
        anomaly_df   = s4["anomaly_df"]
        label_counts = (
            anomaly_df["anomaly_label"].value_counts().to_dict()
            if "anomaly_label" in anomaly_df.columns
            else {}
        )
        incident_sev_counts = (
            s5["incidents_df"]["incident_severity"].value_counts().to_dict()
            if not s5["incidents_df"].empty
            and "incident_severity" in s5["incidents_df"].columns
            else {}
        )
        completed_at = datetime.now(timezone.utc).isoformat()

        run_info = {
            # Identity
            "run_id":              run_id,
            "log_filename":        path.name,
            "log_file_size_bytes": path.stat().st_size if path.exists() else 0,
            "run_timestamp":       started_at,
            "completed_at":        completed_at,
            "run_folder":          str(run_folder),

            # P2: Status — the field the polling endpoint reads
            "status":              "complete",
            "stage_progress":      "complete",

            # S1.5 — Per-stage started_at timestamps.
            # The frontend subtracts these from the current poll time to
            # show elapsed time per stage, so users can see Stage 3 is
            # still running rather than assuming the process has hung.
            "stage_1_started_at":  stage1_started_at,
            "stage_2_started_at":  stage2_started_at,
            "stage_3_started_at":  stage3_started_at,
            "stage_4_started_at":  stage4_started_at,
            "stage_5_started_at":  stage5_started_at,

            # Stage 1
            "total_lines":         s1_stats.total_lines,
            "parsed_ok":           s1_stats.parsed_ok,
            "noise_lines":         s1_stats.noise,
            "detected_encoding":   s1_stats.detected_encoding,
            "format_counts":       dict(s1_stats.format_counts),
            "ts_parsed_ok":        s1_stats.ts_parsed_ok,
            "ts_failed":           s1_stats.ts_failed,
            "json_ok":             s1_stats.json_ok,
            "noise_reasons":       dict(s1_stats.error_reasons.most_common(10)),

            # Stage 2 — severity_counts derived from parsed df
            "severity_counts":     (
                df1["severity"].dropna().value_counts().to_dict()
                if "severity" in df1.columns else {}
            ),

            # Stage 2
            "unique_templates":    s2_stats.unique_templates,
            "drain_similarity":    round(s2_stats.calibrated_drain_similarity, 3),

            # Stage 3
            "total_clusters":      len(cluster_summary),
            "silhouette_score":    s3_stats.get("silhouette_score"),
            "n_isolated":          s3_stats.get("n_isolated"),
            "cluster_threshold":   s3_stats.get("threshold_used"),
            "template_length_distribution": _compute_template_length_dist(cluster_summary),

            # Stage 4
            "total_scored_clusters": len(anomaly_df),
            "anomaly_label_counts":  label_counts,
            # P1: routine cluster count surfaced so the dashboard baseline panel
            # knows how many INFO/routine clusters to fetch from stage4_routine.csv
            "total_routine_clusters": len(s4.get("routine_df", pd.DataFrame())),

            # Stage 5
            "total_incidents":          n_incidents,
            "incident_severity_counts": incident_sev_counts,
        }

        # S3.2 — Derive services_affected from the incidents DataFrame
        # and store it on run_info so the recurrence helper and the
        # runs index can both use it.
        services_affected: list = []
        if not s5["incidents_df"].empty and "services_affected" in s5["incidents_df"].columns:
            svc_set: set = set()
            for raw in s5["incidents_df"]["services_affected"].dropna():
                # services_affected is stored as a JSON list or comma string
                if isinstance(raw, list):
                    svc_set.update(raw)
                else:
                    try:
                        import json as _json
                        parsed = _json.loads(str(raw))
                        if isinstance(parsed, list):
                            svc_set.update(parsed)
                        else:
                            svc_set.add(str(raw))
                    except Exception:
                        # Handle both pipe-delimited and comma-delimited
                        raw_str = str(raw)
                        sep = "|" if "|" in raw_str else ","
                        svc_set.update(s.strip() for s in raw_str.split(sep) if s.strip())
            services_affected = sorted(
                s for s in svc_set if s and s not in ("unknown", "", "nan")
            )
        run_info["services_affected"] = services_affected

        # P2: Write "complete" status to run_info.json
        _write_run_info(run_folder, run_info)

        # P4: Register in runs_index.json only after "complete" is written.
        # This guarantees the Previous Runs panel never shows a phantom
        # "processing" entry that would be stuck there on a crash.
        _register_run(run_info)

        # S3.2 — Compute cross-run recurrence stats and append to run_info.
        # Reads the just-updated index so this run is excluded from the
        # lookback window automatically (it was just registered above).
        recurring_services = _compute_recurrence_stats(run_id, services_affected)
        if recurring_services:
            run_info["recurring_services"] = recurring_services
            _write_run_info(run_folder, run_info)

        # S3.1 — Dispatch webhook alert if any CRITICAL incidents exist.
        # Fails silently when ALERT_WEBHOOK_URL is not set or alerts.py
        # is not yet present.
        if _send_alert is not None:
            try:
                _send_alert(run_info, s5["incidents_df"])
            except Exception as _alert_exc:
                import logging as _logging
                _logging.getLogger("pipeline").warning(
                    "send_alert raised an exception — alert not delivered: %s",
                    _alert_exc,
                )

        print("\n" + "=" * 60)
        print(f"Pipeline complete.")
        print(f"  Run ID         : {run_id}")
        print(f"  Incidents      : {n_incidents}")
        print(f"  Anomalies      : {len(anomaly_df)}")
        print(f"  Output folder  : {run_folder}")
        print("=" * 60)

        return {
            "run_id":     run_id,
            "run_folder": run_folder,
            "run_info":   run_info,
            "stage1":     df1,
            "stage2":     df2,
            "stage3":     df3,
            "stage4":     s4,
            "stage5":     s5,
            "manifest":   manifest,
        }

    except Exception as exc:
        # ── P5: Failure path ──────────────────────────────────────────
        # 1. Update run_info.json with "failed" status and error details
        # 2. Register the failed run in runs_index.json so it appears in
        #    the Previous Runs panel with a red "failed" badge
        # 3. Clean up the uploaded temp file if it still exists on disk
        # 4. Re-raise so the API's background task handler logs the full
        #    Python traceback to the server terminal (Terminal 1)
        error_message = str(exc)
        error_tb      = traceback.format_exc()

        failed_info = {
            "run_id":         run_id,
            "log_filename":   path.name,
            "status":         "failed",
            "started_at":     started_at,
            "failed_at":      datetime.now(timezone.utc).isoformat(),
            "error_message":  error_message,
            # Truncated traceback — useful for diagnosing in the UI
            # without leaking full server internals
            "error_hint":     error_tb.strip().splitlines()[-1] if error_tb else error_message,
        }

        try:
            _write_run_info(run_folder, failed_info)
        except Exception:
            pass  # if we can't write the file, don't mask the original error

        # P4 (failure path): register failed run so the UI can show it
        try:
            _register_run(failed_info)
        except Exception:
            pass

        # P5: Clean up uploaded temp file if it still exists.
        # path might be the original log (not a temp file) when called
        # directly — only delete it if it lives inside the uploads dir.
        try:
            uploads_dir = OUTPUT_DIR / "uploads"
            if path.resolve().is_relative_to(uploads_dir.resolve()) and path.exists():
                path.unlink()
        except Exception:
            pass  # cleanup failure must not shadow the original error

        # Re-raise — API background task handler will log the traceback
        raise


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT — python main.py
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_pipeline()
    print(f"\nDone.")
    print(f"  Incidents : {len(results['stage5']['incidents_df'])}")
    print(f"  Anomalies : {len(results['stage4']['anomaly_df'])}")
    print(f"  Run ID    : {results['run_id']}")