# backend/api.py
#
# ══════════════════════════════════════════════════════════════════════
# FASTAPI BACKEND — Phase 2 (BackgroundTasks edition)
# ══════════════════════════════════════════════════════════════════════
#
# Integration notes vs the doc's generic template
# ─────────────────────────────────────────────────
# 1. DELETE /runs/{run_id}  — calls pipeline.delete_run(run_id) instead
#    of reimplementing shutil.rmtree.  delete_run() handles both the
#    folder AND the runs_index.json entry atomically in one call.
#
# 2. GET /runs/{run_id}/results  — reads "incidents.csv" (the filename
#    Stage 5 actually writes) not "stage5_incidents.csv" (the doc's
#    generic template name).  pipeline.get_run() already knows all the
#    correct filenames, so /results delegates to it entirely.
#
# 3. _run_pipeline_task error handling  — pipeline.py already writes
#    status="failed" to run_info.json and registers the run in the index
#    before re-raising.  The task handler here just catches and logs;
#    it does NOT re-write run_info.json (that would overwrite pipeline's
#    richer error details with a blander dict).
#
# 4. Model caching  — _CACHED_MODEL is loaded once in the lifespan
#    handler and passed through run_pipeline(embedding_model=...) so
#    Stage 3 skips the ~2-minute disk load on every subsequent run.
#
# 5. Temp-file cleanup  — pipeline.py's except block already deletes the
#    uploaded file when it lives under outputs/uploads/.  The task handler
#    here has a belt-and-braces finally: clause that unlinks the file if
#    it still exists after pipeline returns or raises.
#
# UPGRADE PATH (when ready):
#    Replace _run_pipeline_task + BackgroundTasks with log_queue.py:
#      - Import enqueue_run, pipeline_worker, QueueFullError
#      - Start pipeline_worker() as asyncio.create_task() in lifespan
#      - In /analyze: call enqueue_run(job) instead of background_tasks.add_task()
#      - Catch QueueFullError → HTTP 429
#
# PLACEMENT: backend/api.py
# RUN FROM PROJECT ROOT: uvicorn backend.api:app --reload --port 8000
# ══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Allow imports from the project root (where config.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RUNS_DIR, RUNS_INDEX_PATH, validate_paths_for_run
from backend.pipeline import delete_run as pipeline_delete_run
from backend.pipeline import get_run, list_runs, run_pipeline
try:
    from backend.stages.stage3 import _load_embedding_model
except ImportError:
    from stages.stage3 import _load_embedding_model

logger = logging.getLogger("api")

# ── Model cache (module-level — shared across all requests) ───────────
_CACHED_MODEL = None


# ══════════════════════════════════════════════════════════════════════
# LIFESPAN — load the embedding model once at server startup
# ══════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _CACHED_MODEL
    logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
    print("Loading embedding model — this takes ~30-60 s on first run...")
    _CACHED_MODEL = _load_embedding_model("all-MiniLM-L6-v2")
    print("Embedding model loaded and cached.  Ready to serve requests.")
    logger.info("Embedding model loaded.")
    yield
    # Shutdown cleanup (nothing needed for this model)
    logger.info("API shutting down.")


# ══════════════════════════════════════════════════════════════════════
# APP + CORS
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Log Monitor API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS MUST be added before any endpoint definitions.
# Without this, React on localhost:3000 is blocked by the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════
# JSON SERIALISATION HELPER
# ══════════════════════════════════════════════════════════════════════

def df_to_json_safe(df: pd.DataFrame) -> list:
    """
    Convert a DataFrame to a JSON-safe list of dicts.

    Handles:
    - datetime64 / datetimetz columns  → string
    - pd.NA / pd.NaT / np.nan         → None
    - numpy int64                      → handled automatically by to_dict()
    - pd.StringDtype columns           → handled by the where(notnull) pattern
    """
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        df[col] = df[col].astype(str)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════
# BACKGROUND TASK — runs the pipeline after /analyze returns
# ══════════════════════════════════════════════════════════════════════

def _run_pipeline_task(temp_path: str, run_id: str) -> None:
    """
    Synchronous function called by FastAPI's BackgroundTasks.

    pipeline.py already handles:
      - Writing "failed" status + error details to run_info.json
      - Registering the failed run in runs_index.json
      - Deleting the temp file if it lives under outputs/uploads/

    So this handler's job is only:
      1. Call run_pipeline() with the cached model
      2. Catch any re-raised exception and log the traceback
      3. Belt-and-braces temp-file cleanup in finally:
    """
    try:
        run_pipeline(
            log_path        = temp_path,
            run_id          = run_id,
            embedding_model = _CACHED_MODEL,
        )
    except Exception as exc:
        # pipeline.py has already written "failed" status to run_info.json.
        # We only log here — do NOT overwrite run_info.json again.
        logger.error(
            "Background pipeline task FAILED  run_id=%s  error=%s",
            run_id, exc,
            exc_info=True,
        )
    finally:
        # Belt-and-braces: remove the uploaded temp file if it still exists.
        # pipeline.py's except block should have done this already, but we
        # guard here in case the pipeline succeeded and left the file, or in
        # case an unusual exception path skipped pipeline's cleanup.
        p = Path(temp_path)
        if p.exists():
            try:
                p.unlink()
                logger.debug("Temp file cleaned up: %s", temp_path)
            except Exception as cleanup_exc:
                logger.warning("Could not delete temp file %s: %s", temp_path, cleanup_exc)


# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

# ── 1. Health check ───────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Confirm the API is reachable.
    Test first: http://localhost:8000/health  →  {"status": "ok"}
    """
    return {"status": "ok"}


# ── 2. Upload + start analysis ────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept a log file upload, validate it, write it to disk, and start
    the pipeline in the background.  Returns the run_id immediately so
    the dashboard can begin polling /status.

    Accepts: .log and .txt files up to 100 MB.
    """
    # ── Validate file type ────────────────────────────────────────────
    allowed_suffixes = {".log", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Only .log and .txt files are accepted (got '{suffix}').",
        )

    # ── Read and validate size ────────────────────────────────────────
    content = await file.read()
    max_bytes = 100 * 1024 * 1024  # 100 MB
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content):,} bytes).  Maximum is 100 MB.",
        )

    # ── Generate run_id + create run folder ───────────────────────────
    # run_id format: YYYY-MM-DD_HHMMSS_<stem>
    # validate_paths_for_run() creates the run folder and returns its Path.
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    stem      = Path(file.filename).stem
    run_id    = f"{timestamp}_{stem}"
    run_dir   = validate_paths_for_run(run_id)

    # ── Save the uploaded file into the run folder ────────────────────
    temp_path = run_dir / file.filename
    temp_path.write_bytes(content)

    # ── Write initial run_info.json so polling works from the first second
    # pipeline.py will overwrite this with richer detail as it progresses,
    # but we need something on disk before the background task starts.
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "run_id":     run_id,
                "log_filename": file.filename,
                "status":     "processing",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "stage_progress": "queued",
            },
            indent=2,
        )
    )

    # ── Enqueue the pipeline task ─────────────────────────────────────
    background_tasks.add_task(
        _run_pipeline_task,
        str(temp_path),
        run_id,
    )

    logger.info("Run enqueued: run_id=%s  file=%s  size=%d bytes",
                run_id, file.filename, len(content))

    return {"run_id": run_id, "status": "processing"}


# ── 3. Poll run status ────────────────────────────────────────────────

@app.get("/runs/{run_id}/status")
def get_run_status(run_id: str):
    """
    Return the current status of a run by reading run_info.json.

    The dashboard polls this every 3 seconds.

    status values written by pipeline.py:
      "processing"  — pipeline is running (stage_progress shows which stage)
      "complete"    — all stages finished; /results is now available
      "failed"      — an exception occurred; error_message contains details
    """
    info_path = RUNS_DIR / run_id / "run_info.json"
    if not info_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read run_info.json for run '{run_id}': {exc}",
        )


# ── 4. Fetch results ──────────────────────────────────────────────────

@app.get("/runs/{run_id}/results")
def get_run_results(run_id: str):
    """
    Return the full anomaly + incident JSON for a completed run.

    Delegates to pipeline.get_run() which reads:
      - run_info.json    → run metadata
      - stage4_anomaly.csv
      - incidents.csv    (the filename Stage 5 actually writes)
      - manifest.json

    Returns HTTP 425 if the run is still processing or failed.
    Returns HTTP 404 if the run does not exist.
    """
    # Check run exists and is complete before doing any file I/O
    info_path = RUNS_DIR / run_id / "run_info.json"
    if not info_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    status = info.get("status", "unknown")

    if status != "complete":
        raise HTTPException(
            status_code=425,
            detail=(
                f"Run is '{status}' — results are not yet available.  "
                f"Poll /runs/{run_id}/status until status == 'complete'."
            ),
        )

    # Delegate to pipeline.get_run() — it knows all the correct filenames
    result = get_run(run_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run folder for '{run_id}' missing from disk.",
        )

    # result keys: run_info, anomalies, incidents, stage3, manifest
    # Serialise DataFrames that pipeline.get_run() returns as list-of-dicts
    # (get_run already calls .to_dict(orient="records") internally, so
    # result["anomalies"] / result["incidents"] are already plain lists).

    # ── Enrich anomaly rows with template fields from stage3_output.csv ──
    # stage4_anomaly.csv uses semantic_cluster_id as event_id but does NOT
    # carry template_id or event_template — those live in stage3_output.csv.
    # Build a lookup keyed on semantic_cluster_id → {template_id, event_template}
    # and merge it into every anomaly row before sending to the frontend.
    # Without this, AnomaliesPage always shows "No template available" and
    # handleFeedback can never capture a valid template_id for the safelist.
    try:
        import csv as _csv
        _stage3_path = RUNS_DIR / run_id / "stage3_output.csv"
        _template_map: dict = {}
        if _stage3_path.exists():
            with open(_stage3_path, newline="", encoding="utf-8") as _fh:
                for _r in _csv.DictReader(_fh):
                    _sid = (_r.get("semantic_cluster_id") or "").strip()
                    if _sid and _sid not in _template_map:
                        _template_map[_sid] = {
                            "template_id":    (_r.get("template_id")    or "").strip(),
                            "event_template": (_r.get("event_template") or "").strip(),
                        }
            logger.info(
                "Stage3 template map built: %d entries for run %s",
                len(_template_map), run_id,
            )
        else:
            logger.warning("stage3_output.csv not found for run %s — template fields will be empty", run_id)

        def _enrich_anomalies(anomaly_rows: list) -> list:
            """Merge template_id + event_template from stage3 into each anomaly row."""
            for row in anomaly_rows:
                eid = row.get("event_id", "")
                stage3 = _template_map.get(eid, {})
                # Only set if not already present (don't overwrite if pipeline ever adds them)
                if not row.get("template_id"):
                    row["template_id"]    = stage3.get("template_id",    "")
                if not row.get("event_template"):
                    row["event_template"] = stage3.get("event_template", "")
            return anomaly_rows
    except Exception as _enrich_exc:
        logger.warning("Stage3 enrichment failed (non-fatal): %s", _enrich_exc)
        def _enrich_anomalies(anomaly_rows: list) -> list:
            return anomaly_rows  # pass-through — don't break the whole response

    def sanitise(records: list) -> list:
        """Convert all values to JSON-safe Python natives."""
        import math, json as _json
        import numpy as np
        sanitised = []
        for row in records:
            clean = {}
            for k, v in row.items():
                # Convert numpy types to Python natives first
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = None if not np.isfinite(v) else float(v)
                elif isinstance(v, (np.bool_,)):
                    v = bool(v)
                elif isinstance(v, float) and not math.isfinite(v):
                    v = None
                # Parse services_affected — handles list, JSON array, pipe string, or plain string
                if k == "services_affected":
                    if isinstance(v, list):
                        v = [str(s).strip() for s in v if s is not None and str(s).strip()]
                    elif isinstance(v, str) and v.strip():
                        try:
                            parsed = _json.loads(v)
                            v = parsed if isinstance(parsed, list) else [parsed]
                        except Exception:
                            if "|" in v:
                                v = [s.strip() for s in v.split("|") if s.strip()]
                            else:
                                v = [v.strip()]
                    else:
                        v = []
                clean[k] = v
            sanitised.append(clean)
        return sanitised

    anomalies       = sanitise(_enrich_anomalies(result.get("anomalies", [])))
    incidents       = sanitise(result.get("incidents",       []))
    cluster_summary = sanitise(result.get("cluster_summary", []))

    return JSONResponse(
        content={
            "run_id":         run_id,
            "filename":       info.get("log_filename", ""),
            "anomaly_count":  len(anomalies),
            "incident_count": len(incidents),
            "anomalies":      anomalies,
            "incidents":      incidents,
            "cluster_summary": cluster_summary,
            "run_info":       result.get("run_info", {}),
        }
    )


# ── 5. List all runs ──────────────────────────────────────────────────

@app.get("/runs")
def get_all_runs():
    """
    Return the master runs index — used to populate the Previous Runs panel.

    Delegates to pipeline.list_runs() which reads runs_index.json.
    Returns an empty list if no runs exist yet.

    NOTE: Only "complete" and "failed" runs appear here.
    pipeline.py only calls _register_run() after status is finalised,
    so in-progress runs never pollute this list.
    """
    return list_runs()


# ── 6. Delete a run ───────────────────────────────────────────────────

@app.delete("/runs/{run_id}")
def delete_run(run_id: str):
    """
    Delete a run's folder and remove it from runs_index.json.

    Delegates to pipeline.delete_run(run_id) which:
      - Calls shutil.rmtree on the run folder
      - Rewrites runs_index.json without this entry
      - Handles the case where the folder was already manually deleted

    Returns HTTP 404 if the run_id is not found anywhere.
    """
    deleted = pipeline_delete_run(run_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in folder or index.",
        )
    logger.info("Run deleted: run_id=%s", run_id)
    return {"deleted": run_id}

# ══════════════════════════════════════════════════════════════════════
# GROUND TRUTH — Human-in-the-loop feedback endpoints
# ══════════════════════════════════════════════════════════════════════
#
# These endpoints wire the True Positive / False Positive buttons in the
# frontend to the SQLite ground truth store (backend/ground_truth.py).
#
# Flow:
#   User clicks "False Positive" on a cluster card
#     → POST /feedback  { cluster_ref, label, template_id, ... }
#     → GroundTruthStore.upsert() saves to ground_truth.db
#     → Next pipeline run reads get_false_positive_template_ids()
#     → Stage 3 classify_singletons() receives known_normal_tids
#     → That cluster is never re-flagged as true_anomaly again
#
# Endpoints
# ─────────────────────────────────────────────────────────────────────
#   POST   /feedback                 — submit / update a label
#   GET    /feedback/{cluster_ref}   — get label for one cluster
#   DELETE /feedback/{cluster_ref}   — remove a label
#   GET    /feedback                 — list all labels
#   GET    /feedback/stats           — TP/FP counts summary
#   GET    /feedback/export          — download as CSV
# ══════════════════════════════════════════════════════════════════════

from pydantic import BaseModel
from fastapi.responses import FileResponse
import tempfile

try:
    from backend.ground_truth import GroundTruthStore
except ImportError:
    from ground_truth import GroundTruthStore

# Module-level singleton — one DB connection pool for all requests
_GT_STORE: GroundTruthStore = None


def _get_gt_store() -> GroundTruthStore:
    """Lazily initialise the ground truth store on first use."""
    global _GT_STORE
    if _GT_STORE is None:
        try:
            from config import GROUND_TRUTH_DB_PATH
            _GT_STORE = GroundTruthStore(db_path=GROUND_TRUTH_DB_PATH)
        except Exception:
            _GT_STORE = GroundTruthStore()
    return _GT_STORE


class FeedbackRequest(BaseModel):
    cluster_ref:   str
    label:         str                    # "true_positive" | "false_positive"
    template_id:   Optional[str]  = None  # required — must be non-empty string
    run_id:        Optional[str]  = None
    service:       Optional[str]  = None
    log_template:  Optional[str]  = None
    severity:      Optional[str]  = None
    anomaly_score: Optional[float] = None
    notes:         Optional[str]  = None


# ── 7. Submit / update feedback ───────────────────────────────────────

@app.post("/feedback")
def submit_feedback(body: FeedbackRequest):
    """
    Save a True Positive or False Positive label for a cluster.

    If the cluster_ref already has a label, it is overwritten — so users
    can change their mind and resubmit.  labelled_at is always refreshed.

    False Positive labels flow into Stage 3 automatically on the next
    pipeline run via get_false_positive_template_ids() → known_normal_tids.

    Returns HTTP 400 if template_id is missing or blank — a record without
    a template_id cannot participate in the known_normal_tids safelist and
    would be silently useless.
    """
    if body.label not in ("true_positive", "false_positive"):
        raise HTTPException(
            status_code=422,
            detail="label must be 'true_positive' or 'false_positive'",
        )

    # ── Enforce template_id — the safelist key ────────────────────────
    # An empty / missing template_id means Stage 3 can never suppress this
    # cluster on reruns.  Reject early rather than save a broken record.
    template_id = (body.template_id or "").strip() or None
    if template_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "template_id is required and must not be empty. "
                "Check that the anomaly card row contains a 'template_id' field "
                "from the pipeline output before submitting feedback."
            ),
        )

    store = _get_gt_store()
    record = store.upsert(
        cluster_ref   = body.cluster_ref,
        label         = body.label,
        template_id   = template_id,                              # validated — never None
        run_id        = (body.run_id        or "").strip() or None,
        service       = (body.service       or "").strip() or None,
        log_template  = (body.log_template  or "").strip() or None,
        severity      = (body.severity      or "").strip() or None,
        anomaly_score = body.anomaly_score,
        notes         = (body.notes         or "").strip() or None,
    )
    logger.info(
        "Feedback saved: cluster_ref=%s  label=%s  template_id=%s  run_id=%s",
        body.cluster_ref, body.label, template_id, body.run_id,
    )
    return record


# ── 8. Get feedback for one cluster ──────────────────────────────────

@app.get("/feedback/{cluster_ref}")
def get_feedback(cluster_ref: str):
    """
    Return the current label for a single cluster_ref.
    Returns HTTP 404 if no feedback has been submitted for this cluster.
    """
    record = _get_gt_store().get(cluster_ref)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No feedback found for cluster_ref '{cluster_ref}'.",
        )
    return record


# ── 9. Delete feedback for one cluster ───────────────────────────────

@app.delete("/feedback/{cluster_ref}")
def delete_feedback(cluster_ref: str):
    """
    Remove the label for a cluster_ref.
    Returns HTTP 404 if no record exists.
    """
    deleted = _get_gt_store().delete(cluster_ref)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No feedback found for cluster_ref '{cluster_ref}'.",
        )
    return {"deleted": cluster_ref}


# ── 10. List all feedback ─────────────────────────────────────────────

@app.get("/feedback")
def list_feedback(label: str = None):
    """
    Return all feedback records, newest first.

    Optional query parameter:
      ?label=false_positive   — return only false positives
      ?label=true_positive    — return only true positives
    """
    store = _get_gt_store()
    if label:
        if label not in ("true_positive", "false_positive"):
            raise HTTPException(
                status_code=422,
                detail="label filter must be 'true_positive' or 'false_positive'",
            )
        return store.list_by_label(label)
    return store.list_all()


# ── 11. Feedback stats ────────────────────────────────────────────────

@app.get("/feedback/stats")
def feedback_stats():
    """
    Return summary counts of all feedback labels.

    Response: { "total": N, "true_positive": N, "false_positive": N }
    """
    return _get_gt_store().get_stats()


# ── 12. Export ground truth as CSV ────────────────────────────────────

@app.get("/feedback/export")
def export_feedback():
    """
    Download the full ground truth table as a CSV file.
    Useful for offline review, model evaluation, or sharing with the team.
    """
    store = _get_gt_store()
    tmp = Path(tempfile.mktemp(suffix=".csv"))
    store.export_csv(tmp)
    return FileResponse(
        path        = str(tmp),
        filename    = "ground_truth.csv",
        media_type  = "text/csv",
    )