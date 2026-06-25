# backend/ground_truth.py
#
# ══════════════════════════════════════════════════════════════════════
# GROUND TRUTH — Human-in-the-loop feedback store
# ══════════════════════════════════════════════════════════════════════
#
# Stores user feedback (True Positive / False Positive) per cluster ref
# in a local SQLite database.  The store feeds back into Stage 3 via the
# known_normal_tids safelist — clusters marked False Positive are never
# re-flagged as true_anomaly in subsequent pipeline runs.
#
# Schema (ground_truth table)
# ─────────────────────────────────────────────────────────────────────
#   cluster_ref   TEXT  PRIMARY KEY   — semantic_cluster_id (e.g. SC8DE1D…)
#   template_id   TEXT                — template_id from stage3 output
#   label         TEXT                — "true_positive" | "false_positive"
#   run_id        TEXT                — which run the feedback came from
#   service       TEXT                — service name (from anomaly card)
#   log_template  TEXT                — human-readable template text
#   severity      TEXT                — dominant severity at time of labelling
#   anomaly_score REAL                — score at time of labelling
#   labelled_at   TEXT                — ISO-8601 UTC timestamp
#   notes         TEXT                — optional free-text from user
#
# Usage
# ─────────────────────────────────────────────────────────────────────
#   from backend.ground_truth import GroundTruthStore
#   store = GroundTruthStore()                    # opens / creates DB
#   store.upsert(cluster_ref, template_id, ...)   # save feedback
#   store.get_false_positive_template_ids()       # feed into stage3
#   store.export_csv(path)                        # export for review
#
# PLACEMENT: backend/ground_truth.py
# ══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import csv
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("ground_truth")

# ── Default DB path — sits next to runs_index.json in outputs/ ────────
# Overridable by passing db_path explicitly to GroundTruthStore().
_DEFAULT_DB_PATH: Optional[Path] = None  # resolved lazily from config


def _resolve_default_db_path() -> Path:
    """Resolve the default DB path from config.py (lazy import)."""
    global _DEFAULT_DB_PATH
    if _DEFAULT_DB_PATH is None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from config import OUTPUT_DIR
            _DEFAULT_DB_PATH = OUTPUT_DIR / "ground_truth.db"
        except Exception:
            # Fallback: same directory as this file
            _DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "outputs" / "ground_truth.db"
    return _DEFAULT_DB_PATH


# ══════════════════════════════════════════════════════════════════════
# GROUND TRUTH STORE
# ══════════════════════════════════════════════════════════════════════

class GroundTruthStore:
    """
    Thread-safe SQLite-backed store for cluster feedback labels.

    One instance is shared across all API requests (module-level singleton
    in api.py).  SQLite's WAL mode handles concurrent reads safely.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS ground_truth (
            cluster_ref   TEXT PRIMARY KEY,
            template_id   TEXT,
            label         TEXT NOT NULL CHECK(label IN ('true_positive', 'false_positive')),
            run_id        TEXT,
            service       TEXT,
            log_template  TEXT,
            severity      TEXT,
            anomaly_score REAL,
            labelled_at   TEXT NOT NULL,
            notes         TEXT
        );
    """

    _CREATE_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_gt_label
        ON ground_truth (label);
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = Path(db_path) if db_path else _resolve_default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        logger.info("GroundTruthStore initialised at %s", self._db_path)

    # ── Internal helpers ──────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL mode: readers don't block writers; safe for FastAPI concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(self._CREATE_TABLE)
                conn.execute(self._CREATE_INDEX)
                conn.commit()

    # ── Public API ────────────────────────────────────────────────────

    def upsert(
        self,
        cluster_ref:   str,
        label:         str,                      # "true_positive" | "false_positive"
        template_id:   Optional[str]  = None,
        run_id:        Optional[str]  = None,
        service:       Optional[str]  = None,
        log_template:  Optional[str]  = None,
        severity:      Optional[str]  = None,
        anomaly_score: Optional[float] = None,
        notes:         Optional[str]  = None,
    ) -> dict:
        """
        Insert or update a label for a cluster_ref.

        If the same cluster_ref is submitted again (e.g. user changes their
        mind from false_positive to true_positive) the row is updated in place.
        labelled_at is always refreshed to the current UTC time on update.
        """
        if label not in ("true_positive", "false_positive"):
            raise ValueError(f"label must be 'true_positive' or 'false_positive', got {label!r}")

        # ── Normalise and validate template_id ───────────────────────────
        # template_id must be a non-empty string.  Saving NULL here means
        # get_false_positive_template_ids() will silently skip this record
        # and the safelist will never suppress the cluster on reruns.
        if not template_id or not str(template_id).strip():
            raise ValueError(
                f"template_id must be a non-empty string for cluster_ref={cluster_ref!r}. "
                "Ensure the pipeline output contains a 'template_id' column and that it "
                "is forwarded through the API payload before calling upsert()."
            )
        template_id = str(template_id).strip()

        labelled_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO ground_truth
                        (cluster_ref, template_id, label, run_id, service,
                         log_template, severity, anomaly_score, labelled_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cluster_ref) DO UPDATE SET
                        template_id   = excluded.template_id,
                        label         = excluded.label,
                        run_id        = excluded.run_id,
                        service       = excluded.service,
                        log_template  = excluded.log_template,
                        severity      = excluded.severity,
                        anomaly_score = excluded.anomaly_score,
                        labelled_at   = excluded.labelled_at,
                        notes         = excluded.notes
                    """,
                    (
                        cluster_ref, template_id, label, run_id, service,
                        log_template, severity, anomaly_score, labelled_at, notes,
                    ),
                )
                conn.commit()

        logger.info(
            "GT upsert: cluster_ref=%s  label=%s  run_id=%s",
            cluster_ref, label, run_id,
        )
        return self.get(cluster_ref)

    def get(self, cluster_ref: str) -> Optional[dict]:
        """Return a single record by cluster_ref, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ground_truth WHERE cluster_ref = ?",
                (cluster_ref,),
            ).fetchone()
        return dict(row) if row else None

    def delete(self, cluster_ref: str) -> bool:
        """Remove a label. Returns True if a row was deleted."""
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM ground_truth WHERE cluster_ref = ?",
                    (cluster_ref,),
                )
                conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("GT delete: cluster_ref=%s", cluster_ref)
        return deleted

    def list_all(self) -> List[dict]:
        """Return every record, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ground_truth ORDER BY labelled_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_by_label(self, label: str) -> List[dict]:
        """Return all records with a specific label."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ground_truth WHERE label = ? ORDER BY labelled_at DESC",
                (label,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_false_positive_template_ids(self) -> List[str]:
        """
        Return the list of template_ids marked false_positive.

        This is the direct input to Stage 3's known_normal_tids safelist:

            from backend.ground_truth import GroundTruthStore
            store = GroundTruthStore()
            known_normal = store.get_false_positive_template_ids()
            df = classify_singletons(df, known_normal_tids=known_normal)

        Only returns rows where template_id is not NULL.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT template_id
                FROM ground_truth
                WHERE label = 'false_positive'
                  AND template_id IS NOT NULL
                  AND template_id != ''
                """
            ).fetchall()
        tids = [r["template_id"] for r in rows]
        logger.debug(
            "get_false_positive_template_ids: returning %d template_ids", len(tids)
        )
        return tids

    def get_stats(self) -> dict:
        """Return summary counts — used by the /feedback/stats endpoint."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM ground_truth"
            ).fetchone()[0]
            tp = conn.execute(
                "SELECT COUNT(*) FROM ground_truth WHERE label = 'true_positive'"
            ).fetchone()[0]
            fp = conn.execute(
                "SELECT COUNT(*) FROM ground_truth WHERE label = 'false_positive'"
            ).fetchone()[0]
        return {
            "total":          total,
            "true_positive":  tp,
            "false_positive": fp,
        }

    def export_csv(self, path: Path) -> Path:
        """
        Write the full ground truth table to a CSV file.
        Returns the path written.
        """
        records = self.list_all()
        if not records:
            logger.warning("export_csv: no records to export")
            path.write_text("")
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        logger.info("GT exported to %s (%d rows)", path, len(records))
        return path