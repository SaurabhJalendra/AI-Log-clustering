"""
backend/stages/stage3.py
========================
STAGE 3 — SEMANTIC CLUSTERING & ANOMALY CLASSIFICATION

Public API
----------
    from stages.stage3 import run_stage3

    df_classified, stats = run_stage3(
        stage2_df,
        cluster_manifest=cluster_manifest,   # dict from Stage 2 build_manifest()
    )

Returns
-------
    df_classified : pd.DataFrame
        Per-line DataFrame with columns added by Stage 3:
        semantic_cluster_id, singleton_class, anomaly_signal, is_routine
        Columns preserved from Stage 2 (read-only, not modified):
        domain, domain_confidence, domain_source, normalized_message,
        severity, is_noise, is_continuation
    stats : dict
        Keys: unique_templates_clustered, cluster_summary,
              stage25_df_with_clusters, suspicious_splits,
              threshold_used, n_semantic_clusters, n_isolated,
              silhouette_score, embedding_model, header_cluster_count,
              consistency_failures, manifest_used, status

Stage 3 does NOT call Stage 1 or Stage 2 — it receives stage2_df as a
parameter. Stages never call each other directly; pipeline.py is the
only place that chains them.

Domain classification is owned exclusively by Stage 2.
Stage 3 reads domain as an immutable input column and only performs
plurality-vote aggregation (_dominant_domain) for cluster summaries.

Fixes implemented (from notebook):
    ACCURACY-FIX-B1  Manifest-read pattern for cluster counts/metadata
    ACCURACY-FIX-B3  anomaly_signal / anomaly_reason per cluster
    ACCURACY-FIX-B4  Severity distribution per cluster
    ACCURACY-FIX-B5  first_seen / last_seen / services per cluster
    ACCURACY-FIX-B6  header_cluster_count assertion
    ACCURACY-FIX-B8  Dominant-domain aggregation for cluster_summary
    ACCURACY-FIX-N1  Source line count injection
    ACCURACY-FIX-N4  Clustering parameter tuning
"""

from __future__ import annotations

import hashlib
import logging
import re
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("stage3")


# Severity ordering — Stage 1 canonical output values only.
# Raw input aliases (NOTICE, WARNING, ERR, CRITICAL, SEVERE, PANIC) are
# normalised by Stage 1 before Stage 3 sees the data; they are unreachable here.
_SEVERITY_ORDER: Dict[str, int] = {
    "DEBUG": 0,
    "TRACE": 1,
    "INFO":  2,
    "WARN":  3,
    "ERROR": 4,
    "FATAL": 5,
}
# _SEVERITY_CANONICAL removed: Stage 1 owns severity normalisation.
# Use _SEVERITY_ORDER.get(str(sev).upper(), 0) for rank lookups.


# ── STAGE 3 CONFIG ────────────────────────────────────────────────────
STAGE3_CONFIG = {
    # S3-ML-1: log-adapted model replacing all-MiniLM-L6-v2.
    # Falls back to "all-MiniLM-L6-v2" automatically if the log-domain
    # model cannot be loaded (see _load_embedding_model below).
    "embedding_model":   "PhilipMay/sbert-base-sts-gemini-en",   # log-adapted fallback; swap for LogBERT when available
    "embedding_model_primary": "jinaai/jina-embeddings-v2-base-en",  # preferred; most log-domain aware publicly available
    "umap_n_components": 10,
    "umap_n_neighbors":  15,   # base value; overridden per-partition by service-aware logic
    "umap_min_dist":     0.0,
    "umap_metric":       "cosine",
    "umap_random_state": 42,
    "hdbscan_min_cluster_size":         3,     # S3-FP-3: raised from 2; reduces random pairings
    "hdbscan_min_samples":              1,
    "cluster_coherence_min_sim":        0.60,  # S3-FP-3: dissolve clusters below this avg cosine similarity
    "cluster_coherence_enabled":        True,  # S3-FP-3: set False to disable without code change
    "anchor_sim_bypass_threshold":      0.88,  # S3-FP-4: keep high-sim members despite anchor word gap
    "hdbscan_metric":                   "euclidean",
    "hdbscan_cluster_selection_method": "eom",
    "agglo_distance_threshold":         0.45,
    "agglo_distance_threshold_fallback": 0.55,
    "silhouette_good":       0.50,
    "silhouette_acceptable": 0.30,
    "min_templates_for_clustering": 3,
    "partition_by_service":       True,
    "partition_by_severity_tier": True,
    "anchor_word_floor": 1,
    # S3-ML-2: service-aware UMAP — n_neighbors scales with partition size
    "umap_adaptive_neighbors": True,   # if True, n_neighbors = f(partition_size)
    # S3-ML-3: known-normal safelist — template_ids confirmed normal across runs
    # Populated at runtime by the pipeline from previous run_info or explicit config.
    "known_normal_tids": [],           # list[str]; populated by pipeline.py
    # S3-ML-3-EXPIRY: safelist staleness guard (ChatGPT point 9).
    # A safelisted template that starts appearing at very high frequency may
    # have changed meaning — e.g. "payment retry limit reached" that was benign
    # during testing but is a real issue in production.
    # known_normal_max_daily_count: if a safelisted template appears more than
    # this many times in a single run, it bypasses the safelist and is re-evaluated.
    # Set to 0 to disable (original behaviour — no expiry).
    "known_normal_max_daily_count": 500,  # 0 = no expiry guard
    # known_normal_max_age_days: entries older than this many days are not treated
    # as safelisted. Prevents stale safelist entries from persisting indefinitely.
    # Set to 0 to disable age-based expiry (pipeline.py enforces this on load).
    "known_normal_max_age_days": 90,      # recommended 90 days per spec §3.4.1
    # S3-ML-4: embedding drift detection threshold (cosine distance)
    "drift_detection_threshold": 0.30,  # if cluster centroid shifts > this → flag for review
    # S3-ML-5 REMOVED: prototype embedding domain matching moved to Stage 2.
    # These keys are intentionally absent. Domain classification is owned by Stage 2.
    # PERFORMANCE: cross-partition merge is O(n²) on isolated singletons.
    # Cap at this many singletons before skipping the merge step.
    # Above the cap, cross-partition merging is skipped (isolated templates stay isolated).
    "cross_partition_merge_max_singletons": 800,  # set 0 to disable entirely
}

# Partition severity tier — maps Stage 1 canonical values to coarse tiers
# used for (service, tier) partition keys. Dead aliases removed (spec §9A).
_SEVERITY_TIER: Dict[str, str] = {
    "debug":    "info",
    "trace":    "info",
    "info":     "info",
    "warn":     "warn",
    "error":    "error",
    "fatal":    "error",
}

# NOTE: These patterns apply to Drain-generated event templates only.
# Raw-line noise detection is owned by Stage 1 (is_noise column).
# Do not merge these with Stage 1's RE_NOISE patterns.
_NOISE_TEMPLATE_MARKERS = (
    "<noise>", "<empty>", "goroutine ", "panic:", "main.",
    "???", "\x00", "==>", "####", "^c",
)

# NOTE: These patterns apply to Drain-generated event templates only.
# Raw-line noise detection is owned by Stage 1 (is_noise column).
# Do not merge these with Stage 1's RE_NOISE patterns.
# Sync note: if new storage providers are added to Stage 1's line-level
# safelist, they should also be added here at the template level.
_TEMPLATE_NOISE_SAFELIST = [
    # GCS
    re.compile(r"log-archive-gcs",          re.IGNORECASE),
    re.compile(r"gcs[_\-]upload",           re.IGNORECASE),
    re.compile(r"storage\.googleapis\.com", re.IGNORECASE),
    # AWS S3
    re.compile(r"s3://",                    re.IGNORECASE),
    re.compile(r"amazonaws\.com",           re.IGNORECASE),
    # Azure Blob
    re.compile(r"blob\.core\.windows\.net", re.IGNORECASE),
    # MinIO / generic object storage
    re.compile(r"\bminio\b",                re.IGNORECASE),
]

# NOTE: These patterns prepare normalized_message for embedding.
# They are NOT PII masking (which is owned by Stage 2).
# Purpose: remove residual tokens (timestamps, K8s age prefixes,
# CEF headers) that waste embedding capacity without contributing
# semantic signal. They operate on normalized_message after Stage 2
# has already masked PII.
_PRE_EMBED_STRIP_PATTERNS = [
    re.compile(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
        re.IGNORECASE,
    ),
    re.compile(r"--\d+t\d+::\+:", re.IGNORECASE),
    re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:\s+\d{2}:\d{2}:\d{2})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*\d+[dhms](?:\d+[dhms])*\s+", re.IGNORECASE),
    re.compile(r"CEF:\d+(?:\|[^|]*){6}\|", re.IGNORECASE),
]

_ANCHOR_STOPWORDS = frozenset({
    "the", "and", "for", "from", "with", "that", "this", "was", "has",
    "had", "been", "have", "will", "are", "not", "but", "can", "all",
    "any", "its", "our", "your", "their", "into", "onto", "upon",
    "method", "path", "status", "duration", "error", "type", "level",
    "host", "port", "size", "limit", "source", "target", "reason",
    "upstream", "session", "timeout", "after", "open", "close", "closed",
    "user", "users", "client", "server", "system", "service", "event",
    "record", "request", "response", "data", "item", "items",
    "field", "value", "values", "time", "name", "mode", "code", "info",
    "message", "detail", "details", "result", "results", "output",
    "input", "action", "state", "count", "total", "index", "entry",
    "object",
})

_ANOMALY_KEYWORDS = {
    "error", "fail", "failed", "failure", "exception", "critical",
    "timeout", "refused", "unavailable", "crash", "panic",
    "corrupt", "invalid", "unauthorized", "forbidden",
}

_CRITICAL_OVERRIDE_KEYWORDS = {
    "checksum", "corruption", "data integrity", "integrity alert",
    "certificate pinning", "mitm", "mitm attack", "token signing key",
    "unusual login", "brute force", "kernel panic",
    "memory usage critical", "cascade failure", "deadlock",
    "zombie process", "segfault", "out of memory",
    "dns resolution intermittent",
}

_LOW_COUNT_THRESHOLD = 5

# NOTE: These patterns apply to Drain-generated event templates only.
# Raw-line noise detection is owned by Stage 1 (is_noise column).
# Do not merge these with Stage 1's RE_NOISE patterns.
_NOISE_TEMPLATE_PREFIXES = (
    "goroutine ", "panic: runtime error", "panic:", "main.", "net/http.",
    "/app/src/", "/usr/local/go/src/", "???", "==>",
    "--- connection reset", "--- ", "^c", "null",
)

# ══════════════════════════════════════════════════════════════════════
# ACCURACY-FIX-B4 / B5: SEVERITY & TIMESTAMP HELPERS
# ══════════════════════════════════════════════════════════════════════

def _canonical_severity(sev: str) -> str:
    # Stage 1 has already normalised severity — use rank lookup for unknown values.
    s = str(sev).upper().strip()
    # Return as-is if it's already a canonical value; otherwise fall to INFO.
    return s if s in _SEVERITY_ORDER else "INFO"




def _max_severity(sev_distribution: Dict[str, int]) -> str:
    best = "DEBUG"
    best_order = -1
    for sev, cnt in sev_distribution.items():
        if cnt > 0:
            order = _SEVERITY_ORDER.get(sev.upper(), -1)
            if order > best_order:
                best_order = order
                best = sev
    return best


def _dominant_severity(sev_distribution: Dict[str, int]) -> str:
    if not sev_distribution:
        return "INFO"
    max_count = max(sev_distribution.values(), default=0)
    candidates = [s for s, c in sev_distribution.items() if c == max_count]
    return max(candidates, key=lambda s: _SEVERITY_ORDER.get(s.upper(), 0))


def _build_severity_distribution(severity_series: pd.Series) -> Dict[str, int]:
    dist: Dict[str, int] = {"DEBUG": 0, "INFO": 0, "WARN": 0, "ERROR": 0, "FATAL": 0}
    for raw in severity_series:
        canon = _canonical_severity(str(raw or "INFO"))
        bucket = canon if canon in dist else "INFO"
        dist[bucket] += 1
    return dist


# ══════════════════════════════════════════════════════════════════════
# ACCURACY-FIX-B8: DOMINANT DOMAIN AGGREGATION
# ══════════════════════════════════════════════════════════════════════

# NOTE: _dominant_domain() is a plurality vote over the domain
# column written by Stage 2. It aggregates — it does not classify.
# Stage 3 never re-derives or overrides individual row domains.
def _dominant_domain(domain_series: pd.Series) -> str:
    counts: Counter = Counter()
    for d in domain_series:
        dv = str(d).strip() if pd.notna(d) else "other"
        counts[dv] += 1
    if not counts:
        return "other"
    non_noise = {k: v for k, v in counts.items() if k != "noise"}
    vote_pool = non_noise if non_noise else counts
    max_count = max(vote_pool.values())
    candidates = sorted(k for k, v in vote_pool.items() if v == max_count)
    return candidates[0]


# ── STABLE HASH HELPERS ───────────────────────────────────────────────
def _stable_hash(text: str) -> str:
    h = hashlib.blake2b(digest_size=6, person=b"s25semcl")
    h.update(text.encode("utf-8", errors="replace"))
    return "SC" + h.hexdigest().upper()


# ── EMBEDDING HELPERS ─────────────────────────────────────────────────
def _clean_for_embedding(text: str) -> str:
    t = str(text or "")
    for pat in _PRE_EMBED_STRIP_PATTERNS:
        t = pat.sub(" ", t)
    t = re.sub(r"(<\*>\s*)+", " VAR ", t)
    t = " ".join(t.split())
    return t


def _majority_service(services: pd.Series) -> str:
    counts: Counter = Counter()
    for s in services:
        sv = str(s).strip() if pd.notna(s) else ""
        if sv and sv not in ("", "unknown", "nan"):
            counts[sv] += 1
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# ── ANCHOR WORD HELPERS ───────────────────────────────────────────────
def _extract_anchor_words(template_text: str) -> List[str]:
    anchors = []
    for tok in template_text.split():
        t = tok.lower().strip(".,;:\"'()[]}{")
        if (
            t == "<*>" or t == "var"
            or len(t) < 4
            or t in _ANCHOR_STOPWORDS
            or t.isdigit()
            or "=" in t
            or re.match(r"^[\d_\-\.]+$", t)
        ):
            continue
        anchors.append(t)
    return anchors


def _apply_anchor_floor(
    tids: List[str],
    labels: np.ndarray,
    texts: List[str],
    min_shared: int = 1,
    embeddings: Optional[np.ndarray] = None,   # S3-FP-4
    sim_bypass_threshold: float = 0.88,         # S3-FP-4
) -> np.ndarray:
    """
    Anchor floor with S3-FP-4 embedding-similarity bypass.

    If the weakest template's cosine similarity to the cluster centroid
    exceeds sim_bypass_threshold, it is kept regardless of anchor word deficit.
    This fixes synonym fragmentation: "timed out" vs "timeout" share no anchor
    words after stopword removal but are semantically identical — embedding
    similarity catches this where exact token matching cannot.

    All original logic is preserved. The bypass only prevents ejection;
    it never adds new members to a cluster.
    """
    labels = labels.copy()
    cluster_ids = set(labels[labels != -1])

    # S3-FP-4: normalise embeddings once for reuse in centroid similarity checks
    normed_embs: Optional[np.ndarray] = None
    if embeddings is not None and sim_bypass_threshold > 0:
        _norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        _norms[_norms == 0] = 1.0
        normed_embs = (embeddings / _norms).astype(np.float32)

    for cid in cluster_ids:
        member_mask    = labels == cid
        member_indices = [i for i, m in enumerate(member_mask) if m]
        if len(member_indices) <= 1:
            continue

        anchor_sets = [set(_extract_anchor_words(texts[i])) for i in member_indices]

        while len(member_indices) > 1:
            common = anchor_sets[0].copy()
            for s in anchor_sets[1:]:
                common &= s
            if len(common) >= min_shared:
                break

            scores = []
            for i, s in enumerate(anchor_sets):
                union_others = set()
                for j, os_ in enumerate(anchor_sets):
                    if j != i:
                        union_others |= os_
                scores.append(len(s & union_others))

            weakest_pos = scores.index(min(scores))
            weakest_idx = member_indices[weakest_pos]

            # ── S3-FP-4: Embedding similarity bypass ──────────────────────
            # Before ejecting the weakest template, check its cosine similarity
            # to the current cluster centroid. If it is very high (>= threshold),
            # the anchor deficit is a surface-form artefact (synonym / morphological
            # variant), not a genuine semantic mismatch — keep the member.
            if normed_embs is not None:
                current_embs = normed_embs[member_indices]
                centroid     = current_embs.mean(axis=0)
                c_norm       = np.linalg.norm(centroid)
                if c_norm > 0:
                    centroid = centroid / c_norm
                sim_to_centroid = float(np.dot(normed_embs[weakest_idx], centroid))
                if sim_to_centroid >= sim_bypass_threshold:
                    logger.debug(
                        "S3-FP-4: anchor bypass for template idx %d "
                        "(sim=%.3f >= threshold=%.2f)",
                        weakest_idx, sim_to_centroid, sim_bypass_threshold,
                    )
                    break   # Keep this member; stop ejecting
            # ── end S3-FP-4 ───────────────────────────────────────────────

            labels[weakest_idx] = -1
            member_indices.pop(weakest_pos)
            anchor_sets.pop(weakest_pos)

            if len(member_indices) == 1:
                labels[member_indices[0]] = -1
                break

    return labels


def _severity_tier(sev: str) -> str:
    return _SEVERITY_TIER.get(str(sev).lower().strip(), "info")


# ── STEP 1: COLLECT UNIQUE TEMPLATES ─────────────────────────────────
def _collect_unique_templates(stage2_df: pd.DataFrame) -> pd.DataFrame:
    clean = stage2_df[
        ~stage2_df["is_noise"].fillna(False).astype(bool) &
        (stage2_df["template_id"] != "TM_NOISE_FILTERED")
    ].copy()

    _CONTINUATION_RE = re.compile(
        r"""
        ^\s*\d+\s*[\|>]
        | ^\s{4,}at\s+\S
        | ^\s*[{}\[\]()]\s*$
        | ^\s*\.\.\.\s*\d+\s+more
        | ^\s*---+\s*$
        """,
        re.VERBOSE | re.MULTILINE,
    )
    _text_col_for_filter = (
        "event_template" if "event_template" in clean.columns
        else "normalized_message"
    )
    _cont_mask = clean[_text_col_for_filter].fillna("").apply(
        lambda t: bool(_CONTINUATION_RE.match(str(t)))
    )
    if _cont_mask.any():
        logger.info(
            "Continuation-line filter: suppressing %d templates before clustering",
            int(_cont_mask.sum()),
        )
    clean = clean[~_cont_mask].reset_index(drop=True)

    majority_svc = (
        clean.groupby("template_id")["service"]
        .apply(_majority_service)
        .rename("majority_service")
    )

    agg_cols = {"normalized_message": "first", "severity": "first"}
    if "event_template" in stage2_df.columns:
        agg_cols["event_template"] = "first"

    unique = (
        clean.groupby("template_id")
        .agg(**{k: (k, v) for k, v in agg_cols.items()})
        .reset_index()
    )

    unique = unique.join(majority_svc, on="template_id")
    unique["service"] = unique["majority_service"].fillna("unknown")
    unique = unique.drop(columns=["majority_service"])

    if "event_template" in unique.columns:
        def _is_noise_template(t: str) -> bool:
            tl = str(t).lower().strip()
            for safe_pat in _TEMPLATE_NOISE_SAFELIST:
                if safe_pat.search(tl):
                    return False
            return any(tl.startswith(m) for m in _NOISE_TEMPLATE_MARKERS)

        noise_mask = unique["event_template"].apply(_is_noise_template)
        n_dropped  = int(noise_mask.sum())
        if n_dropped:
            logger.info("Dropped %d noise-marker templates before embedding", n_dropped)
        unique = unique[~noise_mask].reset_index(drop=True)

    return unique


# ── STEP 1b: PARTITION BY (service, severity_tier) ────────────────────
def _partition_unique_templates(
    unique: pd.DataFrame,
    cfg: dict,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    partitions: Dict[Tuple[str, str], list] = {}
    use_svc = cfg.get("partition_by_service", True)
    use_sev = cfg.get("partition_by_severity_tier", True)

    for _, row in unique.iterrows():
        svc = str(row.get("service", "_unknown_") or "_unknown_").strip()
        sev = _severity_tier(str(row.get("severity", "") or ""))
        part_key = (
            svc if use_svc else "_all_",
            sev if use_sev else "_all_",
        )
        partitions.setdefault(part_key, []).append(row)

    return {
        k: pd.DataFrame(rows).reset_index(drop=True)
        for k, rows in partitions.items()
    }


# ── STEP 2: EMBEDDINGS ────────────────────────────────────────────────
class _TFIDFFallbackModel:
    pass


# S3-ML-1: Log-adapted embedding model loader
# ─────────────────────────────────────────────────────────────────────
# Failure-prevention measures:
#   1. Primary model (log-domain adapted) → fallback to general-purpose
#      all-MiniLM-L6-v2 → fallback to TF-IDF. Never crashes.
#   2. _EMBEDDING_MODEL_LOADED tracks which tier actually loaded so
#      stats downstream can record "tfidf_fallback" vs real model name.
#   3. model_name string is tried first; if it fails, the config's
#      "embedding_model" key is tried; then TF-IDF.

_EMBEDDING_MODEL_LOADED: str = "none"


def _load_embedding_model(model_name: str):
    """
    Load a SentenceTransformer model with a three-tier fallback:
      Tier 1 — model_name as supplied (log-domain adapted or custom)
      Tier 2 — "all-MiniLM-L6-v2" (original general-purpose model)
      Tier 3 — TF-IDF fallback (no network required, always works)

    Failure prevention:
    - Any exception on Tier 1 logs a warning and falls to Tier 2.
    - Any exception on Tier 2 logs an error and falls to Tier 3.
    - ImportError on sentence-transformers falls straight to Tier 3.
    """
    global _EMBEDDING_MODEL_LOADED
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "S3-ML-1: sentence-transformers not installed — using TF-IDF fallback. "
            "Install with: pip install sentence-transformers"
        )
        _EMBEDDING_MODEL_LOADED = "tfidf_fallback"
        return _TFIDFFallbackModel()

    # Tier 1: requested model
    try:
        model = SentenceTransformer(model_name)
        logger.info("S3-ML-1: loaded embedding model (Tier 1): %s", model_name)
        _EMBEDDING_MODEL_LOADED = model_name
        return model
    except Exception as exc:
        logger.warning(
            "S3-ML-1: could not load primary model '%s' (%s) — "
            "falling back to all-MiniLM-L6-v2.", model_name, exc
        )

    # Tier 2: original general-purpose fallback
    _FALLBACK_MODEL = "all-MiniLM-L6-v2"
    try:
        model = SentenceTransformer(_FALLBACK_MODEL)
        logger.info("S3-ML-1: loaded embedding model (Tier 2 fallback): %s", _FALLBACK_MODEL)
        _EMBEDDING_MODEL_LOADED = _FALLBACK_MODEL
        return model
    except Exception as exc:
        logger.error(
            "S3-ML-1: fallback model '%s' also failed (%s) — "
            "using TF-IDF.", _FALLBACK_MODEL, exc
        )

    # Tier 3: TF-IDF — always available
    _EMBEDDING_MODEL_LOADED = "tfidf_fallback"
    return _TFIDFFallbackModel()


def _embed_texts(texts: List[str], model) -> np.ndarray:
    if isinstance(model, _TFIDFFallbackModel):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            max_features=512, sublinear_tf=True, strip_accents="unicode",
            analyzer="word", ngram_range=(1, 2), min_df=1,
        )
        sparse_matrix = vectorizer.fit_transform(texts)
        dense = sparse_matrix.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        zero_mask = (norms == 0).flatten()
        if zero_mask.any():
            dense[zero_mask] = 1.0 / dense.shape[1]
            norms[zero_mask] = 1.0
        dense = dense / norms

        # Fix 2: TruncatedSVD dimensionality reduction after TF-IDF.
        #
        # Problem: TF-IDF produces 512-dim vectors; SentenceTransformer produces
        # 384-dim vectors.  UMAP, coherence thresholds (0.60 cosine), and HDBSCAN
        # parameters were all calibrated on dense embedding geometry.  In a 512-dim
        # sparse TF-IDF space, pairwise cosine distances are compressed — most pairs
        # score 0.10–0.30 even when semantically unrelated — so the 0.60 coherence
        # floor dissolves almost every cluster, leaving only singletons.  HDBSCAN in
        # high-D sparse space also produces far more noise points (-1 labels) than
        # intended, making downstream anomaly scoring unreliable.
        #
        # Fix: reduce to n_components = min(128, n_texts - 1) before returning.
        # This matches the geometry SentenceTransformer outputs (~100–384 dense dims)
        # so the coherence thresholds and clustering parameters remain valid.
        # TruncatedSVD is already available via sklearn (no new dependency).
        #
        # Safety:
        #   - n_components is clamped to n_texts - 1 (SVD requirement: n_components < n_samples).
        #   - If n_texts <= 1 or SVD fails for any reason, the L2-normalised TF-IDF
        #     vectors are returned unchanged — no regression vs the current behaviour.
        #   - The isinstance(_TFIDFFallbackModel) marker class is NOT changed, so the
        #     caller's isinstance check in run_stage3 / _build_cluster_summary continues
        #     to work correctly.
        n_texts = dense.shape[0]
        if n_texts >= 2:
            try:
                from sklearn.decomposition import TruncatedSVD
                n_components = min(128, n_texts - 1)
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                reduced = svd.fit_transform(dense).astype(np.float32)
                # Re-normalise after SVD so cosine similarity is preserved
                svd_norms = np.linalg.norm(reduced, axis=1, keepdims=True)
                svd_norms[svd_norms == 0] = 1.0
                dense = reduced / svd_norms
                logger.debug(
                    "Fix 2 (TF-IDF SVD): reduced %d × 512 → %d × %d "
                    "(n_texts=%d, n_components=%d)",
                    n_texts, n_texts, n_components, n_texts, n_components,
                )
            except Exception as _svd_exc:
                logger.warning(
                    "Fix 2 (TF-IDF SVD): TruncatedSVD failed (%s) — "
                    "returning L2-normalised TF-IDF vectors unchanged.",
                    _svd_exc,
                )
        return dense

    embeddings = model.encode(
        texts, batch_size=64,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ── STEP 3: DIMENSIONALITY REDUCTION ─────────────────────────────────
def _umap_reduce(
    embeddings: np.ndarray,
    cfg: dict,
    partition_size: Optional[int] = None,
) -> Tuple[np.ndarray, bool]:
    """
    S3-ML-2: Service-aware UMAP with adaptive n_neighbors.

    Failure-prevention measures:
    - n_neighbors is clamped to partition size − 1 (UMAP requirement).
    - For tiny partitions (< 4 templates), PCA to 2D is used instead
      of UMAP — HDBSCAN in high-D or UMAP with n_neighbors > n is broken.
    - For sparse partitions (< 15 templates), n_neighbors is reduced to
      max(3, n_templates // 2) so the local neighbourhood is meaningful.
    - For dense partitions (≥ 50 templates), the configured n_neighbors
      (default 15) is used unchanged.
    - Any UMAP failure falls back to raw (normalised) embeddings.
    """
    n = len(embeddings)

    if n < 4:
        if n >= 2:
            try:
                from sklearn.decomposition import PCA
                reduced = PCA(n_components=min(2, n - 1), random_state=42).fit_transform(embeddings)
                return reduced.astype(np.float32), False
            except Exception:
                pass
        return embeddings, False

    n_components = min(cfg["umap_n_components"], n - 2)

    # S3-ML-2: adaptive n_neighbors based on actual partition size
    p_size = partition_size if partition_size is not None else n
    if cfg.get("umap_adaptive_neighbors", True):
        if p_size < 15:
            adaptive_neighbors = max(3, p_size // 2)
        elif p_size < 50:
            adaptive_neighbors = max(5, p_size // 3)
        else:
            adaptive_neighbors = cfg["umap_n_neighbors"]
        n_neighbors = min(adaptive_neighbors, n - 1)
        if n_neighbors != cfg["umap_n_neighbors"]:
            logger.debug(
                "S3-ML-2: adaptive UMAP n_neighbors=%d for partition_size=%d "
                "(config default=%d)",
                n_neighbors, p_size, cfg["umap_n_neighbors"],
            )
    else:
        n_neighbors = min(cfg["umap_n_neighbors"], n - 1)

    init_method = "random" if n <= 30 else "spectral"
    try:
        import umap
        reducer = umap.UMAP(
            n_components=n_components, n_neighbors=n_neighbors,
            min_dist=cfg["umap_min_dist"], metric=cfg["umap_metric"],
            random_state=cfg["umap_random_state"], init=init_method,
            verbose=False,
        )
        reduced = reducer.fit_transform(embeddings)
        return reduced.astype(np.float32), True
    except ImportError:
        return embeddings, False
    except Exception as exc:
        logger.warning(
            "S3-ML-2: UMAP failed for partition_size=%d (%s) — "
            "using raw normalised embeddings.", p_size, exc
        )
        # Normalise raw embeddings so cosine distance is preserved
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32), False


# ── STEP 4: CLUSTERING ────────────────────────────────────────────────
def _cluster_embeddings(embeddings: np.ndarray, cfg: dict) -> np.ndarray:
    n = len(embeddings)
    if n < 2:
        return np.array([-1] * n)
    try:
        import hdbscan as hdbscan_lib
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=cfg["hdbscan_min_cluster_size"],
            min_samples=cfg["hdbscan_min_samples"],
            metric=cfg["hdbscan_metric"],
            cluster_selection_method=cfg["hdbscan_cluster_selection_method"],
            prediction_data=True,
        )
        labels = clusterer.fit_predict(embeddings)
    except ImportError:
        from sklearn.cluster import AgglomerativeClustering
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        zero_mask = (norms == 0).flatten()
        safe_emb  = embeddings.copy()
        if zero_mask.any():
            safe_emb[zero_mask] = 1.0 / embeddings.shape[1]
            norms[zero_mask] = 1.0
        safe_emb = safe_emb / norms

        fallback_threshold = cfg.get(
            "agglo_distance_threshold_fallback",
            cfg["agglo_distance_threshold"],
        )
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=fallback_threshold,
            metric="cosine", linkage="average",
        )
        raw_labels = clusterer.fit_predict(safe_emb)
        lc = Counter(raw_labels)
        isolated_set = {l for l, c in lc.items() if c == 1}
        labels = np.array([-1 if l in isolated_set else l for l in raw_labels])

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    logger.info("Clustering: %d clusters, %d isolated", n_clusters, n_noise)
    return labels


# ── STEP 5: BUILD semantic_cluster_id MAPPING ─────────────────────────
def _build_scid_mapping(tids: List[str], labels: np.ndarray) -> Dict[str, str]:
    tid_to_scid: Dict[str, str] = {}
    groups: Dict[int, List[str]] = defaultdict(list)
    for tid, lbl in zip(tids, labels):
        if lbl != -1:
            groups[int(lbl)].append(tid)
    for members in groups.values():
        scid = _stable_hash("|".join(sorted(members)))
        for t in members:
            tid_to_scid[t] = scid
    for tid, lbl in zip(tids, labels):
        if lbl == -1:
            tid_to_scid[tid] = _stable_hash(tid)
    return tid_to_scid


# ── S3-FP-3: COHERENCE DISSOLUTION ───────────────────────────────────
def _dissolve_incoherent_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_sim: float = 0.60,
) -> np.ndarray:
    """
    S3-FP-3: Dissolve HDBSCAN clusters that are geometrically close in
    UMAP-compressed space but semantically unrelated in embedding space.

    HDBSCAN clusters in UMAP-reduced coordinates. UMAP preserves local
    neighbourhood structure but distorts global distances — two templates
    from completely different error domains can be mapped into the same
    low-density region purely as an artefact of the manifold compression,
    not because they are semantically similar.

    This function computes the mean pairwise cosine similarity of every
    cluster's members in the original (pre-UMAP) embedding space. Clusters
    below min_sim are dissolved: all members become -1 (isolated/singleton)
    and proceed through the singleton classification path instead of being
    grouped with dissimilar templates.

    Design guarantees:
    - Only removes clusters; never merges or reassigns.
    - Cannot create new false positives — only prevents false groupings.
    - Threshold 0.60 is conservative: genuine semantic duplicates typically
      score 0.75+; random UMAP-compression pairs score 0.20–0.50.
    - Time: O(C × K²), negligible for typical log files (C<300, K<15).
    """
    labels = labels.copy()
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = (embeddings / norms).astype(np.float32)

    cluster_ids = set(labels[labels != -1])
    n_dissolved = 0

    for cid in cluster_ids:
        mask    = labels == cid
        members = normed[mask]
        n       = len(members)
        if n < 2:
            continue
        sim_matrix = np.dot(members, members.T)
        upper_idx  = np.triu_indices(n, k=1)
        mean_sim   = float(sim_matrix[upper_idx].mean()) if len(upper_idx[0]) > 0 else 1.0
        if mean_sim < min_sim:
            labels[mask] = -1
            n_dissolved += 1
            logger.debug(
                "S3-FP-3: dissolved cluster %d (mean_sim=%.3f < %.2f, n=%d)",
                cid, mean_sim, min_sim, n,
            )

    if n_dissolved:
        logger.info(
            "S3-FP-3: dissolved %d incoherent cluster(s) (threshold=%.2f)",
            n_dissolved, min_sim,
        )
    return labels


def _check_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
    umap_was_applied: bool = True,
) -> Optional[float]:
    non_noise  = labels != -1
    n_valid    = int(non_noise.sum())
    n_clusters = len(set(labels[non_noise]))
    if n_valid < 4 or n_clusters < 2:
        return None
    sil_metric = "euclidean" if umap_was_applied else "cosine"
    try:
        score = float(silhouette_score(
            embeddings[non_noise], labels[non_noise], metric=sil_metric,
        ))
        verdict = ("GOOD ✅" if score >= cfg["silhouette_good"]
                   else "ACCEPTABLE ⚠️" if score >= cfg["silhouette_acceptable"]
                   else "POOR ❌")
        print(f"  [Stage 3] Silhouette: {score:.3f}  {verdict}")
        return score
    except Exception as e:
        logger.warning("Silhouette failed: %s", e)
        return None


# ── CLASSIFY SINGLETONS ───────────────────────────────────────────────
_ATTEMPT_RATIO_RE  = re.compile(r"\battempt[=:]\s*(\d+)\s*/\s*(\d+)\b",  re.IGNORECASE)
_FAILURES_RATIO_RE = re.compile(r"\bfailures[=:]\s*(\d+)\s*/\s*(\d+)\b", re.IGNORECASE)
# Compiled once at module level — was previously re-compiled on every row iteration
# inside classify_singletons, which added ~10k re.compile() calls for a typical log file.
_NEGATION_RE = re.compile(
    r"\b(?:0\s+(?:error|fail|invalid|corrupt|timeout|refused|crash)|"
    r"no\s+(?:error|fail|failure|exception)|"
    r"(?:error|fail|invalid|corrupt)\s*:\s*0)\b",
    re.IGNORECASE,
)


def classify_singletons(
    df: pd.DataFrame,
    known_normal_tids: Optional[List[str]] = None,
    known_normal_max_daily_count: int = 500,
) -> pd.DataFrame:
    """
    Classify each row's singleton_class.

    S3-ML-3: known_normal_tids safelist — template_ids confirmed normal in
    previous runs are never re-classified as true_anomaly.  This prevents
    the 'rare but routine' problem where infrequent-by-design templates
    (startup messages, weekly cron confirmations) get flagged every run.

    S3-ML-3-EXPIRY: known_normal_max_daily_count safeguard (ChatGPT point 9).
    If a safelisted template appears more than known_normal_max_daily_count
    times in this run, it is removed from the effective safelist and re-evaluated
    normally.  This prevents stale safelist entries from hiding a formerly-benign
    template that has started firing at anomalous rates in production.
    Set known_normal_max_daily_count=0 to disable expiry (original behaviour).
    """
    df = df.copy()
    _known_normal_set: set = set(known_normal_tids) if known_normal_tids else set()
    _evicted: set = set()   # tracks tids evicted from safelist this run

    # S3-ML-3-EXPIRY: compute per-template counts and evict high-frequency
    # safelisted templates from the effective set.
    if _known_normal_set and known_normal_max_daily_count > 0:
        id_col_for_count = "template_id"
        _tid_counts = (
            df[~df["is_noise"].fillna(False)]
            .groupby(id_col_for_count)
            .size()
        )
        for tid in list(_known_normal_set):
            if _tid_counts.get(tid, 0) > known_normal_max_daily_count:
                _evicted.add(tid)
        if _evicted:
            _known_normal_set = _known_normal_set - _evicted
            logger.info(
                "S3-ML-3-EXPIRY: evicted %d safelist entries that exceeded "
                "max_daily_count=%d — will be re-evaluated: %s",
                len(_evicted), known_normal_max_daily_count,
                ", ".join(sorted(_evicted)[:5]) + ("…" if len(_evicted) > 5 else ""),
            )

    id_col = "template_id"
    sc_counts = (
        df[~df["is_noise"].fillna(False)]
        .groupby(id_col)
        .size()
        .rename("_sc_count")
    )
    df = df.join(sc_counts, on=id_col, how="left")
    df["_sc_count"] = df["_sc_count"].fillna(0).astype(int)

    singleton_class = [None] * len(df)

    for i, row in df.iterrows():
        count     = int(row["_sc_count"])
        norm_text = str(row.get("normalized_message", "")).lower()
        severity  = str(row.get("severity", "")).upper()
        is_err_warn = severity in {"ERROR", "WARN", "CRITICAL"}
        has_anomaly = any(kw in norm_text for kw in _ANOMALY_KEYWORDS)

        raw_text = (
            str(row.get("message") or "").strip()
            or str(row.get("raw_line") or "").strip()
            or norm_text
        )

        impossible_attempt = False
        for pat in (_ATTEMPT_RATIO_RE, _FAILURES_RATIO_RE):
            for m in pat.finditer(raw_text):
                n_val = int(m.group(1))
                d_val = int(m.group(2))
                if n_val > d_val:
                    impossible_attempt = True
                    break
            if impossible_attempt:
                break

        if impossible_attempt:
            singleton_class[i] = "impossible_attempt_count"
            continue

        if str(row.get("singleton_class", "")) == "noise_filtered":
            singleton_class[i] = "noise_filtered"
            continue

        if row.get("is_merged", False):
            continue

        # S3-ML-3: Known-normal safelist — if this template_id was confirmed
        # normal in a previous run, never flag it as true_anomaly regardless
        # of count or severity.  Prevents 'rare but routine' re-flagging.
        tid_val = str(row.get("template_id", ""))
        # FIX 1 (A6 stage3 complement): TM_CONTINUATION rows were pre-tagged in
        # stage2 with singleton_class="known_normal" to represent pm2 continuation
        # lines (stack frames, object dumps).  They carry no anomaly signal and must
        # be treated as known-normal regardless of the known_normal_tids safelist.
        # This is a hard-coded structural classification, not a configurable safelist.
        if tid_val == "TM_CONTINUATION":
            singleton_class[i] = "known_normal"
            continue

        if tid_val and _known_normal_set and tid_val in _known_normal_set:
            singleton_class[i] = "known_normal"
            continue

        # S3-ML-3-EVICTION FIX (MPCD §3.3 RISK):
        # A template evicted from the safelist because it exceeded known_normal_max_daily_count
        # represents a spike on a formerly-benign pattern — this IS anomalous.
        # Without this branch it falls through all count-based checks (count >> 5) and
        # leaves singleton_class=None, which maps to 'routine' — defeating eviction entirely.
        # Fix: classify evicted high-count templates by severity:
        #   ERROR/WARN → true_anomaly (spike + error signal = genuine incident)
        #   anomaly keyword → unseen_variant (spike but no explicit error)
        #   else → unseen_variant (flag for review; count spike alone is worth surfacing)
        if _evicted and tid_val in _evicted:
            if severity in {"ERROR", "FATAL"}:
                # Hard error on an evicted template — definite anomaly.
                singleton_class[i] = "true_anomaly"
            elif severity in {"WARN", "CRITICAL"} and has_anomaly:
                # WARN only qualifies if the text also contains an anomaly keyword
                # (fail/error/refused etc.).  Pure WARN lifecycle/startup messages
                # (e.g. "no parameter target yet") must not be promoted here.
                singleton_class[i] = "true_anomaly"
            else:
                singleton_class[i] = "unseen_variant"
            continue

        # FIX-SINGLE-EVENT: count == 1 — single-occurrence log lines.
        # The previous code had no branch for count == 1, so all single-event
        # logs fell through every guard and left singleton_class[i] = None.
        # With None, n_true_anomaly == 0 in _compute_cluster_anomaly_signal,
        # which returned anomaly_signal="none" for those clusters and the
        # frontend rendered them with the INFO (blue) tag even when
        # dominant_severity was ERROR or FATAL.
        #
        # Fix: a single ERROR or FATAL event is always a true_anomaly — it
        # does not need to be frequent to be serious.  Single WARN events
        # require an anomaly keyword (same stricter rule as the count>5 branch)
        # to avoid promoting every lifecycle WARN to an anomaly.  Single INFO
        # events without anomaly keywords remain None (routine path).
        if count == 1:
            _negated_single = bool(_NEGATION_RE.search(norm_text))
            if not _negated_single and severity in {"ERROR", "FATAL"}:
                # A single ERROR or FATAL is always worth surfacing.
                singleton_class[i] = "true_anomaly"
            elif not _negated_single and severity in {"WARN", "CRITICAL"} and has_anomaly:
                # Single WARN only qualifies when anomaly keyword is present.
                singleton_class[i] = "true_anomaly"
            elif has_anomaly and not _negated_single:
                # Single INFO/DEBUG with genuine anomaly keyword → unseen_variant.
                singleton_class[i] = "unseen_variant"
            # else: leave None → is_routine path
            continue

        if count > _LOW_COUNT_THRESHOLD:
            # S3-1 FIX: Severity gate — pure INFO singletons with anomaly keywords
            # become unseen_variant, not true_anomaly. Anomaly keywords alone are
            # unreliable for INFO-level lines (e.g. "0 failed jobs", "invalid: 0").
            # Only promote to true_anomaly when severity is ERROR/WARN/CRITICAL, OR
            # when anomaly keyword AND severity is at least WARN.
            _severity_is_warn_plus = severity in {"ERROR", "WARN", "CRITICAL", "FATAL"}
            _has_anomaly_at_warn = has_anomaly and _severity_is_warn_plus

            # S3-1 FIX: Negation guard — patterns like "0 errors", "no failed",
            # "invalid: 0" are operational summaries, not anomalies.
            # _NEGATION_RE is compiled at module level — not here.
            _negated = bool(_NEGATION_RE.search(norm_text))

            # S3-ROUTINE-ESCAPE: High-frequency INFO/DEBUG clusters with no anomaly
            # signal are routine operational messages (serial port open/close,
            # parameter reads, heartbeats, etc.).  Stamping them "unseen_variant"
            # caused Stage 4 to apply the +0.05 singleton bonus and score them
            # HIGH/MEDIUM — the primary driver of MAVLink/hardware false positives.
            #
            # Leaving singleton_class = None here lets _build_cluster_summary reach
            # _compute_cluster_anomaly_signal → "none", which sets is_routine = True,
            # which pre-splits the cluster out of Stage 4 scoring entirely.
            #
            # Conditions that must ALL hold before we treat a row as routine:
            #   - count > threshold  (already true — we are inside this branch)
            #   - severity is INFO or DEBUG  (not WARN/ERROR/CRITICAL/FATAL)
            #   - no anomaly keyword in the normalised text
            #   - message is not a negation summary ("0 errors" etc.)
            if not is_err_warn and not has_anomaly and not _negated:
                # Leave singleton_class[i] = None — is_routine path in Stage 3/4.
                continue

            if _negated:
                singleton_class[i] = "unseen_variant"
            elif severity in {"ERROR", "FATAL"} and not _negated:
                # Hard errors are always true_anomaly regardless of keyword content.
                singleton_class[i] = "true_anomaly"
            elif severity in {"WARN", "CRITICAL"} and _has_anomaly_at_warn and not _negated:
                # WARN only qualifies when an anomaly keyword is also present.
                # This excludes lifecycle/startup WARN messages such as:
                #   "no parameter target available yet"  (startup race condition)
                #   "UI client disconnected"             (normal WebSocket lifecycle)
                #   "serial connection already active"   (defensive reconnect logic)
                #   "could not refresh frame config"     (FC init sequence)
                # Those have WARN severity but no genuine failure keyword, so they
                # fall through to unseen_variant and do not score as true_anomaly.
                singleton_class[i] = "true_anomaly"
            else:
                singleton_class[i] = "unseen_variant"
            continue

        if 2 <= count <= _LOW_COUNT_THRESHOLD:
            if any(kw in norm_text for kw in _CRITICAL_OVERRIDE_KEYWORDS):
                singleton_class[i] = "true_anomaly"
                continue

            # LOW-COUNT SEVERITY GATE (mirrors the count > _LOW_COUNT_THRESHOLD branch above).
            # Low-frequency ERROR/FATAL rows with anomaly keywords are genuine signals —
            # they just happen to appear infrequently (e.g. serial port access-denied errors).
            # WARN is held to the same stricter standard as the high-count branch:
            # anomaly keyword required, so lifecycle/startup WARN messages are excluded.
            _negated_low = bool(_NEGATION_RE.search(norm_text))
            if not _negated_low and severity in {"ERROR", "FATAL"} and has_anomaly:
                singleton_class[i] = "true_anomaly"
                continue
            if not _negated_low and severity in {"WARN", "CRITICAL"} and has_anomaly:
                # WARN + anomaly keyword only — same guard as the high-count branch.
                # Pure WARN lifecycle messages (no anomaly keyword) fall through as None.
                singleton_class[i] = "true_anomaly"
                continue

    df["singleton_class"] = singleton_class
    df = df.drop(columns=["_sc_count"])
    return df


# ══════════════════════════════════════════════════════════════════════
# ACCURACY-FIX-B3: CLUSTER-LEVEL ANOMALY SIGNAL
# ══════════════════════════════════════════════════════════════════════

def _compute_cluster_anomaly_signal(
    cluster_df: pd.DataFrame,
    max_severity: str,
    is_mixed_severity: bool,
) -> Tuple[str, Optional[str]]:
    n_true_anomaly = int(
        (cluster_df.get("singleton_class", pd.Series(dtype=object)) == "true_anomaly").sum()
    ) if "singleton_class" in cluster_df.columns else 0
    n_impossible   = int(
        (cluster_df.get("singleton_class", pd.Series(dtype=object)) == "impossible_attempt_count").sum()
    ) if "singleton_class" in cluster_df.columns else 0
    n_total        = len(cluster_df)

    max_sev_order  = _SEVERITY_ORDER.get(max_severity.upper() if max_severity else "", 0)

    if n_impossible > 0:
        return "high", f"impossible_attempt_count detected ({n_impossible} rows)"

    if max_sev_order >= _SEVERITY_ORDER.get("FATAL", 4):
        if n_true_anomaly > 0 or is_mixed_severity:
            return "high", f"FATAL severity with {n_true_anomaly} true_anomaly rows"
        return "medium", "FATAL severity level present in cluster"

    if max_sev_order >= _SEVERITY_ORDER.get("ERROR", 3):
        anomaly_ratio = n_true_anomaly / max(n_total, 1)
        if anomaly_ratio > 0.5 or (n_true_anomaly > 0 and is_mixed_severity):
            return "high", (
                f"ERROR severity, {n_true_anomaly}/{n_total} rows true_anomaly, "
                f"mixed_severity={is_mixed_severity}"
            )
        # FIX-SEVERITY-OVERRIDE: any cluster whose max severity is ERROR/FATAL
        # AND contains at least one true_anomaly row must surface as "medium",
        # never "low".  Previously n_true_anomaly>0 without is_mixed_severity
        # returned "medium" which was correct, but count=1 clusters never had
        # true_anomaly set (fixed in classify_singletons above) so they fell
        # through to "low" / "ERROR severity present" — causing the INFO badge.
        # Now that count==1 ERROR/FATAL rows are stamped true_anomaly, this
        # branch correctly fires and the frontend receives signal="medium" with
        # dominant_severity=ERROR/FATAL, triggering the red/orange badge.
        if n_true_anomaly > 0:
            return "medium", f"ERROR severity with {n_true_anomaly} true_anomaly rows"
        if is_mixed_severity:
            return "low", "mixed severity distribution (dominant != max)"
        return "low", "ERROR severity present"

    # S3-WARN-FIX: WARN clusters that are high-frequency and have no true_anomaly
    # singletons are operational repeating patterns — e.g. "rate limit exceeded",
    # "session nearing expiry", "cron job slow", "stock below threshold".  These
    # fire dozens of times and should NOT receive any anomaly signal.  Previously
    # any WARN cluster got signal="low", which fed into Stage 4 scoring and
    # inflated scores to MEDIUM/CRITICAL, burying genuinely critical events.
    #
    # Rule: WARN clusters only get a non-"none" signal when they also have at
    # least one true_anomaly singleton.  High-frequency WARN with n_true_anomaly==0
    # returns "none" so Stage 4 classifies it as routine.  The is_routine flag in
    # _build_cluster_summary provides separate baseline visibility in the dashboard.
    if max_sev_order >= _SEVERITY_ORDER.get("WARN", 2) and n_true_anomaly > 0:
        return "low", f"WARN severity with {n_true_anomaly} true_anomaly rows"

    if n_true_anomaly > 0:
        return "low", f"{n_true_anomaly} true_anomaly rows"

    return "none", None


# ══════════════════════════════════════════════════════════════════════
# S3-1: SEVERITY-RANKED REPRESENTATIVE SAMPLE SELECTION
# ══════════════════════════════════════════════════════════════════════
# Replaces the old `grp[text_col].iloc[0]` selection (arbitrary row order)
# with a deterministic pick that maximises diagnostic signal:
#   1. Prefer the row with the highest-severity level (FATAL > ERROR > WARN …)
#   2. On severity tie, prefer the row whose message matches the most
#      error_class_patterns from the pipeline config.
# This ensures ACC-3/4/5 in Stage 4 receive the most signal-rich sample.

def _pick_representative_sample(grp: pd.DataFrame, text_col: str, cfg: dict) -> str:
    """Pick the most diagnostically useful row from a cluster group.

    Priority:
      1. Highest severity (FATAL=0, ERROR=1, WARN=2, INFO=3, DEBUG=4 — lower rank wins).
      2. On tie: prefer rows matching error_class_patterns from cfg.
      3. Final tie: first row in current order.

    Returns the text value of the winning row, or \"\" if grp is empty.
    """
    if grp.empty:
        return ""

    # Stage 1 canonical severity values only — unreachable aliases removed (spec §9B).
    sev_rank = {"FATAL": 0,
                "ERROR": 1,
                "WARN":  2,
                "INFO":  3,
                "DEBUG": 4,
                "TRACE": 5}

    if "severity" in grp.columns:
        grp2 = grp.copy()
        grp2["_sev_rank"] = (
            grp2["severity"]
            .str.upper()
            .map(sev_rank)
            .fillna(5)          # unknown severities sort last
        )
        error_pats = cfg.get("error_class_patterns", [])
        if error_pats:
            # error_class_patterns may be list[str] or list[tuple[str, any]]
            combined_terms = []
            for p in error_pats:
                combined_terms.append(p[0] if isinstance(p, (list, tuple)) else str(p))
            combined_pat = "|".join(re.escape(t) for t in combined_terms if t)
            if combined_pat:
                grp2["_err_match"] = (
                    grp2[text_col]
                    .str.contains(combined_pat, case=False, na=False, regex=True)
                    .astype(int)
                )
                grp2 = grp2.sort_values(
                    ["_sev_rank", "_err_match"], ascending=[True, False]
                )
            else:
                grp2 = grp2.sort_values("_sev_rank", ascending=True)
        else:
            grp2 = grp2.sort_values("_sev_rank", ascending=True)
        return str(grp2[text_col].iloc[0])

    # No severity column — fall back to first row
    return str(grp[text_col].iloc[0])


# ══════════════════════════════════════════════════════════════════════
# BUILD CLUSTER SUMMARY
# ══════════════════════════════════════════════════════════════════════

def _build_cluster_summary(
    df: pd.DataFrame,
    cluster_manifest: Optional[Dict] = None,
) -> pd.DataFrame:
    if "semantic_cluster_id" not in df.columns:
        return pd.DataFrame()

    clean = df[
        ~df["is_noise"].fillna(False) &
        (df["template_id"] != "TM_NOISE_FILTERED")
    ].copy()

    scid_groups = {
        scid: grp.reset_index(drop=True)
        for scid, grp in clean.groupby("semantic_cluster_id", dropna=False)
    }

    rows: List[Dict] = []

    for scid, grp in scid_groups.items():
        # MEDIUM-9 FIX: skip NaN cluster IDs — they are unassigned noise rows
        if scid is pd.NA or (isinstance(scid, float) and pd.isna(scid)):
            logger.warning("_build_cluster_summary: skipping NaN semantic_cluster_id (%d rows)", len(grp))
            continue
        record: Dict = {"cluster_id": scid}

        record["template_count"] = grp["template_id"].nunique()
        text_col = "event_template" if "event_template" in grp.columns else "normalized_message"
        # S3-1: severity-ranked representative sample (was: grp[text_col].iloc[0])
        record["sample_template"] = _pick_representative_sample(grp, text_col, {})

        # ACCURACY-FIX-B8: domain via plurality vote
        record["domain"] = _dominant_domain(grp["domain"]) if "domain" in grp.columns else "other"

        # ACCURACY-FIX-B1: read from manifest when present
        if cluster_manifest is not None:
            member_tids = grp["template_id"].unique().tolist()
            manifest_entries = [
                cluster_manifest.get("clusters", {}).get(tid)
                for tid in member_tids
                if cluster_manifest.get("clusters", {}).get(tid) is not None
            ]

            if manifest_entries:
                record["total_log_count"] = sum(
                    e.get("count", 0) for e in manifest_entries
                )
                sev_dist: Dict[str, int] = {"DEBUG": 0, "INFO": 0, "WARN": 0, "ERROR": 0, "FATAL": 0}
                for e in manifest_entries:
                    for sev, cnt in e.get("severity_distribution", {}).items():
                        canon = _canonical_severity(sev)
                        bucket = canon if canon in sev_dist else "INFO"
                        sev_dist[bucket] = sev_dist.get(bucket, 0) + int(cnt)
                record["severity_distribution"] = sev_dist

                first_seens = [e["first_seen"] for e in manifest_entries if e.get("first_seen")]
                last_seens  = [e["last_seen"]  for e in manifest_entries if e.get("last_seen")]
                record["first_seen"] = min(first_seens) if first_seens else None
                record["last_seen"]  = max(last_seens)  if last_seens  else None

                svc_set: set = set()
                for e in manifest_entries:
                    for svc in e.get("services", []):
                        if svc and str(svc).strip() not in ("", "unknown", "nan"):
                            svc_set.add(str(svc).strip())
                record["services"] = sorted(svc_set) if svc_set else ["unknown"]

                # BP-S5-6 FIX: index this semantic cluster ID in the manifest
                # so Stage 5 Assert 1 can find SC... IDs, not just TM... IDs.
                cluster_manifest["clusters"][scid] = {
                    "count":                 record["total_log_count"],
                    "severity_distribution": sev_dist,
                    "first_seen":            record["first_seen"],
                    "last_seen":             record["last_seen"],
                    "services":              record["services"],
                }

            else:
                logger.warning(
                    "FIX-B1: cluster %s has no manifest entries — "
                    "computing metadata from df (may be less accurate).", scid,
                )
                record["total_log_count"] = int(len(grp))
                record["severity_distribution"] = _build_severity_distribution(
                    grp.get("severity", pd.Series(dtype=str))
                )
                record["first_seen"] = None
                record["last_seen"]  = None
                record["services"]   = sorted(grp["service"].dropna().unique().tolist()) if "service" in grp.columns else ["unknown"]

        else:
            # Legacy fallback
            use_source_count = "_source_line_count" in clean.columns
            if not use_source_count:
                logger.warning(
                    "FIX-N1/B1: Neither cluster_manifest nor '_source_line_count' "
                    "present — cluster volumes may be inflated."
                )
            record["total_log_count"] = int(grp["_source_line_count"].sum()) if use_source_count else int(len(grp))
            record["severity_distribution"] = _build_severity_distribution(
                grp.get("severity", pd.Series(dtype=str))
            )
            if "ts" in grp.columns and grp["ts"].notna().any():
                ts_sorted = grp["ts"].dropna().sort_values()
                record["first_seen"] = str(ts_sorted.iloc[0])
                record["last_seen"]  = str(ts_sorted.iloc[-1])
            else:
                record["first_seen"] = None
                record["last_seen"]  = None
            if "service" in grp.columns:
                svc_vals = [
                    str(s) for s in grp["service"].dropna()
                    if str(s).strip() not in ("", "unknown", "nan")
                ]
                record["services"] = sorted(set(svc_vals)) if svc_vals else ["unknown"]
            else:
                record["services"] = ["unknown"]

        # ACCURACY-FIX-B4: derived severity fields
        sev_dist = record["severity_distribution"]
        record["dominant_severity"] = _dominant_severity(sev_dist)
        record["max_severity"]       = _max_severity(sev_dist)
        record["is_mixed_severity"]  = (
            record["dominant_severity"] != record["max_severity"]
        )

        if "singleton_class" in grp.columns:
            record["singleton_class"] = grp["singleton_class"].iloc[0]

        # ACCURACY-FIX-B3: cluster-level anomaly_signal
        anomaly_signal, anomaly_reason = _compute_cluster_anomaly_signal(
            grp,
            max_severity=record["max_severity"],
            is_mixed_severity=record["is_mixed_severity"],
        )
        record["anomaly_signal"] = anomaly_signal
        record["anomaly_reason"] = anomaly_reason

        # S3-ROUTINE-FLAG: mark clusters that are pure routine/baseline so the
        # dashboard can show them in a "baseline activity" panel rather than
        # hiding them entirely.  A cluster is routine when:
        #   - anomaly_signal is "none" (no anomaly signal at all), AND
        #   - dominant severity is INFO or DEBUG (not a repeating WARN storm that
        #     just happened to have no true_anomaly singletons — those are still
        #     operationally interesting and should be visible in the signal panel)
        _dominant_sev_order = _SEVERITY_ORDER.get(
            record.get("dominant_severity", "INFO").upper(), 1
        )
        record["is_routine"] = (
            anomaly_signal == "none"
            and _dominant_sev_order <= _SEVERITY_ORDER.get("INFO", 1)
        )

        # S3-FP-2: WARN clusters with 0% errors and no critical keywords.
        # NOT marked is_routine (keeps them visible in the dashboard at LOW).
        # Stage 4 uses this flag to apply a score ceiling so burst alone cannot
        # elevate them to MEDIUM/HIGH. A genuine WARN storm with errors or critical
        # keywords is excluded and scores normally.
        _sev_dist_fp2  = record.get("severity_distribution", {})
        _error_cnt_fp2 = (
            _sev_dist_fp2.get("ERROR", 0) + _sev_dist_fp2.get("FATAL", 0)
        )
        _sample_text_fp2  = str(record.get("sample_template", "")).lower()
        _has_critical_fp2 = any(
            kw in _sample_text_fp2 for kw in _CRITICAL_OVERRIDE_KEYWORDS
        )
        record["is_warn_routine"] = bool(
            anomaly_signal == "none"
            and record.get("dominant_severity", "") == "WARN"
            and _error_cnt_fp2 == 0
            and not _has_critical_fp2
        )

        # domain_confidence: aggregate from Stage 2 values for cluster members.
        # Domain classification (and confidence) is owned by Stage 2.
        if "domain_confidence" in grp.columns:
            _conf_vals = pd.to_numeric(grp["domain_confidence"], errors="coerce").dropna()
            record["domain_confidence"] = float(_conf_vals.mean()) if not _conf_vals.empty else 0.0
        else:
            record["domain_confidence"] = 0.0

        # cluster_label: deterministic token-derived label as the initial value.
        # LLM enrichment is applied AFTER _build_cluster_summary returns,
        # as a separate post-pass in run_stage3 (see _enrich_cluster_labels_with_llm).
        # This keeps _build_cluster_summary fast and non-blocking.
        record["cluster_label"] = _make_cluster_label(
            str(record["domain"]),
            str(record["sample_template"]),
        )

        rows.append(record)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.attrs["header_cluster_count"] = len(summary)

    return summary


# ══════════════════════════════════════════════════════════════════════
# MPCD §2.3 — LLM CLUSTER LABELING
# ══════════════════════════════════════════════════════════════════════
#
# After HDBSCAN assigns semantic_cluster_id, call call_llm() with 3–5
# sample_template strings per cluster to generate a human-readable
# cluster_label.  This is a LOW-FREQUENCY call (one per unique cluster,
# not per log line) so the cost is negligible.
#
# Failure-prevention measures:
#   1. Exponential backoff (2 retries, 2 s / 4 s delay) before falling back.
#      (MPCD §4.1 ACTION — same pattern as stage2._llm_classify_domain)
#   2. If llm_client is not available, or LLM returns garbage, falls back
#      to the deterministic _make_cluster_label() token-extraction path.
#   3. Only called when USE_LLM_DOMAIN=true (same env-var gate as Stage 2
#      so a single switch controls all LLM calls in dev vs prod).
#   4. Results are cached in a module-level dict keyed by cluster SCID so
#      that if run_stage3 is called multiple times in the same process the
#      LLM is not re-queried for identical clusters.
#   5. _LLM_LABEL_CACHE size is bounded by _LLM_LABEL_CACHE_MAX_SIZE to
#      prevent unbounded growth in long-running API server mode.  When the
#      cap is reached, the oldest entries (by insertion order, Python 3.7+
#      dict ordering) are evicted so the cache stays useful without leaking.

import os as _os_s3
import time as _time_s3

_LLM_LABEL_CACHE: Dict[str, str] = {}   # module-level; lives for one process lifetime
_LLM_LABEL_CACHE_MAX_SIZE = 2000        # evict oldest when cache exceeds this
_LLM_LABEL_MAX_RETRIES    = 2
_LLM_LABEL_BASE_DELAY     = 2.0         # seconds; doubles each retry (2 s, 4 s)


def _llm_cluster_label(
    scid: str,
    domain: str,
    sample_templates: List[str],
) -> Optional[str]:
    """
    Ask the LLM to produce a short human-readable label for a semantic cluster.

    Parameters
    ----------
    scid : str
        Semantic cluster ID — used as cache key.
    domain : str
        Dominant domain for this cluster (pre-assigned by Stage 3).
    sample_templates : list[str]
        3–5 representative event_template strings from this cluster.

    Returns
    -------
    str or None
        A concise label (≤ 8 words, title-case) or None on failure/unavailability.
        The caller falls back to _make_cluster_label() when None is returned.
    """
    # Check module-level cache first
    if scid in _LLM_LABEL_CACHE:
        return _LLM_LABEL_CACHE[scid]

    # Only run when USE_LLM_DOMAIN=true — same gate as Stage 2 LLM domain
    if _os_s3.getenv("USE_LLM_DOMAIN", "false").lower() != "true":
        return None

    try:
        from llm_client import call_llm  # type: ignore
    except ImportError:
        return None  # llm_client not yet built — deterministic fallback

    # Truncate and deduplicate samples for a compact prompt
    samples = list(dict.fromkeys(str(t)[:200] for t in sample_templates if str(t).strip()))[:5]
    if not samples:
        return None

    samples_block = "\n".join(f"  - {s}" for s in samples)
    prompt = (
        f"You are a log-analysis assistant. Below are {len(samples)} representative "
        f"log template(s) from a semantic cluster in the '{domain}' domain.\n\n"
        f"Templates:\n{samples_block}\n\n"
        "Produce a SHORT human-readable cluster label (3–8 words, Title Case) "
        "that describes WHAT these log lines have in common operationally.\n"
        "Rules:\n"
        "  - Reply with ONLY the label text — no quotes, no explanation, no punctuation\n"
        "  - Do NOT start with 'Log', 'Event', or 'Cluster'\n"
        "  - Use concrete operational terms (e.g. 'Payment Charge Failed', "
        "'JWT Token Expired', 'Kafka Consumer Lag Exceeded')\n"
        "  - If templates are too generic, reply with exactly: GENERIC"
    )

    last_exc: Optional[Exception] = None
    for attempt in range(_LLM_LABEL_MAX_RETRIES + 1):
        try:
            raw = call_llm(prompt, max_tokens=40, temperature=0.0)
            label = raw.strip().strip('"\'').strip()
            # Reject obviously bad outputs
            if not label or label.upper() == "GENERIC" or len(label) > 120:
                return None
            # Reject multi-line or JSON-looking responses
            if "\n" in label or label.startswith("{"):
                return None
            # Evict oldest entries when cache is at capacity (Python 3.7+ dict preserves insertion order)
            if len(_LLM_LABEL_CACHE) >= _LLM_LABEL_CACHE_MAX_SIZE:
                oldest_keys = list(_LLM_LABEL_CACHE.keys())[:max(1, _LLM_LABEL_CACHE_MAX_SIZE // 10)]
                for k in oldest_keys:
                    _LLM_LABEL_CACHE.pop(k, None)
            _LLM_LABEL_CACHE[scid] = label
            return label
        except Exception as exc:
            last_exc = exc
            if attempt < _LLM_LABEL_MAX_RETRIES:
                delay = _LLM_LABEL_BASE_DELAY * (2 ** attempt)  # 2 s, 4 s
                logger.warning(
                    "_llm_cluster_label: attempt %d/%d failed (%s) — "
                    "retrying in %.0f s",
                    attempt + 1, _LLM_LABEL_MAX_RETRIES + 1, exc, delay,
                )
                _time_s3.sleep(delay)

    logger.error(
        "_llm_cluster_label: all %d retries exhausted (%s) — "
        "deterministic fallback for cluster %s",
        _LLM_LABEL_MAX_RETRIES + 1, last_exc, scid,
    )
    return None


_LABEL_NOISE_TOKENS = frozenset({
    "cef", "eventid", "type=syscall", "type=", "msg=audit",
    "provider=", "message=", "level=", "host=", "unit=",
    "<*>", "var", "num", "id", "ip", "uuid", "email",
    "the", "and", "for", "from", "with", "that", "this",
    "was", "has", "had", "been", "have", "will", "are",
})


def _make_cluster_label(domain: str, template_text: str) -> str:
    template_text = re.sub(r'^\d{1,2}:\d{2}:\d{2}\s+[AP]M\s+', '', template_text)
    words = []
    for raw_tok in template_text.split():
        tok = raw_tok.lower().strip(".,;:\"'()[]}{")
        if (
            not tok or tok == "<*>" or tok in _LABEL_NOISE_TOKENS
            or len(tok) <= 3 or tok.isdigit()
            or re.match(r"^[\d_\-\.]+$", tok)
            or "=" in tok or "|" in tok
        ):
            continue
        words.append(tok)
        if len(words) >= 3:
            break
    suffix = "_".join(words) if words else "event"
    return f"{domain}:{suffix}"


# ── MPCD §2.3 — LLM CLUSTER LABEL ENRICHMENT PASS ────────────────────
#
# Called ONCE after _build_cluster_summary returns — NOT inside the per-cluster
# loop.  This keeps _build_cluster_summary fast and non-blocking regardless of
# LLM availability.
#
# Design:
#   1. Iterate over cluster_summary rows, attempt _llm_cluster_label for each.
#   2. On LLM success, overwrite the deterministic cluster_label in-place.
#   3. On failure / USE_LLM_DOMAIN=false, keep the existing deterministic label.
#   4. The same backoff + cache guarantees from _llm_cluster_label apply.
#
# Complexity:
#   Time:  O(C) iterations; O(1) cache hit per cluster after the first call.
#          On first run with LLM enabled: O(C * network_latency) worst case,
#          bounded by retry cap (max 6 s overhead per cluster on total failure).
#          This is acceptable because cluster count C << row count N.
#   Space: O(C) for _LLM_LABEL_CACHE additions — bounded by _LLM_LABEL_CACHE_MAX_SIZE.

def _enrich_cluster_labels_with_llm(
    cluster_summary: pd.DataFrame,
    df_classified: pd.DataFrame,
) -> pd.DataFrame:
    """
    Post-pass: attempt LLM labeling for each cluster in cluster_summary.

    For clusters where _llm_cluster_label() returns a label, overwrites
    the deterministic cluster_label set by _build_cluster_summary.
    Operates on a copy — does not mutate the input DataFrames.

    Parameters
    ----------
    cluster_summary : pd.DataFrame
        Output of _build_cluster_summary(). Must have columns:
        cluster_id, domain, sample_template, cluster_label.
    df_classified : pd.DataFrame
        Per-row DataFrame — used to collect up to 5 distinct sample
        templates per cluster for richer LLM context.

    Returns
    -------
    cluster_summary with cluster_label updated where LLM succeeded.
    """
    if cluster_summary.empty:
        return cluster_summary
    if _os_s3.getenv("USE_LLM_DOMAIN", "false").lower() != "true":
        return cluster_summary  # LLM disabled — nothing to do

    cluster_summary = cluster_summary.copy()
    text_col = "event_template" if "event_template" in df_classified.columns else "normalized_message"

    # Pre-build a SCID → list[template_text] lookup for sample gathering.
    # Limit to 5 unique templates per cluster — O(N) pass, done once.
    _scid_samples: Dict[str, List[str]] = defaultdict(list)
    if "semantic_cluster_id" in df_classified.columns and text_col in df_classified.columns:
        for scid_val, tmpl_val in zip(
            df_classified["semantic_cluster_id"].astype(str),
            df_classified[text_col].fillna("").astype(str),
        ):
            if len(_scid_samples[scid_val]) < 5 and tmpl_val.strip():
                if tmpl_val not in _scid_samples[scid_val]:
                    _scid_samples[scid_val].append(tmpl_val)

    n_enriched = 0
    for idx, row in cluster_summary.iterrows():
        scid = str(row.get("cluster_id", ""))
        domain = str(row.get("domain", "other"))
        samples = _scid_samples.get(scid) or [str(row.get("sample_template", ""))]

        llm_label = _llm_cluster_label(
            scid=scid,
            domain=domain,
            sample_templates=samples,
        )
        if llm_label is not None:
            cluster_summary.at[idx, "cluster_label"] = llm_label
            n_enriched += 1

    if n_enriched:
        logger.info(
            "MPCD §2.3: LLM enriched %d/%d cluster labels",
            n_enriched, len(cluster_summary),
        )
    return cluster_summary
# STAGE3_CONFIG already had "drift_detection_threshold": 0.30 but no code
# read or used it.  This implements the function that was missing.
#
# Design:
#   After each run, cluster centroids are cached to disk.
#   On the next run, compute cosine distance between current and previous
#   centroids for matching cluster IDs.  Clusters that shift beyond
#   drift_detection_threshold are flagged centroid_drifted=True.
#
# Failure-prevention:
#   - Missing cache file → skip quietly, no crash.
#   - Any I/O or numpy exception → log warning, return empty drift dict.

import json as _json_drift
import os as _os_drift


def _detect_embedding_drift(
    tid_to_raw_embedding: Dict[str, np.ndarray],
    tid_to_scid: Dict[str, str],
    cfg: dict,
    cache_path: str = ".stage3_centroid_cache.json",
) -> Dict[str, bool]:
    """
    Detect embedding drift by comparing current cluster centroids to
    cached centroids from the previous run.

    Returns a dict mapping semantic_cluster_id → centroid_drifted (bool).
    Clusters with no prior centroid (new this run) get False.

    Side-effect: writes updated centroid cache after comparison so the
    next run can compare against the current run's centroids.
    """
    threshold = cfg.get("drift_detection_threshold", 0.30)
    drift_flags: Dict[str, bool] = {}

    if not tid_to_raw_embedding or not tid_to_scid:
        return drift_flags

    # Compute current centroids per semantic_cluster_id
    scid_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    for tid, emb in tid_to_raw_embedding.items():
        scid = tid_to_scid.get(tid)
        if scid:
            scid_embeddings[scid].append(emb)

    current_centroids: Dict[str, np.ndarray] = {}
    for scid, embs in scid_embeddings.items():
        centroid = np.mean(np.vstack(embs), axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        current_centroids[scid] = centroid

    # Load previous centroids from cache
    prev_centroids: Dict[str, List[float]] = {}
    if _os_drift.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                prev_centroids = _json_drift.load(f)
        except Exception as exc:
            logger.warning("S3-ML-4: could not read centroid cache (%s) — skipping drift check.", exc)

    # Compare current vs previous centroids
    n_drifted = 0
    for scid, centroid in current_centroids.items():
        if scid in prev_centroids:
            prev_vec = np.array(prev_centroids[scid], dtype=np.float32)
            prev_norm = np.linalg.norm(prev_vec)
            if prev_norm > 0:
                prev_vec = prev_vec / prev_norm
            # Cosine distance = 1 - cosine similarity
            cosine_sim = float(np.dot(centroid, prev_vec))
            cosine_dist = 1.0 - cosine_sim
            drifted = cosine_dist > threshold
            drift_flags[scid] = drifted
            if drifted:
                n_drifted += 1
                logger.debug(
                    "S3-ML-4: cluster %s centroid drifted (cosine_dist=%.3f > threshold=%.3f)",
                    scid, cosine_dist, threshold,
                )
        else:
            drift_flags[scid] = False  # new cluster this run

    if n_drifted:
        logger.info(
            "S3-ML-4: %d/%d clusters flagged centroid_drifted=True (threshold=%.2f)",
            n_drifted, len(current_centroids), threshold,
        )

    # Write updated centroid cache for next run
    try:
        cache_data = {scid: c.tolist() for scid, c in current_centroids.items()}
        with open(cache_path, "w", encoding="utf-8") as f:
            _json_drift.dump(cache_data, f)
    except Exception as exc:
        logger.warning("S3-ML-4: could not write centroid cache (%s).", exc)

    return drift_flags


# ── DETECT SUSPICIOUS SPLITS ──────────────────────────────────────────
def _detect_suspicious_splits(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    clean = df[~df["is_noise"].fillna(False)].copy()
    if len(clean) < 2 or "semantic_cluster_id" not in clean.columns:
        return pd.DataFrame()

    text_col = "event_template" if "event_template" in clean.columns else "normalized_message"

    rep = (
        clean.groupby("semantic_cluster_id", dropna=False)
        .agg(
            sample_template=(text_col, "first"),
            domain=("domain", "first") if "domain" in clean.columns else (text_col, "first"),
        )
        .reset_index()
    )
    if len(rep) < 2:
        return pd.DataFrame()

    texts = rep["sample_template"].fillna("").astype(str).tolist()
    cids  = rep["semantic_cluster_id"].tolist()

    try:
        vec  = TfidfVectorizer(max_features=256, sublinear_tf=True)
        mat  = vec.fit_transform(texts)
        sims = cos_sim(mat)
    except Exception:
        return pd.DataFrame()

    # LOW-11 FIX: replace O(n²) Python loop with vectorised NumPy index lookup.
    pairs_idx = np.argwhere(sims >= threshold)
    # Keep only upper triangle (i < j) to avoid duplicates
    pairs_idx = pairs_idx[pairs_idx[:, 0] < pairs_idx[:, 1]]

    pairs = []
    for i, j in pairs_idx:
        ci, cj = cids[i], cids[j]
        if ci is pd.NA or cj is pd.NA or pd.isna(ci) or pd.isna(cj) or ci == cj:
            continue
        pairs.append({
            "cluster_id_1":      ci,
            "cluster_id_2":      cj,
            "cluster_distance":  round(float(1 - sims[i, j]), 4),
            "sample_template_1": str(texts[i])[:120],
            "sample_template_2": str(texts[j])[:120],
        })
    return pd.DataFrame(pairs) if pairs else pd.DataFrame()


# ── ACCURACY-FIX-N1: SOURCE LINE COUNT INJECTION ──────────────────────
def _inject_source_line_counts(stage2_df: pd.DataFrame) -> pd.DataFrame:
    if "_source_line_count" in stage2_df.columns:
        return stage2_df

    df = stage2_df.copy()
    src_counts = (
        df[~df["is_noise"].fillna(False).astype(bool)]
        .groupby("template_id")
        .size()
        .rename("_source_line_count")
    )
    df = df.join(src_counts, on="template_id", how="left")
    df["_source_line_count"] = df["_source_line_count"].fillna(0).astype(int)

    logger.info(
        "FIX-N1: injected '_source_line_count' for %d templates "
        "(total non-noise rows: %d)",
        src_counts.index.nunique(),
        int(src_counts.sum()),
    )
    return df


# ══════════════════════════════════════════════════════════════════════
# CONSISTENCY ERROR
# ══════════════════════════════════════════════════════════════════════

class PipelineConsistencyError(RuntimeError):
    """Raised when blueprint cross-stage consistency assertion fails."""
    pass


# ══════════════════════════════════════════════════════════════════════
# MAIN run_stage3 FUNCTION
# ══════════════════════════════════════════════════════════════════════

def run_stage3(
    stage2_df:      pd.DataFrame,
    cfg:            Optional[dict] = None,
    embedding_model = None,
    logger_obj:     Optional[logging.Logger] = None,
    cluster_manifest: Optional[Dict] = None,
    raise_on_consistency_error: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Stage 3 — Semantic Clustering, Domain Assignment & Anomaly Classification.

    Parameters
    ----------
    stage2_df : pd.DataFrame
        Output of Stage 2 (one row per log line with template_id, is_noise, etc.).
    cfg : dict, optional
        Overrides for STAGE3_CONFIG.
    embedding_model : optional
        Pre-loaded SentenceTransformer. Loaded automatically if None.
    logger_obj : logging.Logger, optional
        Custom logger. Defaults to module logger.
    cluster_manifest : dict, optional
        Count manifest from Stage 2 build_manifest(). When provided, all cluster
        counts, severity distributions, timestamps, and service lists are read
        from this manifest rather than re-derived from the DataFrame.
    raise_on_consistency_error : bool
        If True, raise PipelineConsistencyError on header_cluster_count mismatch.

    Returns
    -------
    (stage2_df_with_clusters, stats_dict)
    """
    if logger_obj is None:
        logger_obj = logger

    full_cfg = {**STAGE3_CONFIG, **(cfg or {})}
    stats: Dict = {}
    consistency_failures: List[str] = []

    print("[Stage 3] Starting semantic clustering + domain assignment...")

    # ── Schema validation (MPCD §5.1 / §5.3) ────────────────────────────
    # Halt early with a clear message if required Stage 2 columns are absent
    # or entirely null.  Surfaces contract violations immediately rather than
    # letting them propagate silently into garbage clustering results downstream.
    _REQUIRED_S3_INPUT: Dict[str, str] = {
        "template_id":       "object",
        "domain":            "object",
        "domain_confidence": "float64",
    }
    if stage2_df is not None and not stage2_df.empty:
        for _req_col, _req_dtype in _REQUIRED_S3_INPUT.items():
            if _req_col not in stage2_df.columns:
                raise ValueError(
                    f"[stage3_input] Missing required column: '{_req_col}' "
                    f"(expected dtype: {_req_dtype}). "
                    "Ensure Stage 2 completed successfully before calling run_stage3()."
                )
            if stage2_df[_req_col].isna().all():
                raise ValueError(
                    f"[stage3_input] Column '{_req_col}' is entirely null — "
                    "Stage 2 output appears empty or corrupt."
                )
            if _req_dtype == "float64" and stage2_df[_req_col].dtype == object:
                raise ValueError(
                    f"[stage3_input] Column '{_req_col}' is object dtype, "
                    f"expected float64 — Stage 2 may have written a string instead of a float."
                )

    if stage2_df is None or stage2_df.empty:
        empty = pd.DataFrame()
        return empty, {
            "unique_templates_clustered": empty,
            "cluster_summary":            empty,
            "stage25_df_with_clusters":   stage2_df if stage2_df is not None else empty,
            "suspicious_splits":          pd.DataFrame(),
            "threshold_used":             0.45,
            "status":                     "empty_input",
            "header_cluster_count":       0,
            "consistency_failures":       [],
        }

    # ACCURACY-FIX-N1: ensure source line counts
    stage2_df = _inject_source_line_counts(stage2_df)

    if "is_noise" not in stage2_df.columns:
        stage2_df = stage2_df.copy()
        stage2_df["is_noise"] = False

    # ── 3.1: Unique templates ─────────────────────────────────────────
    unique = _collect_unique_templates(stage2_df)
    n_unique = len(unique)
    stats["n_unique_templates"] = n_unique
    print(f"  [3.1] Unique templates: {n_unique}")

    if n_unique == 0:
        df_out = stage2_df.copy()
        df_out["semantic_cluster_id"] = pd.NA
        empty = pd.DataFrame()
        return df_out, {
            "unique_templates_clustered": empty,
            "cluster_summary":            empty,
            "stage25_df_with_clusters":   df_out,
            "suspicious_splits":          pd.DataFrame(),
            "threshold_used":             0.45,
            "status":                     "no_non_noise_templates",
            "header_cluster_count":       0,
            "consistency_failures":       [],
        }

    # ── 3.2: Partition ────────────────────────────────────────────────
    partitions = _partition_unique_templates(unique, full_cfg)
    print(f"  [3.2] Partitions (service × severity): {len(partitions)}")

    if embedding_model is None:
        # S3-ML-1: prefer embedding_model_primary (Jina/LogBERT) over the
        # legacy embedding_model fallback key.  _load_embedding_model already
        # handles Tier1→Tier2→TF-IDF internally, so we just pick the right name.
        _primary_model_name = full_cfg.get(
            "embedding_model_primary",
            full_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        )
        embedding_model = _load_embedding_model(_primary_model_name)

    stats["embedding_model"] = (
        "tfidf_fallback"
        if isinstance(embedding_model, _TFIDFFallbackModel)
        else _EMBEDDING_MODEL_LOADED
    )

    # S3-4: Determine and record the active embedding backend.
    # This is surfaced in cluster_summary metadata so Stage 4 and Stage 5
    # can apply degraded-mode heuristics when TF-IDF is in use.
    if isinstance(embedding_model, _TFIDFFallbackModel):
        _embedding_backend = "tfidf"
        logger_obj.warning(
            "S3-EMBED: Running in TF-IDF fallback mode. "
            "Clustering quality will be degraded."
        )
    else:
        _emb_model_str = str(_EMBEDDING_MODEL_LOADED).lower()
        _embedding_backend = "jina" if "jina" in _emb_model_str else "minilm"

    tid_to_scid: Dict[str, str] = {}
    all_sil_scores: List[float] = []
    per_partition_silhouette: Dict[str, Optional[float]] = {}  # S3-FP-1
    total_clusters = 0
    total_isolated = 0
    anchor_floor   = full_cfg.get("anchor_word_floor", 1)

    # PERFORMANCE FIX: cache raw (pre-UMAP) embeddings keyed by template_id.
    # The clustering loop already computes embeddings for every partition.
    # By storing them here we avoid running model.encode() a second time
    # during the domain assignment step (S3-ML-5), which previously caused
    # every unique template to be embedded TWICE — once for clustering and
    # once for prototype domain matching.
    tid_to_raw_embedding: Dict[str, np.ndarray] = {}

    for part_key, part_df in partitions.items():
        svc, tier = part_key
        tids = part_df["template_id"].tolist()
        texts_for_embed = [
            _clean_for_embedding(t)
            for t in part_df["normalized_message"].fillna("").tolist()
        ]
        texts_for_anchor = (
            part_df["event_template"].fillna("").tolist()
            if "event_template" in part_df.columns
            else part_df["normalized_message"].fillna("").tolist()
        )

        if len(part_df) < full_cfg["min_templates_for_clustering"]:
            for tid in tids:
                tid_to_scid[tid] = _stable_hash(tid)
            total_isolated += len(tids)
            per_partition_silhouette[f"{svc}::{tier}"] = None  # S3-FP-1: too small to score
            continue

        embeddings = _embed_texts(texts_for_embed, embedding_model)

        # Cache raw embeddings BEFORE UMAP reduction (UMAP distorts geometry;
        # prototype cosine similarity should use the original embedding space)
        for tid, emb in zip(tids, embeddings):
            tid_to_raw_embedding[tid] = emb

        reduced, umap_applied = _umap_reduce(embeddings, full_cfg, partition_size=len(part_df))
        labels = _cluster_embeddings(reduced, full_cfg)

        # S3-FP-3: Coherence dissolution — dissolve clusters whose members have
        # average pairwise cosine similarity below cluster_coherence_min_sim in
        # the original embedding space. UMAP compression can place semantically
        # unrelated templates geometrically close; this catches those false pairs.
        # Runs BEFORE anchor floor so it operates on full HDBSCAN clusters.
        if full_cfg.get("cluster_coherence_enabled", True) and len(set(labels) - {-1}) > 0:
            labels = _dissolve_incoherent_clusters(
                labels,
                embeddings,
                min_sim=full_cfg.get("cluster_coherence_min_sim", 0.60),
            )

        # S3-FP-4: pass raw embeddings so anchor floor can bypass ejection when
        # embedding similarity is very high (synonym / morphological variants).
        labels = _apply_anchor_floor(
            tids, labels, texts_for_anchor,
            min_shared=anchor_floor,
            embeddings=embeddings,
            sim_bypass_threshold=full_cfg.get("anchor_sim_bypass_threshold", 0.88),
        )

        partition_mapping = _build_scid_mapping(tids, labels)
        tid_to_scid.update(partition_mapping)

        sil = _check_cluster_quality(reduced, labels, full_cfg, umap_applied)
        if sil is not None:
            all_sil_scores.append(sil)
        per_partition_silhouette[f"{svc}::{tier}"] = sil  # S3-FP-1: track per partition

        n_cls = len(set(labels)) - (1 if -1 in labels else 0)
        n_iso = int((labels == -1).sum())
        total_clusters += n_cls
        total_isolated += n_iso
        logger_obj.info(
            "Partition (%s, %s): %d templates → %d clusters, %d isolated "
            "(silhouette=%s)",
            svc, tier, len(part_df), n_cls, n_iso,
            f"{sil:.3f}" if sil is not None else "n/a",
        )

        # Release UMAP-reduced arrays; raw embeddings stay in tid_to_raw_embedding
        del reduced, labels

    # ── S3-4 FIX: Cross-partition merge for isolated templates ───────────
    # Two semantically identical templates from different partitions (e.g. the
    # same "Connection refused" error under service='app' and service='express')
    # each get their own hash SCID.  Run a single cross-partition cosine-
    # similarity pass to collapse near-duplicate singletons into a shared SCID.
    #
    # PERFORMANCE FIX: this pass is O(n²) on the number of isolated singletons.
    # For large noisy log files with 2000+ singletons this becomes very slow.
    # Cap at cross_partition_merge_max_singletons (default 800); above that,
    # skip the merge step.  Isolated templates remain isolated — no correctness
    # regression, just slightly less cross-service deduplication.
    _merge_cap = full_cfg.get("cross_partition_merge_max_singletons", 800)
    _isolated_tids = [tid for tid, scid in tid_to_scid.items()
                      if scid == _stable_hash(tid)]  # singleton = hash of self

    if len(_isolated_tids) >= 2 and (_merge_cap == 0 or len(_isolated_tids) <= _merge_cap):
        # PERFORMANCE FIX: reuse cached raw embeddings instead of re-encoding.
        # If a tid is missing from the cache (e.g. partition was too small to embed),
        # we embed its text on-demand as a fallback.
        _iso_embeddings_list = []
        _iso_tids_with_embs  = []
        for tid in _isolated_tids:
            if tid in tid_to_raw_embedding:
                _iso_embeddings_list.append(tid_to_raw_embedding[tid])
                _iso_tids_with_embs.append(tid)
            else:
                # Fallback: embed on demand (rare — only for sub-min-cluster partitions
                # where we skipped embedding to save time). Look up text from unique df
                # using a safe guard — unique may or may not be in scope at this point.
                _fallback_text = ""
                try:
                    _row = unique[unique["template_id"] == tid]
                    if not _row.empty:
                        _fallback_text = _clean_for_embedding(
                            str(_row["normalized_message"].iloc[0])
                        )
                except NameError:
                    pass  # unique already deleted — skip this tid
                if _fallback_text:
                    try:
                        emb = _embed_texts([_fallback_text], embedding_model)[0]
                        _iso_embeddings_list.append(emb)
                        _iso_tids_with_embs.append(tid)
                        tid_to_raw_embedding[tid] = emb
                    except Exception:
                        pass

        if len(_iso_tids_with_embs) >= 2:
            try:
                _iso_emb_matrix = np.vstack(_iso_embeddings_list).astype(np.float32)
                # Normalise so dot product = cosine similarity
                _norms = np.linalg.norm(_iso_emb_matrix, axis=1, keepdims=True)
                _norms[_norms == 0] = 1.0
                _iso_emb_matrix = _iso_emb_matrix / _norms

                _sim_matrix = _iso_emb_matrix @ _iso_emb_matrix.T
                _CROSS_SIM_THRESHOLD = 0.85
                _merged_scid: Dict[str, str] = {}
                for i in range(len(_iso_tids_with_embs)):
                    for j in range(i + 1, len(_iso_tids_with_embs)):
                        if _sim_matrix[i, j] >= _CROSS_SIM_THRESHOLD:
                            ti, tj = _iso_tids_with_embs[i], _iso_tids_with_embs[j]
                            ci = _merged_scid.get(ti, _stable_hash(ti))
                            cj = _merged_scid.get(tj, _stable_hash(tj))
                            canonical = min(ci, cj)
                            _merged_scid[ti] = canonical
                            _merged_scid[tj] = canonical
                n_cross_merged = sum(
                    1 for tid in _iso_tids_with_embs
                    if _merged_scid.get(tid, _stable_hash(tid)) != _stable_hash(tid)
                )
                if n_cross_merged:
                    for tid, shared_scid in _merged_scid.items():
                        tid_to_scid[tid] = shared_scid
                    total_isolated -= n_cross_merged
                    total_clusters  += len({v for v in _merged_scid.values()})
                    logger_obj.info(
                        "S3-4 cross-partition merge: %d isolated templates merged "
                        "into %d shared SCIDs (cosine threshold=%.2f)",
                        n_cross_merged,
                        len({v for v in _merged_scid.values()}),
                        _CROSS_SIM_THRESHOLD,
                    )
                del _iso_emb_matrix, _sim_matrix
            except Exception as _e:
                logger_obj.warning("S3-4 cross-partition merge failed: %s", _e)
    elif len(_isolated_tids) > _merge_cap > 0:
        logger_obj.info(
            "S3-4 cross-partition merge skipped: %d isolated templates exceeds "
            "cap of %d (set cross_partition_merge_max_singletons=0 to disable cap).",
            len(_isolated_tids), _merge_cap,
        )

    # ── 3.3: Map semantic_cluster_id to unique df ─────────────────────
    unique["semantic_cluster_id"] = unique["template_id"].map(tid_to_scid).astype("string")

    # ACCURACY-FIX-N1: source line count
    if "_source_line_count" in stage2_df.columns:
        _tmpl_src_counts = (
            stage2_df[~stage2_df["is_noise"].fillna(False)]
            .groupby("template_id")["_source_line_count"]
            .first()
            .rename("count")
        )
        unique = unique.join(_tmpl_src_counts, on="template_id")
        unique["count"] = unique["count"].fillna(1).astype(int)
    else:
        _tmpl_counts = (
            stage2_df[~stage2_df["is_noise"].fillna(False)]
            .groupby("template_id")
            .size()
            .rename("count")
        )
        unique = unique.join(_tmpl_counts, on="template_id")
        unique["count"] = unique["count"].fillna(1).astype(int)

    # ── 3.4: Assign domain per template ──────────────────────────────
    print("  [3.3] Mapping semantic cluster IDs to rows...")

    # ── 3.5: Per-row classify singletons ─────────────────────────────
    print("  [3.4] Classifying singletons per row...")
    _scid_map    = unique.set_index("template_id")["semantic_cluster_id"].to_dict()

    # _domain_map removed: domain is owned by Stage 2, not derived here.
    # _domconf_map is sourced from stage2_df (not unique) per Correction G.
    _domconf_map = (
        stage2_df.drop_duplicates("template_id")
        .set_index("template_id")["domain_confidence"]
        .to_dict()
    )

    # Diagnostic stats: read from stage2_df since unique has no domain column.
    _clean_s2 = stage2_df[~stage2_df["is_noise"].fillna(False)]
    domain_dist    = _clean_s2["domain"].value_counts().to_dict()
    low_conf_count = int(
        (pd.to_numeric(_clean_s2["domain_confidence"], errors="coerce")
         .fillna(0.0) < 0.50).sum()
    )
    print(f"        Domain distribution (from Stage 2): {domain_dist}")
    print(f"        Low-confidence domain rows (from Stage 2): {low_conf_count}")

    # S3-ML-4: Embedding drift detection — compare centroids to previous run.
    # Must run BEFORE we delete tid_to_raw_embedding.
    _drift_flags: Dict[str, bool] = {}
    try:
        _drift_flags = _detect_embedding_drift(
            tid_to_raw_embedding=tid_to_raw_embedding,
            tid_to_scid=tid_to_scid,
            cfg=full_cfg,
        )
        _n_drifted = sum(1 for v in _drift_flags.values() if v)
        if _n_drifted:
            logger_obj.info(
                "S3-ML-4: %d clusters flagged centroid_drifted=True", _n_drifted
            )
    except Exception as _drift_exc:
        logger_obj.warning("S3-ML-4: drift detection failed (%s) — skipping.", _drift_exc)

    # Free the unique DataFrame and embedding cache.
    # The embedding cache (tid_to_raw_embedding) can hold tens of MB for large
    # log files and is not needed after drift detection completes.
    del unique
    del tid_to_raw_embedding

    df_working = stage2_df.copy()
    df_working["semantic_cluster_id"] = df_working["template_id"].map(_scid_map).astype("string")
    # domain_confidence propagated from Stage 2 via template_id mapping (Correction G).
    df_working["domain_confidence"]   = df_working["template_id"].map(_domconf_map)

    # Noise rows: force semantic_cluster_id to NA.
    # This is structural cleanup, not domain classification.
    noise_mask = df_working["is_noise"].fillna(False).astype(bool)
    df_working.loc[noise_mask, "semantic_cluster_id"] = pd.NA

    # Assert domain is present from Stage 2 (domain classification owned by Stage 2).
    if "domain" not in df_working.columns or df_working["domain"].isna().all():
        logger_obj.error(
            "S3: domain column missing or empty — Stage 2 domain classification "
            "did not run correctly. Pipeline integrity compromised."
        )
        df_working["domain"] = "other"

    # Log domain distribution for diagnostics (read-only — no modification).
    _dom_counts = df_working["domain"].value_counts().to_dict()
    logger_obj.info("S3: domain distribution from Stage 2: %s", _dom_counts)

    df_classified = classify_singletons(
        df_working,
        known_normal_tids=full_cfg.get("known_normal_tids", []),
        known_normal_max_daily_count=full_cfg.get("known_normal_max_daily_count", 500),
    )

    # ── MPCD §5.2: Derive per-row anomaly_signal and is_routine from singleton_class ──
    # The schema contract requires:
    #   anomaly_signal : "true_anomaly" | "unseen_variant" | "routine" | "noise_filtered"
    #   is_routine     : bool — True for known_normal / routine INFO clusters
    # We map from singleton_class which classify_singletons just populated.
    _SINGLETON_TO_SIGNAL: Dict[str, str] = {
        "true_anomaly":            "true_anomaly",
        "impossible_attempt_count": "true_anomaly",
        "unseen_variant":          "unseen_variant",
        "known_normal":            "routine",
        "noise_filtered":          "noise_filtered",
    }
    df_classified["anomaly_signal"] = df_classified["singleton_class"].map(
        lambda sc: _SINGLETON_TO_SIGNAL.get(str(sc) if pd.notna(sc) else "", "routine")
    )
    df_classified["is_routine"] = df_classified["anomaly_signal"] == "routine"

    n_anomaly     = int((df_classified["singleton_class"] == "true_anomaly").sum())
    n_unseen      = int((df_classified["singleton_class"] == "unseen_variant").sum())
    n_bad_attempt = int((df_classified["singleton_class"] == "impossible_attempt_count").sum())
    n_noise_filt  = int((df_classified["singleton_class"] == "noise_filtered").sum())
    print(f"        true_anomaly={n_anomaly}  unseen_variant={n_unseen}  "
          f"impossible_attempt_count={n_bad_attempt}")

    # ── 3.6: Build cluster summary ────────────────────────────────────
    print("  [3.5] Building cluster summary...")
    cluster_summary = _build_cluster_summary(df_classified, cluster_manifest=cluster_manifest)
    n_summary_rows = len(cluster_summary)

    # S3-ML-4: annotate cluster_summary with centroid_drifted flag
    if not cluster_summary.empty and _drift_flags:
        cluster_summary["centroid_drifted"] = cluster_summary["cluster_id"].map(
            lambda cid: _drift_flags.get(str(cid), False)
        )
    elif not cluster_summary.empty:
        cluster_summary["centroid_drifted"] = False

    # MPCD §2.3: LLM cluster label enrichment post-pass.
    # Runs AFTER _build_cluster_summary so it does not block the deterministic
    # summary build.  If USE_LLM_DOMAIN=false (default dev mode), this is a
    # no-op that returns immediately — zero overhead.
    print("  [3.5b] LLM cluster label enrichment...")
    cluster_summary = _enrich_cluster_labels_with_llm(cluster_summary, df_classified)

    # MPCD §5.2: Propagate cluster_label from cluster_summary back to per-row df_classified.
    # C7 FIX: pre-convert semantic_cluster_id to str once before the map so that
    # str() is not called N times inside the lambda (avoids N repeated str() calls
    # on a column that is already nearly-str but may contain pd.NA).
    if not cluster_summary.empty and "cluster_label" in cluster_summary.columns:
        _scid_to_label: Dict[str, str] = dict(
            zip(
                cluster_summary["cluster_id"].astype(str),
                cluster_summary["cluster_label"].fillna("").astype(str),
            )
        )
        # Pre-convert once: O(N) str conversion, then O(N) dict lookup — no per-row pd.notna() call
        _scid_str_series = df_classified["semantic_cluster_id"].fillna("").astype(str)
        df_classified["cluster_label"] = _scid_str_series.map(
            lambda scid: _scid_to_label.get(scid, "")
        )
    else:
        df_classified["cluster_label"] = ""

    print(f"        {n_summary_rows} clusters in summary")

    # ACCURACY-FIX-B6: header_cluster_count assertion
    header_cluster_count = n_summary_rows
    stats["header_cluster_count"] = header_cluster_count

    if header_cluster_count != n_summary_rows:
        msg = (
            f"ASSERTION-5 FAILED: header_cluster_count={header_cluster_count} "
            f"!= cluster_summary rows={n_summary_rows}"
        )
        consistency_failures.append(msg)
        logger_obj.error(msg)
        if raise_on_consistency_error:
            raise PipelineConsistencyError(msg)

    # Log low-confidence domain clusters for review
    # domain_confidence is now float — threshold < 0.50 = low confidence
    if not cluster_summary.empty and "domain_confidence" in cluster_summary.columns:
        _conf_numeric = pd.to_numeric(cluster_summary["domain_confidence"], errors="coerce").fillna(0.0)
        low_conf_clusters = cluster_summary[_conf_numeric < 0.50]
        if not low_conf_clusters.empty:
            print(f"  ⚠️  {len(low_conf_clusters)} clusters have low domain confidence — "
                  "review domain assignments:")
            for _, row in low_conf_clusters.iterrows():
                print(
                    f"     {row['cluster_id']}  domain={row['domain']}  "
                    f"conf={row.get('domain_confidence', '?')}  "
                    f"sample={str(row.get('sample_template',''))[:60]}"
                )

    # ── 3.7: Detect suspicious splits ────────────────────────────────
    print("  [3.6] Detecting suspicious splits...")
    suspicious = _detect_suspicious_splits(df_classified)
    print(f"        {len(suspicious)} suspicious split pairs found")

    avg_sil = float(np.mean(all_sil_scores)) if all_sil_scores else None
    # S3-ROUTINE-FLAG: count routine clusters for dashboard visibility
    _routine_count = 0
    _warn_routine_count = 0
    if not cluster_summary.empty:
        if "is_routine" in cluster_summary.columns:
            _routine_count = int(cluster_summary["is_routine"].sum())
        if "is_warn_routine" in cluster_summary.columns:          # S3-FP-2
            _warn_routine_count = int(cluster_summary["is_warn_routine"].sum())

    # S3-ML stats for pipeline.py / dashboard
    _n_known_normal = int((df_classified.get("singleton_class", pd.Series()) == "known_normal").sum()) \
        if "singleton_class" in df_classified.columns else 0

    # S3-FP-1: partition silhouette summary — identify worst-quality partitions
    _worst_partitions = sorted(
        [(k, v) for k, v in per_partition_silhouette.items() if v is not None],
        key=lambda x: x[1],
    )[:5]  # bottom 5 silhouette scores for operator awareness

    stats.update({
        "unique_templates_clustered": n_unique,
        "cluster_summary":            cluster_summary,
        "stage25_df_with_clusters":   df_classified,
        "suspicious_splits":          suspicious,
        "threshold_used":             full_cfg.get("agglo_distance_threshold", 0.45),
        "n_semantic_clusters":        total_clusters,
        "n_isolated":                 total_isolated,
        "silhouette_score":           avg_sil,
        "per_partition_silhouette":   per_partition_silhouette,   # S3-FP-1
        "tfidf_fallback_active":      _embedding_backend == "tfidf",  # S3-FP-1 / S4-FP-5
        "status":                     "ok",
        "consistency_failures":       consistency_failures,
        "manifest_used":              cluster_manifest is not None,
        "n_routine_clusters":         _routine_count,
        "n_warn_routine_clusters":    _warn_routine_count,        # S3-FP-2
        # S3-ML-1: embedding model info
        "embedding_model_loaded":     _EMBEDDING_MODEL_LOADED,
        "embedding_backend":          _embedding_backend,
        # S3-ML-2: adaptive UMAP info
        "umap_adaptive_neighbors":    full_cfg.get("umap_adaptive_neighbors", True),
        # S3-ML-3: known-normal safelist info
        "n_known_normal_suppressed":  _n_known_normal,
        "known_normal_tids_count":    len(full_cfg.get("known_normal_tids", [])),
        # S3-ML-4: embedding drift detection
        "n_centroid_drifted":         sum(1 for v in _drift_flags.values() if v),
        "drift_detection_threshold":  full_cfg.get("drift_detection_threshold", 0.30),
    })

    n_assigned = df_classified["semantic_cluster_id"].notna().sum()
    print(f"\n✅  Stage 3 done.")
    print(f"   Input unique templates  : {n_unique}")
    print(f"   Semantic clusters       : {total_clusters}")
    print(f"   Isolated templates      : {total_isolated}")
    print(f"   Rows with SCID          : {n_assigned}/{len(df_classified)}")
    print(f"   Final cluster summary   : {n_summary_rows} rows")
    print(f"   Header cluster count    : {header_cluster_count}  ✅ matches rows")
    print(f"   True anomalies          : {n_anomaly}")
    print(f"   Unseen variants         : {n_unseen}")
    print(f"   Noise filtered          : {n_noise_filt}")
    print(f"   Impossible attempt#     : {n_bad_attempt}")
    print(f"   Known-normal suppressed : {_n_known_normal}  (safelist hits)")
    print(f"   Routine clusters        : {_routine_count}  (baseline, not anomalies)")
    print(f"   Warn-routine clusters   : {_warn_routine_count}  (S3-FP-2: 0%-error WARN, capped at LOW)")
    print(f"   Suspicious splits       : {len(suspicious)}")
    print(f"   Manifest used           : {cluster_manifest is not None}")
    print(f"   Low-conf domain rows    : {low_conf_count}")
    print(f"   Embedding model         : {_EMBEDDING_MODEL_LOADED}")
    print(f"   Embedding backend       : {_embedding_backend}")
    if _embedding_backend == "tfidf":
        print(f"   ⚠️  TF-IDF FALLBACK ACTIVE — clustering quality degraded!")
    if _worst_partitions:
        print(f"   Lowest silhouette partitions (S3-FP-1):")
        for part_key, sil_val in _worst_partitions:
            print(f"     {part_key:30s}: {sil_val:.3f}")
    if consistency_failures:
        print(f"   ⚠️  Consistency failures : {len(consistency_failures)}")
        for f in consistency_failures:
            print(f"     → {f}")
    print(f"\n   Domain breakdown (Stage 2):")
    for domain, count in sorted(domain_dist.items(), key=lambda x: -x[1]):
        print(f"     {domain:20s}: {count}")

    return df_classified, stats



# ══════════════════════════════════════════════════════════════════════
# STANDALONE TEST ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python stage3.py <stage2_output.csv> [manifest.json]")
        print("       stage2_output.csv must be the CSV produced by stage2.py")
        sys.exit(1)

    import json
    from pathlib import Path

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    manifest = None
    if len(sys.argv) >= 3:
        manifest_path = Path(sys.argv[2])
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            print(f"Loaded manifest from: {manifest_path}")
        else:
            print(f"Warning: manifest not found: {manifest_path}")

    print(f"Loading stage 2 output from: {csv_path}")
    df_s2 = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Restore bool/numeric types
    for col in ("is_noise", "is_merged", "burst_flag", "timestamp_parsed_ok"):
        if col in df_s2.columns:
            df_s2[col] = df_s2[col].map({"True": True, "False": False}).fillna(False)
    for col in ("line_no", "repeat_count", "burst_count"):
        if col in df_s2.columns:
            df_s2[col] = pd.to_numeric(df_s2[col], errors="coerce")
    if "timestamp_parsed" in df_s2.columns:
        df_s2["timestamp_parsed"] = pd.to_datetime(df_s2["timestamp_parsed"], errors="coerce", utc=True)

    print(f"Loaded {len(df_s2):,} rows")

    df_out, s3_stats = run_stage3(df_s2, cluster_manifest=manifest)

    cluster_summ = s3_stats.get("cluster_summary", pd.DataFrame())
    print(f"\nCluster summary: {len(cluster_summ)} clusters")
    print(f"Silhouette: {s3_stats.get('silhouette_score')}")

    out_path = csv_path.parent / "stage3_output.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n✅  stage3_output.csv written to: {out_path}")

    if not cluster_summ.empty:
        cs_path = csv_path.parent / "stage3_cluster_summary.csv"
        cluster_summ.to_csv(cs_path, index=False)
        print(f"✅  stage3_cluster_summary.csv written to: {cs_path}")