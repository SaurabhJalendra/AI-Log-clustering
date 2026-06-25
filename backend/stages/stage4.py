"""
backend/stages/stage4.py
========================
STAGE 4 — ANOMALY SCORING

Public API
----------
    from stages.stage4 import run_stage4

    results = run_stage4(df, cluster_summary_df=cluster_summary)

    # Access results via dict keys:
    anomaly_df = results['anomaly_df']     # signal clusters, scored & labelled
    routine_df = results['routine_df']     # suppressed routine telemetry

Returns
-------
    dict with keys:
        anomaly_df, routine_df, freq_df, sev_df, temporal_df,
        trend_df, cascade_df, source_df, verification_table,
        col_map, config_used, sample_msg_map, ml_stats

Stage 4 does NOT call Stage 1, 2, or 3 — it receives the classified
DataFrame as a parameter.  Stages never call each other directly;
pipeline.py is the only place that chains them.

Fixes implemented (from notebook):
    FIX-A   Percentile-normalised error_volume
    FIX-B   WARN singleton guard
    FIX-C   JOIN cluster_label from cluster_summary
    FIX-E   Trend direction analysis
    FIX-F   Cross-service cascade detection
    FIX-G   Baseline comparison (GT file, optional)
    FIX-I   Stuck-state escalation with HTTP 304 guard
    FIX-J   distinct_msg_count per cluster
    FIX-K   Explicit denominator for pct_of_total
    FIX-L   Timestamp tz-aware/naive coercion guard
    FIX-M   ACC-2 routine suppression label guard
    FIX-N   EID resolution hardening
    FIX-O   Empty / single-row DataFrame edge-case guards
    FIX-2   Context-aware 401 cap
    ACC-1   Volume-severity score
    ACC-2   Routine telemetry filter
    ACC-3   Stuck-state priority escalation
    ACC-4   Success-signal severity corrector
    ACC-5   Severity-context contradiction detector
    ACC-6   Rarity score floor for ultra-high-volume clusters
    A12     Epsilon-shifted pd.cut bins for boundary correctness

ML upgrades (S4-ML):
    S4-ML-1  Isolation Forest per-service unsupervised anomaly scoring
    S4-ML-2  Autoencoder reconstruction-error scoring
    S4-ML-3  Ensemble fusion of IF + AE scores with formula baseline
    S4-ML-4  Post-deployment grace period (new-template clusters only)
    S4-ML-5  Incremental autoencoder update on confirmed-normal clusters
    S4-ML-6  anomaly_source and ml_confidence columns on every row
    S4-ML-7  Model persistence (cache IF + AE per service to disk)
    S4-ML-8  Extended normal pool + empty normal pool warning

Spec changes (from Master Upgrade Specification):
    Issue 2  Empty normal pool: union mask + WARNING log
    Issue 3  Per-service contamination estimate (not fixed 0.05)
    Issue 4  import json/pickle/hashlib at module level (not in loops)
    Issue 5  Post-ensemble percentile recalibration of label thresholds
    Issue 6  Grace period multiplier applied to new-template-ID clusters only
    Issue 7  Incremental AE update tuning (5 steps, 0.2x LR)
    Issue 9  verification_table alias bug fixed (anomaly_df.copy())
    Feature  3 new feature columns: domain_confidence_score,
             template_length_norm, hour_of_day_norm (ml_feature_version=2)
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════
# S4-ML: ISOLATION FOREST + AUTOENCODER ENSEMBLE ANOMALY SCORER
# ══════════════════════════════════════════════════════════════════════
#
# Architecture
# ─────────────
#  Per-service models are trained on the CURRENT run's confirmed-normal
#  clusters (high count, stable, INFO/DEBUG-dominant).  Each service
#  gets its own IF and AE so cross-service volume differences don't
#  distort the outlier geometry.
#
#  Scoring pipeline (per cluster):
#    1. Build a feature vector from formula sub-scores
#       [rarity, severity, burstiness, spread, volume_severity]
#    2. IF predicts an outlier score   (0 = normal, 1 = anomalous)
#    3. AE computes reconstruction error (normalised 0–1)
#    4. Ensemble = w_if * IF + w_ae * AE + w_formula * formula
#    5. ensemble_disagreement flagged when |IF - AE| > threshold
#
# Failure-prevention measures (complete list)
# ────────────────────────────────────────────
#  FP-1  Training data contamination guard:
#        Only clusters classified as is_routine=True OR in the top-N by
#        count AND not flagged as true_anomaly are used as training data.
#        This prevents known anomalies from teaching the model that
#        anomalous patterns are normal.
#
#  FP-2  Minimum training size guard:
#        IF requires >= if_min_rows (default 20) training rows.
#        AE requires >= ae_min_rows (default 50) training rows.
#        Services below these floors fall back to formula score only.
#        Prevents overfitting on 3–5 rows that flags everything else.
#
#  FP-3  Post-deployment grace period (S4-ML-4):
#        If > post_deploy_new_tid_threshold fraction of template_ids
#        in this run are new (not in known_normal_tids from stage3),
#        ML scores are multiplied by post_deploy_score_multiplier (0.75)
#        and anomaly_source is set to "post_deploy_caution".
#        Prevents the false-positive flood that follows a big deploy
#        where all new log lines look anomalous to the trained model.
#
#  FP-4  Ensemble disagreement as a meta-signal (S4-ML-6):
#        When |IF_score - AE_score| > ensemble_disagreement_threshold,
#        ensemble_disagreement=True is written to the row.  Stage 5 and
#        the dashboard use this to surface review-recommended clusters.
#
#  FP-5  Graceful degradation:
#        sklearn not installed → formula score only, no crash.
#        PyTorch not installed → IF only (no AE), formula in ensemble.
#        Any per-service training exception → formula score for that service.
#
#  FP-6  Incremental AE update (S4-ML-5):
#        After scoring, the top-N highest-count clusters (clearly normal)
#        are used to run ae_incremental_steps gradient steps on the AE,
#        keeping it current without full retraining.  Only runs if AE
#        training succeeded for that service.
# ══════════════════════════════════════════════════════════════════════

import logging as _logging
_s4_logger = _logging.getLogger("stage4.ml")

# Feature columns used by both IF and AE — order must be stable.
# ml_feature_version=3: added semantic_embedding_score (S4-3/S4-4).
# Delete persisted model files when upgrading from v2.
_ML_FEATURE_COLS = [
    "rarity_score",
    "severity_score",
    "burstiness_score",
    "spread_score",
    "volume_severity_score",
    # v2 additions ────────────────────────────────
    "domain_confidence_score",   # float 0-1 from domain_confidence column
    "template_length_norm",      # normalised character length of event_template
    "hour_of_day_norm",          # hour of first_seen / 23
    # v3 additions (S4-3) ─────────────────────────
    "semantic_embedding_score",  # cosine similarity to known-anomalous log patterns
]

# Increment this when _ML_FEATURE_COLS changes so cached models with the
# wrong input_dim are rejected at load time.  Matches ml_feature_version in config.
_ML_FEATURE_VERSION = 3  # S4-4: was 2


def _check_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _build_feature_matrix(df: "pd.DataFrame") -> "np.ndarray":
    """
    Build a float32 feature matrix from the ML feature columns.
    Missing columns are filled with 0.0.  All values clipped to [0, 1].

    v2 features (domain_confidence_score, template_length_norm,
    hour_of_day_norm) are derived inline if their source columns are
    present, so callers don't have to pre-compute them.

    v3 (semantic_embedding_score): when the column is absent or was set to
    NaN by _compute_semantic_embedding_scores on failure, the column is
    imputed with the column mean across present rows.  If ALL rows are NaN
    (total embedding failure), the column is dropped entirely from the matrix
    rather than contributing a constant 0.5 that adds noise without signal.
    Dropping the column does NOT invalidate cached models because the feature
    version fingerprint (_ML_FEATURE_VERSION) is unchanged — the column is
    still listed in _ML_FEATURE_COLS.  The imputation/drop only affects the
    in-memory numpy matrix; nothing is written back to the DataFrame.
    """
    # Derive v2 feature columns if not already present
    _df = df  # work on original; only copy if we need to add columns
    _added = {}

    # domain_confidence_score — float 0-1 from domain_confidence column
    if "domain_confidence_score" not in _df.columns:
        if "domain_confidence" in _df.columns:
            _added["domain_confidence_score"] = (
                pd.to_numeric(_df["domain_confidence"], errors="coerce")
                .fillna(0.0).clip(0.0, 1.0).values
            )
        else:
            # S4-4B: domain_confidence should always be present after Stage 2 consolidation.
            # A zero-fill here means Stage 2 domain classification did not run correctly.
            _s4_logger.error(
                "S4: domain_confidence column missing from input — "
                "Stage 2 domain classification may not have run correctly. "
                "domain_confidence_score feature will be zeroed."
            )
            _added["domain_confidence_score"] = np.zeros(len(_df), dtype=np.float32)

    # template_length_norm — normalised character length of event_template
    if "template_length_norm" not in _df.columns:
        if "event_template" in _df.columns:
            lengths = _df["event_template"].fillna("").str.len().astype(float)
            max_len = float(lengths.max()) if lengths.max() > 0 else 1.0
            _added["template_length_norm"] = (lengths / max_len).clip(0.0, 1.0).values
        else:
            _added["template_length_norm"] = np.zeros(len(_df), dtype=np.float32)

    # hour_of_day_norm — hour of first_seen divided by 23
    if "hour_of_day_norm" not in _df.columns:
        for ts_candidate in ["first_seen", "_ts", "timestamp_parsed"]:
            if ts_candidate in _df.columns:
                ts = pd.to_datetime(_df[ts_candidate], errors="coerce", utc=True)
                hours = ts.dt.hour.fillna(0.0).astype(float)
                _added["hour_of_day_norm"] = (hours / 23.0).clip(0.0, 1.0).values
                break
        if "hour_of_day_norm" not in _added:
            _added["hour_of_day_norm"] = np.zeros(len(_df), dtype=np.float32)

    # semantic_embedding_score (v3) — computed by _compute_semantic_embedding_scores
    # before _build_feature_matrix is called.
    #
    # WARN fix (MPCD §3.4): a constant 0.5 (or 0.0) across all rows adds noise
    # without contributing signal when embedding fails.  Strategy:
    #   1. If the column is present and has at least one finite value, impute
    #      NaNs with the column mean so the distribution is preserved.
    #   2. If the column is entirely NaN (total failure), skip it entirely —
    #      the loop below will not include it, effectively dropping it for
    #      this run without crashing anything downstream.
    #   3. If the column is absent, treat it the same as entirely NaN (skip).
    #
    # Time complexity: O(N) for nanmean over N rows — negligible.
    # Space complexity: O(N) for the imputed float32 array — same as before.
    _sem_col_name = "semantic_embedding_score"
    if _sem_col_name not in _df.columns:
        # Column absent — leave _added empty for this key; loop will skip it.
        pass
    else:
        _raw_sem = pd.to_numeric(_df[_sem_col_name], errors="coerce").values.astype(np.float64)
        _finite_mask = np.isfinite(_raw_sem)
        if _finite_mask.any():
            # At least some valid scores: impute NaNs with column mean.
            _col_mean = float(np.nanmean(_raw_sem))
            _raw_sem[~_finite_mask] = _col_mean
            _added[_sem_col_name] = _raw_sem.clip(0.0, 1.0).astype(np.float32)
        else:
            # All NaN — total embedding failure; drop column for this run.
            # Log at debug level; the caller already logged a warning.
            _s4_logger.debug(
                "_build_feature_matrix: semantic_embedding_score is entirely NaN "
                "— column excluded from feature matrix for this run."
            )
            # Do NOT add to _added; the loop below will skip this feature.

    cols = []
    for c in _ML_FEATURE_COLS:
        if c in _df.columns and c != _sem_col_name:
            # All non-semantic columns: standard coerce + fillna(0.0) + clip
            cols.append(
                pd.to_numeric(_df[c], errors="coerce")
                .fillna(0.0).clip(0.0, 1.0).values.astype(np.float32)
            )
        elif c == _sem_col_name:
            # Semantic column: use the imputed array if available, skip if not
            if _sem_col_name in _added:
                cols.append(_added[_sem_col_name])
            # else: column dropped — do not append a zero column
        elif c in _added:
            cols.append(_added[c].astype(np.float32))
        else:
            cols.append(np.zeros(len(_df), dtype=np.float32))

    if not cols:
        # Extreme edge case: empty feature set.  Return a zero matrix with
        # shape (N, 1) so downstream numpy operations don't crash on empty.
        return np.zeros((len(_df), 1), dtype=np.float32)
    return np.column_stack(cols).astype(np.float32)


def _compute_semantic_embedding_scores(df: "pd.DataFrame", cfg: dict) -> "pd.Series":
    """
    S4-3: Compute cosine similarity between each cluster's sample template
    and the known_anomaly_patterns library.  Returns a Series of scores in
    [0, 1] where 1 = highly similar to a known-anomalous pattern.

    Uses character n-gram TF-IDF similarity (analyzer='char_wb', ngram_range
    (3,5)) so it works without a loaded embedding model — no network required.
    Falls back gracefully on any import or computation error.

    The score is added as 'semantic_embedding_score' so _build_feature_matrix
    can pick it up as a v3 ML feature column.

    WARN fix (MPCD §3.4): on any failure path, return NaN instead of a
    constant 0.0 or 0.5.  A constant value across all N rows contributes
    zero variance to the IF/AE feature matrix — it shifts every score by
    the same amount without separating anomalous from normal rows, which
    degrades model quality more than simply omitting the column.
    _build_feature_matrix detects an all-NaN column and drops it for that
    run rather than propagating the noise.

    Intentional 0.0 returns (no patterns configured, embedding disabled,
    no template column) are NOT changed to NaN because in those cases
    the 0.0 is a deliberate "no signal" sentinel, not a failure signal.
    Only the exception path is changed.

    Time complexity  : O(N·P·F) where N = cluster count, P = pattern count,
                       F = TF-IDF features (capped at max_features=2000).
                       TfidfVectorizer.fit_transform is O((N+P)·F).
                       cosine_similarity is O(N·P·F).  Total: O(N·P·F).
    Space complexity : O((N+P)·F) for the sparse TF-IDF matrix.
    Both are unchanged from the original — this fix adds only a branch.
    """
    patterns = cfg.get("known_anomaly_patterns", [])
    if not patterns or not cfg.get("semantic_embedding_enabled", True):
        # Deliberate no-signal path — 0.0 is correct here.
        return pd.Series(0.0, index=df.index)

    # Resolve which column holds the template text
    template_col = None
    for candidate in ("sample_template", "sample_message", "event_template",
                      "normalized_message"):
        if candidate in df.columns:
            template_col = candidate
            break
    if template_col is None:
        # No text column available — 0.0 is correct (no information).
        return pd.Series(0.0, index=df.index)

    templates = df[template_col].fillna("").astype(str).tolist()
    if not templates:
        return pd.Series(0.0, index=df.index)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        all_texts = templates + patterns
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=2000
        )
        X = vec.fit_transform(all_texts)
        template_vecs = X[: len(templates)]
        pattern_vecs  = X[len(templates) :]
        sims   = cosine_similarity(template_vecs, pattern_vecs)
        scores = sims.max(axis=1).clip(0.0, 1.0)
        return pd.Series(scores.astype(float), index=df.index)
    except Exception as exc:
        # WARN fix: return NaN, not 0.0.  A constant 0.0 across all rows adds
        # noise without signal — worse than an absent column.
        # _build_feature_matrix will detect the all-NaN column and drop it.
        _s4_logger.warning(
            "S4-3: semantic_embedding_score failed (%s) — returning NaN series. "
            "_build_feature_matrix will exclude the column for this run.", exc
        )
        return pd.Series(np.nan, index=df.index)


def _estimate_contamination(formula_scores: "np.ndarray", cfg: dict) -> float:
    """
    Issue 3: Estimate per-service contamination from the formula score
    distribution — fraction of clusters above threshold_medium —
    clipped to [0.01, 0.40].  Replaces fixed contamination=0.05.
    """
    threshold_medium = cfg.get("threshold_medium", 0.35)
    if len(formula_scores) == 0:
        return 0.05
    above = float((formula_scores > threshold_medium).mean())
    return float(np.clip(above, 0.01, 0.40))


# ── S4-ML-1: Isolation Forest per service ────────────────────────────

def _train_isolation_forest(X_normal: "np.ndarray", cfg: dict,
                            contamination: float = None):
    """
    Train an Isolation Forest on confirmed-normal feature rows.
    Returns the fitted model or None on failure.
    contamination is per-service estimated (Issue 3); falls back to
    cfg['if_contamination'] if not provided.
    """
    if not _check_sklearn():
        return None
    if len(X_normal) < cfg.get("if_min_rows", 15):  # S4-2: was 20
        return None
    if contamination is None:
        contamination = cfg.get("if_contamination", "auto")
    try:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(
            n_estimators=cfg.get("if_n_estimators", 100),
            contamination=contamination,
            random_state=cfg.get("if_random_state", 42),
            n_jobs=-1,
        )
        model.fit(X_normal)
        return model
    except Exception as exc:
        _s4_logger.warning("S4-ML-1: IsolationForest training failed (%s)", exc)
        return None


def _score_isolation_forest(model, X: "np.ndarray") -> "np.ndarray":
    """
    Score rows with a fitted IF.  Returns anomaly scores in [0, 1]
    where 1 = most anomalous.  sklearn's decision_function returns
    negative = anomalous, so we invert and normalise.
    """
    try:
        raw = model.decision_function(X)           # lower = more anomalous
        scores = 1.0 - (raw - raw.min()) / (raw.ptp() + 1e-9)
        return scores.clip(0.0, 1.0).astype(np.float32)
    except Exception as exc:
        _s4_logger.warning("S4-ML-1: IF scoring failed (%s)", exc)
        return np.full(len(X), 0.5, dtype=np.float32)


# ── S4-ML-2: Autoencoder per service ─────────────────────────────────

class _SimpleAutoencoder:
    """
    Minimal numpy/scipy autoencoder using two linear layers with ReLU.
    Falls back to this when PyTorch is not available.
    Uses matrix operations only — no external ML library required.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        rng = np.random.default_rng(42)
        scale_h = np.sqrt(2.0 / input_dim)
        scale_l = np.sqrt(2.0 / hidden_dim)
        scale_d = np.sqrt(2.0 / latent_dim)
        scale_o = np.sqrt(2.0 / hidden_dim)
        # Encoder
        self.W1 = rng.normal(0, scale_h, (input_dim,  hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, scale_l, (hidden_dim, latent_dim)).astype(np.float32)
        self.b2 = np.zeros(latent_dim, dtype=np.float32)
        # Decoder
        self.W3 = rng.normal(0, scale_d, (latent_dim, hidden_dim)).astype(np.float32)
        self.b3 = np.zeros(hidden_dim, dtype=np.float32)
        self.W4 = rng.normal(0, scale_o, (hidden_dim, input_dim)).astype(np.float32)
        self.b4 = np.zeros(input_dim,  dtype=np.float32)
        self.lr = 1e-3

    def _relu(self, x):
        return np.maximum(0.0, x)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x.clip(-30, 30)))

    def forward(self, X):
        h1    = self._relu(X  @ self.W1 + self.b1)
        lat   = self._relu(h1 @ self.W2 + self.b2)
        h3    = self._relu(lat @ self.W3 + self.b3)
        out   = self._sigmoid(h3 @ self.W4 + self.b4)
        return out

    def reconstruction_error(self, X):
        out = self.forward(X)
        return np.mean((X - out) ** 2, axis=1).astype(np.float32)

    def train_step(self, X):
        """Single gradient-descent step (MSE loss, manual backprop)."""
        # Forward
        h1  = self._relu(X @ self.W1 + self.b1)
        lat = self._relu(h1 @ self.W2 + self.b2)
        h3  = self._relu(lat @ self.W3 + self.b3)
        out = self._sigmoid(h3 @ self.W4 + self.b4)

        # Backward (MSE + sigmoid)
        dout  = 2.0 * (out - X) / len(X) * out * (1.0 - out)
        dW4   = h3.T  @ dout;   db4 = dout.sum(0)
        dh3   = dout  @ self.W4.T * (h3 > 0)
        dW3   = lat.T @ dh3;    db3 = dh3.sum(0)
        dlat  = dh3   @ self.W3.T * (lat > 0)
        dW2   = h1.T  @ dlat;   db2 = dlat.sum(0)
        dh1   = dlat  @ self.W2.T * (h1 > 0)
        dW1   = X.T   @ dh1;    db1 = dh1.sum(0)

        for p, g in [(self.W1, dW1), (self.b1, db1),
                     (self.W2, dW2), (self.b2, db2),
                     (self.W3, dW3), (self.b3, db3),
                     (self.W4, dW4), (self.b4, db4)]:
            p -= self.lr * g


def _train_autoencoder(X_normal: "np.ndarray", cfg: dict):
    """
    Train an autoencoder on confirmed-normal feature rows.
    Tries PyTorch first; falls back to _SimpleAutoencoder.
    Returns (model, 'torch'|'numpy') or (None, None) on failure.
    """
    min_rows = cfg.get("ae_min_rows", 30)  # S4-2: was 50
    if len(X_normal) < min_rows:
        return None, None

    input_dim  = X_normal.shape[1]
    hidden_dim = cfg.get("ae_hidden_dim", 32)
    latent_dim = cfg.get("ae_latent_dim", 8)
    epochs     = cfg.get("ae_epochs", 30)
    lr         = cfg.get("ae_lr", 1e-3)
    batch_size = cfg.get("ae_batch_size", 32)

    if _check_torch():
        try:
            import torch
            import torch.nn as nn

            class _TorchAE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.enc = nn.Sequential(
                        nn.Linear(input_dim,  hidden_dim), nn.ReLU(),
                        nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
                    )
                    self.dec = nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                        nn.Linear(hidden_dim, input_dim),  nn.Sigmoid(),
                    )
                def forward(self, x):
                    return self.dec(self.enc(x))

            model = _TorchAE()
            opt   = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            X_t   = torch.from_numpy(X_normal)

            model.train()
            for _ in range(epochs):
                idx   = np.random.permutation(len(X_normal))
                for b in range(0, len(idx), batch_size):
                    xb  = X_t[idx[b: b + batch_size]]
                    opt.zero_grad()
                    loss_fn(model(xb), xb).backward()
                    opt.step()
            model.eval()
            return model, "torch"
        except Exception as exc:
            _s4_logger.warning(
                "S4-ML-2: PyTorch AE training failed (%s) — using numpy AE.", exc
            )

    # Numpy fallback autoencoder
    try:
        model = _SimpleAutoencoder(input_dim, hidden_dim, latent_dim)
        model.lr = lr
        for _ in range(epochs):
            idx = np.random.permutation(len(X_normal))
            for b in range(0, len(idx), batch_size):
                xb = X_normal[idx[b: b + batch_size]]
                model.train_step(xb)
        return model, "numpy"
    except Exception as exc:
        _s4_logger.warning("S4-ML-2: numpy AE training also failed (%s)", exc)
        return None, None


def _score_autoencoder(model, model_type: str, X: "np.ndarray") -> "np.ndarray":
    """
    Score rows with a fitted AE.  Returns reconstruction errors normalised
    to [0, 1] where 1 = most anomalous.
    """
    try:
        if model_type == "torch":
            import torch
            with torch.no_grad():
                X_t  = torch.from_numpy(X)
                recon = model(X_t).numpy()
            errors = np.mean((X - recon) ** 2, axis=1).astype(np.float32)
        else:
            errors = model.reconstruction_error(X)

        e_min, e_max = errors.min(), errors.max()
        if e_max - e_min < 1e-9:
            return np.zeros(len(X), dtype=np.float32)
        return ((errors - e_min) / (e_max - e_min)).clip(0.0, 1.0)
    except Exception as exc:
        _s4_logger.warning("S4-ML-2: AE scoring failed (%s)", exc)
        return np.full(len(X), 0.5, dtype=np.float32)


def _incremental_ae_update(model, model_type: str, X_new_normal: "np.ndarray",
                           cfg: dict) -> None:
    """
    S4-ML-5: Run a few gradient steps on the AE using the top-N confirmed-
    normal clusters from the current run.  Mutates model in-place.
    This keeps the AE current after a deployment without full retraining.

    Issue 7: ae_incremental_steps now defaults to 5 (was 2) and the
    torch-path LR multiplier is 0.2x (was 0.1x).
    """
    if model is None or len(X_new_normal) == 0:
        return
    steps = cfg.get("ae_incremental_steps", 5)   # Issue 7: was 2
    try:
        if model_type == "torch":
            import torch
            import torch.nn as nn
            opt     = torch.optim.Adam(model.parameters(),
                                       lr=cfg.get("ae_lr", 1e-3) * 0.2)   # Issue 7: was 0.1
            loss_fn = nn.MSELoss()
            X_t     = torch.from_numpy(X_new_normal)
            model.train()
            for _ in range(steps):
                opt.zero_grad()
                loss_fn(model(X_t), X_t).backward()
                opt.step()
            model.eval()
        else:
            for _ in range(steps):
                model.train_step(X_new_normal)
    except Exception as exc:
        _s4_logger.debug("S4-ML-5: incremental AE update failed (%s)", exc)



# ── S4-ML-7: Model persistence ────────────────────────────────────────

def _model_cache_dir(cfg: dict) -> Optional[Path]:
    """
    S4-ML-7: Return the model cache directory as a Path, creating it if
    needed.  Returns None gracefully when:
      - ml_model_cache_dir is not set (caching disabled)
      - directory creation fails
    """
    raw = cfg.get("ml_model_cache_dir", None)
    if not raw:
        return None
    try:
        p = Path(raw)
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as exc:
        _s4_logger.warning("S4-ML-7: cannot create cache dir '%s' (%s) — caching disabled", raw, exc)
        return None


def _normal_fingerprint(X_normal: "np.ndarray") -> str:
    """
    S4-ML-7: Produce a 16-character SHA-256 fingerprint of the confirmed-
    normal feature matrix.  Used to detect when training data has changed
    materially so cached models are invalidated.
    """
    shape_str = str(X_normal.shape)
    # Sample a digest of the actual values for content sensitivity
    if X_normal.size > 0:
        sample = X_normal.flat[::max(1, X_normal.size // 256)]
        content = shape_str + sample.tobytes().hex()
    else:
        content = shape_str
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _load_cached_models(service: str, cache_dir: Path,
                        fingerprint: str) -> Tuple:
    """
    S4-ML-7: Load per-service cached IF and AE models from disk.
    Returns (if_model, ae_model, ae_type) or (None, None, None) on any
    miss, fingerprint mismatch, or unpickling error.
    """
    meta_path = cache_dir / f"{service}_meta.json"
    if_path   = cache_dir / f"{service}_if.pkl"
    ae_path   = cache_dir / f"{service}_ae.pkl"

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("fingerprint") != fingerprint:
            _s4_logger.info("S4-ML-7: cache miss for '%s' (fingerprint mismatch)", service)
            return None, None, None
        if meta.get("ml_feature_version") != _ML_FEATURE_VERSION:
            _s4_logger.info(
                "S4-ML-7: cache miss for '%s' (feature version %s vs current %s)",
                service, meta.get("ml_feature_version"), _ML_FEATURE_VERSION,
            )
            return None, None, None

        if_model  = None
        ae_model  = None
        ae_type   = meta.get("ae_type", "numpy")

        if if_path.exists():
            with open(if_path, "rb") as f:
                if_model = pickle.load(f)
        if ae_path.exists():
            with open(ae_path, "rb") as f:
                ae_model = pickle.load(f)

        if if_model is not None or ae_model is not None:
            _s4_logger.info("S4-ML-7: cache hit for service='%s'", service)
            return if_model, ae_model, ae_type

    except Exception as exc:
        _s4_logger.debug("S4-ML-7: cache load failed for '%s' (%s)", service, exc)
    return None, None, None


def _save_cached_models(service: str, cache_dir: Path,
                        if_model, ae_model, ae_type: Optional[str],
                        fingerprint: str) -> None:
    """
    S4-ML-7: Persist per-service IF and AE models to disk alongside a
    metadata JSON containing the fingerprint and feature version.
    Called after initial training AND after incremental AE update.
    """
    try:
        meta = {
            "service"            : service,
            "fingerprint"        : fingerprint,
            "ml_feature_version" : _ML_FEATURE_VERSION,
            "ae_type"            : ae_type or "none",
        }
        meta_path = cache_dir / f"{service}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        if if_model is not None:
            if_path = cache_dir / f"{service}_if.pkl"
            with open(if_path, "wb") as f:
                pickle.dump(if_model, f)

        if ae_model is not None:
            ae_path = cache_dir / f"{service}_ae.pkl"
            with open(ae_path, "wb") as f:
                pickle.dump(ae_model, f)

        _s4_logger.debug("S4-ML-7: saved models for service='%s'", service)
    except Exception as exc:
        _s4_logger.warning("S4-ML-7: model save failed for '%s' (%s)", service, exc)


# ── S4-ML-3: Per-service ensemble scorer ─────────────────────────────

def compute_ml_anomaly_scores(
    anomaly_df: "pd.DataFrame",
    cfg: dict,
    known_normal_tids: Optional[List[str]] = None,
    normal_pool_df: Optional["pd.DataFrame"] = None,
) -> "pd.DataFrame":
    """
    S4-ML-3: Train per-service IF + AE on confirmed-normal clusters,
    score all clusters, fuse with formula score, and write six new columns:

        ml_if_score, ml_ae_score, ml_ensemble_score,
        ensemble_disagreement, anomaly_source, ml_confidence

    anomaly_source values:

    normal_pool_df (S4-ML-NORMAL-FIX): optional DataFrame of confirmed-routine
    rows (the pre-split is_routine=True clusters) to augment the per-service
    normal training pool.  When provided, these rows are appended to svc_normal
    before training IF/AE, preventing X_normal-EMPTY failures on services whose
    anomaly_df rows are all signal-level clusters.
    
        "ae_if_ensemble"       — both IF and AE scored this cluster
        "ml_if_only"           — only IF available
        "ml_ae_only"           — only AE available
        "formula_fallback"     — neither model trained (service too small)
        "post_deploy_caution"  — ML scores dampened (grace period, new tids only)

    Spec changes implemented:
        Issue 2  Extended normal pool (strict | score-based) + WARNING when empty
        Issue 3  Per-service contamination estimate (not fixed 0.05)
        Issue 6  Grace period dampening applied to new-template-ID rows only
        Issue 7  Incremental AE: 5 steps, 0.2x LR (in _incremental_ae_update)
        S4-ML-7  Model persistence: load/save per-service IF + AE
    """
    if anomaly_df.empty:
        return anomaly_df

    df = anomaly_df.copy()

    # Initialise output columns
    df["ml_if_score"]           = 0.5
    df["ml_ae_score"]           = 0.5
    df["ml_ensemble_score"]     = df["anomaly_score"].clip(0.0, 1.0)
    df["ensemble_disagreement"] = False
    df["anomaly_source"]        = "formula_fallback"
    df["ml_confidence"]         = 0.0

    # ── FP-3 / Issue 6: Post-deployment grace period detection ───────
    # Identify new template IDs once; apply multiplier per-cluster later.
    is_post_deploy = False
    new_tid_set: set = set()
    if cfg.get("post_deploy_grace_enabled", True) and known_normal_tids:
        known_set = set(str(t) for t in known_normal_tids)
        tid_col   = "template_id" if "template_id" in df.columns else None
        eid_col_  = None
        for c in ["event_id", "semantic_cluster_id"]:
            if c in df.columns:
                eid_col_ = c
                break
        ref_col = tid_col or eid_col_
        if ref_col:
            all_tids = set(df[ref_col].dropna().astype(str))
            new_tids = all_tids - known_set
            frac_new = len(new_tids) / max(len(all_tids), 1)
            thresh   = cfg.get("post_deploy_new_tid_threshold", 0.30)
            if frac_new > thresh:
                is_post_deploy = True
                new_tid_set    = new_tids   # Issue 6: only these rows get dampened
                _s4_logger.info(
                    "S4-ML-4: post-deploy grace period active — "
                    "%.1f%% of template_ids are new (threshold=%.0f%%)",
                    frac_new * 100, thresh * 100,
                )

    # ── Resolve service column ────────────────────────────────────────
    svc_col = None
    for c in ["service", "top_source"]:
        if c in df.columns:
            svc_col = c
            break
    if svc_col is None:
        df["_svc_tmp"] = "all"
        svc_col = "_svc_tmp"

    # Resolve template/event ID column for per-row grace-period check
    _ref_col = None
    for c in ["template_id", "event_id", "semantic_cluster_id"]:
        if c in df.columns:
            _ref_col = c
            break

    # ── FP-1 + Issue 2: Confirmed-normal mask ────────────────────────
    # Union of strict mask (LOW+INFO/DEBUG) AND score-based mask
    # (formula_score < 0.35 + not known anomaly singleton).
    _anomaly_labels  = df.get("anomaly_label", pd.Series("LOW", index=df.index))
    _singleton_class = df.get("singleton_class", pd.Series("", index=df.index))
    _dom_sev         = df.get("dominant_severity", pd.Series("INFO", index=df.index))
    _formula_scores  = df.get("anomaly_score", pd.Series(0.0, index=df.index)).fillna(0.0)

    strict_normal_mask = (
        _anomaly_labels.isin(["LOW"])
        & ~_singleton_class.isin(["true_anomaly", "impossible_attempt_count"])
        & _dom_sev.isin(["INFO", "DEBUG"])
    )
    score_normal_mask = (
        (_formula_scores < 0.35)
        & ~_singleton_class.isin(["true_anomaly", "impossible_attempt_count"])
    )
    normal_mask = strict_normal_mask | score_normal_mask

    # S4-9: Exclude clusters whose sample message matches a confirmed-anomaly
    # template from the previous Stage 5 run, so they never pollute the normal pool.
    _confirmed_templates = cfg.get("_confirmed_templates", set())
    if _confirmed_templates:
        _sample_col = None
        for _sc in ("sample_message", "sample_template", "event_template",
                    "normalized_message"):
            if _sc in df.columns:
                _sample_col = _sc
                break
        if _sample_col:
            _confirmed_mask = df[_sample_col].apply(
                lambda m: any(t in str(m) for t in _confirmed_templates)
            )
            normal_mask &= ~_confirmed_mask
            _n_excluded = int(_confirmed_mask.sum())
            if _n_excluded:
                _s4_logger.info(
                    "S4-9: excluded %d confirmed-anomaly rows from normal pool",
                    _n_excluded,
                )

    # S4-ML-7: resolve cache directory once
    cache_dir = _model_cache_dir(cfg)

    # S4-3: Compute semantic embedding scores once for all clusters.
    # Written back to df so _build_feature_matrix picks it up as a v3 feature.
    if cfg.get("semantic_embedding_enabled", True):
        df["semantic_embedding_score"] = _compute_semantic_embedding_scores(df, cfg)
        # NaN-safe logging: _compute_semantic_embedding_scores returns an all-NaN
        # Series on failure.  pandas mean()/max() on an all-NaN Series returns NaN
        # (with a RuntimeWarning); float(NaN) is valid Python but misleading in logs.
        # Use nanmean/nanmax so we always log a meaningful number, and emit a
        # separate WARNING when all values are NaN so operators notice the failure.
        _sem_vals = df["semantic_embedding_score"].values.astype(np.float64)
        _sem_finite = _sem_vals[np.isfinite(_sem_vals)]
        if len(_sem_finite) > 0:
            _s4_logger.info(
                "S4-3: semantic_embedding_score computed for %d clusters "
                "(mean=%.3f, max=%.3f)",
                len(df),
                float(np.mean(_sem_finite)),
                float(np.max(_sem_finite)),
            )
        else:
            _s4_logger.warning(
                "S4-3: semantic_embedding_score is all-NaN for %d clusters "
                "— embedding failed; column will be excluded from feature matrix.",
                len(df),
            )

    w_if       = cfg.get("ensemble_if_weight",       0.30)   # S4-5: was 0.40
    w_ae       = cfg.get("ensemble_ae_weight",        0.25)   # S4-5: was 0.40
    w_formula  = cfg.get("ensemble_formula_weight",   0.35)   # S4-5: was 0.20
    w_semantic = cfg.get("ensemble_semantic_weight",  0.10)   # S4-5: new
    disag_thr = cfg.get("ensemble_disagreement_threshold", 0.35)
    score_mul = cfg.get("post_deploy_score_multiplier", 0.75)

    services = df[svc_col].fillna("unknown").unique()

    # S4-6: Build domain → [normal_row_indices] map for cross-service augmentation.
    # When a service has too few confirmed-normal rows to train IF/AE, we borrow
    # from domain peers (same domain, different service) to augment the pool.
    from collections import defaultdict as _defaultdict
    _domain_normal_pool: dict = _defaultdict(list)
    if "domain" in df.columns:
        for _svc_aug in df[svc_col].fillna("unknown").unique():
            _svc_mask_aug = df[svc_col].fillna("unknown") == _svc_aug
            _svc_dom = df.loc[_svc_mask_aug, "domain"].mode()
            _dom = _svc_dom.iloc[0] if len(_svc_dom) > 0 else "other"
            _normal_idx = df.index[_svc_mask_aug & normal_mask].tolist()
            _domain_normal_pool[_dom].extend(_normal_idx)
    for svc in services:
        svc_mask   = df[svc_col].fillna("unknown") == svc
        svc_df     = df[svc_mask]
        svc_normal = svc_df[normal_mask[svc_mask]]

        # S4-ML-NORMAL-FIX: augment svc_normal with pre-split routine rows for
        # this service.  These rows were removed from df before scoring (Step 0b-pre)
        # so they never appear in svc_df — but they ARE confirmed normal and are
        # the best possible training signal for IF/AE.  Without them X_normal is
        # empty for most services and ML falls back to formula/statistical scoring.
        if normal_pool_df is not None and len(normal_pool_df) > 0:
            _pool_svc_col = svc_col if svc_col in normal_pool_df.columns else None
            if _pool_svc_col is None:
                for _c in ["service", "top_source"]:
                    if _c in normal_pool_df.columns:
                        _pool_svc_col = _c
                        break
            if _pool_svc_col:
                _pool_svc_rows = normal_pool_df[
                    normal_pool_df[_pool_svc_col].fillna("unknown") == svc
                ]
            else:
                _pool_svc_rows = normal_pool_df  # no service col — use all
            if len(_pool_svc_rows) > 0:
                # Stringify any list-type columns in both frames before concat
                # so drop_duplicates() doesn't fail with "unhashable type: list".
                # The 'services' column joined from cluster_summary in Step 0a
                # is the most common offender.
                def _make_hashable(frame: pd.DataFrame) -> pd.DataFrame:
                    frame = frame.copy()
                    for _hc in frame.columns:
                        if frame[_hc].apply(
                            lambda v: isinstance(v, (list, dict, set))
                        ).any():
                            frame[_hc] = frame[_hc].apply(
                                lambda v: str(v) if isinstance(v, (list, dict, set)) else v
                            )
                    return frame

                svc_normal = pd.concat(
                    [_make_hashable(svc_normal), _make_hashable(_pool_svc_rows)],
                    ignore_index=True,
                ).drop_duplicates()
                _s4_logger.info(
                    "S4-ML-NORMAL-FIX: added %d pre-split routine rows to "
                    "normal pool for service='%s' (total normal=%d)",
                    len(_pool_svc_rows), svc, len(svc_normal),
                )

        X_all    = _build_feature_matrix(svc_df)
        X_normal = _build_feature_matrix(svc_normal)

        # S4-6: Cross-service normal pool augmentation.
        # If this service has fewer than if_min_rows confirmed-normal rows,
        # borrow rows from domain peers to reach the training threshold.
        if len(X_normal) < cfg.get("if_min_rows", 15) and "domain" in df.columns:
            _svc_dom_series = df.loc[svc_mask, "domain"].mode()
            _svc_dom = _svc_dom_series.iloc[0] if len(_svc_dom_series) > 0 else "other"
            # Exclude this service's own rows to avoid duplication
            _peer_idx = [
                i for i in _domain_normal_pool.get(_svc_dom, [])
                if i not in df.index[svc_mask]
            ]
            if _peer_idx:
                _peer_normal = _build_feature_matrix(df.loc[_peer_idx[:100]])
                X_normal = np.vstack([X_normal, _peer_normal])
                _s4_logger.info(
                    "S4-ML-8/S4-6: augmented '%s' normal pool with %d peer rows "
                    "(domain='%s')",
                    svc, len(_peer_idx[:100]), _svc_dom,
                )

        # Issue 3: per-service contamination estimate
        svc_formula_scores = svc_df["anomaly_score"].fillna(0.0).values.astype(np.float32)
        svc_contamination  = _estimate_contamination(svc_formula_scores, cfg)

        # Issue 2: WARNING when normal pool is empty or very small
        if len(X_normal) == 0:
            _s4_logger.warning(
                "S4-ML-8: X_normal is EMPTY for service='%s' — "
                "formula_fallback will be used. Check Stage 3 output.",
                svc,
            )
        elif len(X_normal) < cfg.get("if_min_rows", 15):
            _s4_logger.warning(
                "S4-ML-8: X_normal has only %d rows for service='%s' "
                "(if_min_rows=%d) — pool is very small, model may overfit.",
                len(X_normal), svc, cfg.get("if_min_rows", 15),
            )

        if_model  = None
        ae_model  = None
        ae_type   = None
        cache_hit = False

        svc_safe = re.sub(r"[^\w\-]", "_", str(svc))[:60]

        # S4-ML-7: try to load from cache (skip when is_post_deploy)
        if cache_dir is not None and not is_post_deploy and len(X_normal) > 0:
            fingerprint = _normal_fingerprint(X_normal)
            if_model_c, ae_model_c, ae_type_c = _load_cached_models(
                svc_safe, cache_dir, fingerprint
            )
            if if_model_c is not None or ae_model_c is not None:
                if_model  = if_model_c
                ae_model  = ae_model_c
                ae_type   = ae_type_c
                cache_hit = True
        else:
            fingerprint = _normal_fingerprint(X_normal) if len(X_normal) > 0 else ""

        if not cache_hit:
            # ── Train IF (FP-2: min_rows guard) ──────────────────────
            if cfg.get("ml_isolation_forest_enabled", True):
                if_model = _train_isolation_forest(
                    X_normal, cfg, contamination=svc_contamination
                )

            # ── Train AE (FP-2: min_rows guard) ──────────────────────
            if cfg.get("ml_autoencoder_enabled", True):
                ae_model, ae_type = _train_autoencoder(X_normal, cfg)

            # S4-ML-7: save newly trained models to disk
            if cache_dir is not None and not is_post_deploy and len(X_normal) > 0:
                _save_cached_models(
                    svc_safe, cache_dir, if_model, ae_model, ae_type, fingerprint
                )

        # ── Score ─────────────────────────────────────────────────────
        if_scores = (
            _score_isolation_forest(if_model, X_all)
            if if_model is not None
            else np.full(len(svc_df), 0.5, dtype=np.float32)
        )
        ae_scores = (
            _score_autoencoder(ae_model, ae_type, X_all)
            if ae_model is not None
            else np.full(len(svc_df), 0.5, dtype=np.float32)
        )
        formula_scores = svc_formula_scores

        # ── Determine source and fuse ─────────────────────────────────
        both_available = (if_model is not None) and (ae_model is not None)
        if_only        = (if_model is not None) and (ae_model is None)
        ae_only        = (if_model is None)     and (ae_model is not None)

        # S4-3: pull per-service semantic scores for ensemble fusion.
        # When _compute_semantic_embedding_scores returned an all-NaN Series
        # (embedding failure), fillna(0.0) silently makes sem_scores a zero
        # vector — which shifts all ensemble scores down by w_semantic*0.0=0,
        # i.e. no shift, BUT the other weights no longer sum to 1.0 (they sum
        # to 0.90).  Fix: detect the all-NaN case and redistribute w_semantic
        # into w_formula so weights always sum to 1.0.  When at least some
        # scores are valid, partial NaNs are imputed with the service-level
        # mean (matching _build_feature_matrix's imputation strategy) so the
        # ensemble weight is fully utilised on rows that have real signal.
        _sem_raw = df.loc[svc_mask, "semantic_embedding_score"].values.astype(np.float64)
        _sem_finite_mask = np.isfinite(_sem_raw)
        _sem_available = _sem_finite_mask.any()

        if _sem_available:
            # Impute NaNs with per-service mean (same strategy as _build_feature_matrix)
            _sem_mean = float(np.mean(_sem_raw[_sem_finite_mask]))
            _sem_raw[~_sem_finite_mask] = _sem_mean
            sem_scores = _sem_raw.clip(0.0, 1.0).astype(np.float32)
            # Effective weights when semantic is available
            _w_if_eff      = w_if
            _w_ae_eff      = w_ae
            _w_formula_eff = w_formula
            _w_sem_eff     = w_semantic
        else:
            # Semantic column entirely NaN — redistribute its weight to formula
            sem_scores     = np.zeros(int(svc_mask.sum()), dtype=np.float32)
            _w_if_eff      = w_if
            _w_ae_eff      = w_ae
            _w_formula_eff = w_formula + w_semantic  # absorb semantic weight
            _w_sem_eff     = 0.0

        if both_available:
            ensemble = (_w_if_eff * if_scores + _w_ae_eff * ae_scores
                        + _w_formula_eff * formula_scores + _w_sem_eff * sem_scores)
            src      = "ae_if_ensemble"
            conf     = min(len(X_normal) / max(cfg.get("ae_min_rows", 30), 1), 1.0)
        elif if_only:
            # Weights: IF=0.55, formula=0.35, semantic=0.10 (original).
            # When semantic unavailable: IF=0.55, formula=0.45, semantic=0.
            _if_sem_w  = 0.10 if _sem_available else 0.0
            _if_form_w = 0.35 + (0.0 if _sem_available else 0.10)
            ensemble = 0.55 * if_scores + _if_form_w * formula_scores + _if_sem_w * sem_scores
            src      = "ml_if_only"
            conf     = min(len(X_normal) / max(cfg.get("if_min_rows", 15), 1), 1.0) * 0.7
        elif ae_only:
            _ae_sem_w  = 0.10 if _sem_available else 0.0
            _ae_form_w = 0.35 + (0.0 if _sem_available else 0.10)
            ensemble = 0.55 * ae_scores + _ae_form_w * formula_scores + _ae_sem_w * sem_scores
            src      = "ml_ae_only"
            conf     = min(len(X_normal) / max(cfg.get("ae_min_rows", 30), 1), 1.0) * 0.7
        else:
            # Fix 3: Z-Score/MAD statistical fallback — intermediate tier between
            # ML models and pure formula scoring.
            #
            # Problem (from fix report):
            #   When a service has fewer rows than if_min_rows (15) or ae_min_rows (30),
            #   both IF and AE skip entirely and the ensemble falls back to a weighted
            #   formula score.  For sparse logs (microservices, startup files) this means
            #   ML contributes nothing — the formula scorer gets 100% weight even though
            #   a simple statistical outlier test on the feature matrix would still catch
            #   genuine anomalies without requiring a trained model.
            #
            # Fix: when X_all has ≥ 2 rows (minimum for variance to be defined),
            #   1. Compute MAD-normalised outlier scores on the feature matrix X_all.
            #      MAD (Median Absolute Deviation) is more robust than Z-Score for log
            #      data because frequency distributions are right-skewed — a burst event
            #      inflates the mean but not the median, so Z-Score would flag everything
            #      around the burst as normal while MAD correctly identifies it.
            #   2. Blend statistical scores with formula at 50/50 weight.
            #      50/50 is deliberate: the statistical score is noisy on small N,
            #      so it should not dominate; the formula score provides the semantic
            #      grounding (severity, rarity) that pure statistics cannot.
            #   3. Set src="statistical_fallback" and conf=0.3 (honest uncertainty:
            #      better than formula alone but far short of a trained model).
            #   4. When semantic scores are available, redistribute 10% weight from
            #      formula into semantic (same redistribution as the ML paths above).
            #
            # Regression safety:
            #   - When X_all has < 2 rows or MAD computation fails for any reason,
            #     the original formula_fallback path is taken unchanged.
            #   - The grace period check (FP-3 / Issue 6) below uses
            #     `src != "formula_fallback"` — statistical_fallback must be included
            #     so the grace period also applies to statistically-scored new templates.
            #     NOTE: this is intentional — a new template that is a statistical outlier
            #     in a post-deploy run should get the same score dampening as ML paths.
            #   - conf=0.3 ensures that compute_ml_anomaly_scores logs this service
            #     as low-confidence, making it visible in the S4-ML summary.
            #
            # Time complexity: O(N·F) for median/MAD computation (N=rows, F=features).
            # Space complexity: O(N·F) — same as X_all which is already allocated.
            _stat_score_computed = False
            if len(X_all) >= 2:
                try:
                    # Compute MAD-normalised outlier scores per feature, then aggregate.
                    # median and MAD are computed across rows (axis=0) so each feature
                    # gets its own scale, then we take the max per-row outlier score.
                    _med   = np.median(X_all, axis=0)              # shape (F,)
                    _diff  = np.abs(X_all - _med)                   # shape (N, F)
                    _mad   = np.median(_diff, axis=0)               # shape (F,)
                    # Replace zero MAD (constant feature) with 1.0 to avoid division.
                    _mad   = np.where(_mad < 1e-9, 1.0, _mad)
                    # Modified Z-score: 0.6745 * |X - median| / MAD
                    # (0.6745 = 1/Φ⁻¹(0.75), making it consistent with standard Z)
                    _mod_z = 0.6745 * _diff / _mad                  # shape (N, F)
                    # Aggregate: take the mean modified Z across features so a single
                    # anomalous feature doesn't dominate (unlike max aggregation).
                    # Clip the raw Z at 5 (beyond 5σ → certain outlier) before
                    # normalising so a single extreme value doesn't compress the rest.
                    _z_agg = _mod_z.mean(axis=1).clip(0.0, 5.0)    # shape (N,)
                    # Normalise to [0, 1] so it's on the same scale as formula scores.
                    _z_min, _z_max = _z_agg.min(), _z_agg.max()
                    if _z_max - _z_min > 1e-9:
                        stat_scores = ((_z_agg - _z_min) / (_z_max - _z_min)).astype(np.float32)
                    else:
                        # All rows are identical → no outlier signal; use formula only
                        stat_scores = np.zeros(len(X_all), dtype=np.float32)

                    # Blend: 50% statistical, semantic-adjusted formula for the rest.
                    # When semantic available: 50% stat + 40% formula + 10% semantic.
                    # When not:               50% stat + 50% formula.
                    _stat_w  = 0.50
                    _sem_w   = 0.10 if _sem_available else 0.0
                    _form_w  = 1.0 - _stat_w - _sem_w   # 0.40 or 0.50
                    ensemble = (_stat_w  * stat_scores
                                + _form_w  * formula_scores
                                + _sem_w   * sem_scores)
                    src  = "statistical_fallback"
                    conf = 0.3   # honest: better than formula but far below trained model
                    _stat_score_computed = True
                    _s4_logger.debug(
                        "Fix 3 (Z-Score/MAD): statistical_fallback for service='%s' "
                        "(n_rows=%d, z_range=[%.3f, %.3f], blend=50%%stat+%.0f%%form+%.0f%%sem)",
                        svc, len(X_all), float(_z_min), float(_z_max),
                        _form_w * 100, _sem_w * 100,
                    )
                except Exception as _stat_exc:
                    _s4_logger.debug(
                        "Fix 3 (Z-Score/MAD): statistical scoring failed for service='%s' "
                        "(%s) — falling back to formula only.", svc, _stat_exc
                    )

            if not _stat_score_computed:
                # Original formula_fallback path — unchanged.
                # Reached when: X_all has < 2 rows, OR Z-Score computation failed.
                _fb_sem_w  = 0.10 if _sem_available else 0.0
                _fb_form_w = 0.90 + (0.0 if _sem_available else 0.10)
                ensemble = _fb_form_w * formula_scores + _fb_sem_w * sem_scores
                src      = "formula_fallback"
                conf     = 0.0

        # ── FP-3 / Issue 6: Apply grace period to new-template rows only ──
        if is_post_deploy and src != "formula_fallback" and _ref_col is not None:
            row_tids      = svc_df[_ref_col].fillna("").astype(str).values
            new_rows_mask = np.array([t in new_tid_set for t in row_tids])
            if new_rows_mask.any():
                ensemble = ensemble.copy()
                ensemble[new_rows_mask] = (
                    ensemble[new_rows_mask] * score_mul
                ).clip(0.0, 1.0)
            src_array = np.where(new_rows_mask, "post_deploy_caution", src)
        else:
            src_array = None  # single value for all rows in this service

        # ── FP-4: Disagreement flag ───────────────────────────────────
        disagreement = np.abs(if_scores - ae_scores) > disag_thr

        # Write back to df
        idx = df.index[svc_mask]
        df.loc[idx, "ml_if_score"]           = np.round(if_scores.clip(0.0, 1.0), 4)
        df.loc[idx, "ml_ae_score"]           = np.round(ae_scores.clip(0.0, 1.0), 4)
        df.loc[idx, "ml_ensemble_score"]     = np.round(ensemble.clip(0.0, 1.0), 4)
        df.loc[idx, "ensemble_disagreement"] = disagreement
        df.loc[idx, "ml_confidence"]         = round(float(conf), 4)
        if src_array is not None:
            df.loc[idx, "anomaly_source"] = src_array
        else:
            df.loc[idx, "anomaly_source"] = src

        # ── FP-6 / Issue 7: Incremental AE update on top-N normal clusters ──
        if ae_model is not None and len(svc_normal) > 0:
            top_n     = cfg.get("ae_incremental_top_n", 50)
            count_col = "count" if "count" in svc_normal.columns else None
            if count_col:
                top_normal = svc_normal.nlargest(top_n, count_col)
            else:
                top_normal = svc_normal.head(top_n)
            X_update = _build_feature_matrix(top_normal)
            if len(X_update) >= 2:
                _incremental_ae_update(ae_model, ae_type, X_update, cfg)
                # S4-ML-7: save updated model after incremental update
                if cache_dir is not None and not is_post_deploy:
                    _save_cached_models(
                        svc_safe, cache_dir, if_model, ae_model, ae_type, fingerprint
                    )

        svc_label = str(svc) if len(str(svc)) <= 20 else str(svc)[:20]
        _s4_logger.info(
            "S4-ML: service='%s' if=%s ae=%s normal_rows=%d scored=%d "
            "contamination=%.3f src=%s cache=%s",
            svc_label,
            "✓" if if_model else "✗",
            "✓" if ae_model else "✗",
            len(X_normal), len(X_all),
            svc_contamination,
            src,
            "hit" if cache_hit else "miss",
        )

    if "_svc_tmp" in df.columns:
        df = df.drop(columns=["_svc_tmp"])

    return df


# ══════════════════════════════════════════════════════════════════════
# END S4-ML
# ══════════════════════════════════════════════════════════════════════

# ── A12 FIX: epsilon-shifted pd.cut bins ─────────────────────────────
_EPS = 1e-9


def _anomaly_label_from_score(score_series, tc, th, tm):
    """Apply Blueprint §5.3 thresholds with correct boundary inclusion.
    score >= tm → MEDIUM (not LOW), score >= th → HIGH, score >= tc → CRITICAL.
    """
    return pd.cut(
        score_series,
        bins=[-np.inf, tm - _EPS, th - _EPS, tc - _EPS, np.inf],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    ).astype(str)


# ── CONFIGURATION ─────────────────────────────────────────────────────
STAGE4_CONFIG = {
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

    "weight_rarity"      : 0.35,
    "weight_severity"    : 0.30,
    "weight_burstiness"  : 0.20,
    "weight_spread"      : 0.05,

    "singleton_bonus": {
        "true_anomaly_error"       : 0.35,
        "true_anomaly_warn"        : 0.20,
        "impossible_attempt_count" : 0.25,
        "unseen_variant"           : 0.05,
    },

    "threshold_critical" : 0.80,
    "threshold_high"     : 0.60,
    "threshold_medium"   : 0.35,

    "burst_target_bins"   : 40,
    "burst_min_window_s"  : 60,
    "burst_multiplier"    : 3.0,
    "burst_min_peak_count": 3,

    "sev_weight": {"ERROR": 1.0, "WARN": 0.5, "INFO": 0.1, "DEBUG": 0.05},
    "sev_char_map": {
        "E": "ERROR", "W": "WARN", "I": "INFO",
        "D": "DEBUG", "e": "ERROR", "w": "WARN", "i": "INFO",
    },

    "error_volume_percentile": 95,

    "warn_singleton_anomaly_keywords": {
        "timeout", "refused", "failed", "failure", "unreachable",
        "unavailable", "error", "exception", "critical", "corrupt",
        "invalid", "unauthorized", "forbidden", "panic", "crash",
        "deadlock", "overflow", "rejected", "denied", "blocked",
        "mitm", "injection", "brute", "cascade",
    },

    "trend_min_count"        : 5,
    "trend_rising_ratio"     : 1.5,
    "trend_falling_ratio"    : 0.67,
    "trend_rising_boost"     : 0.05,

    "cascade_window_s"       : 300,
    "cascade_min_occurrences": 2,
    "cascade_score_boost"    : 0.08,

    "master_gt_path"         : "validation/master_ground_truth.csv",
    "gt_label_col"           : "expected_anomaly_label",
    "gt_template_col"        : "expected_template_id",
    "gt_anomaly_labels"      : {"HIGH", "CRITICAL"},
    "gt_min_rows"            : 3,
    "gt_elevated_threshold"  : 0.25,
    "gt_suppressed_threshold": 0.25,
    "gt_elevated_boost"      : 0.05,

    # ACC-1
    "weight_volume_severity" : 0.10,
    "volume_norm_percentile" : 75,

    # ACC-2
    "routine_count_percentile": 90,
    "routine_burst_max"        : 0.10,
    "suppress_routine"         : True,

    # ACC-3
    # S4-2 FIX: Removed drone/UAV-specific tokens ("frame_class", "frame_type",
    # "no parameter target", "waiting for parameter") — these are ArduPilot/MAVLink
    # firmware terms that will false-positive on any config-management log that
    # mentions "parameter". Only universally-applicable stuck-state phrases remain.
    "stuck_state_tokens": {
        "0%", "sync: 0", "progress: 0",
        "not available yet", "did not return",
        "access is denied", "semaphore timeout",
        "timed out waiting",
    },

    # FIX-I
    "http_304_never_escalate": True,

    # ACC-4
    "success_signal_patterns": [
        r"\b0 error[s]?\b",
        r"\barchived\b",
        r"\bcompleted successfully\b",
        r"\bsuccess(?:ful(?:ly)?)?\b",
        r"^\s*✅",
        r"\bpm2 flush completed\b",
        r"\bfinished:.*archived\b",
        r"\bno error[s]?\b",
        r"\ball.*ok\b",
        r"\bverified successfully\b",
        r"\bconnection verified\b",
        # S4-ACC4-MAVLINK: hardware/telemetry/connectivity operational messages
        # that are unambiguously successful.  These are MAVLink/ArduPilot/serial-
        # port patterns that produce 0% error rates but were being scored HIGH
        # because no ACC-4 pattern matched them.  Only add patterns that are
        # definitively non-error regardless of context.
        r"\bserial port (open|clos)",       # "Opening serial port", "Serial port closed"
        r"\bopening serial\b",              # "Opening Serial connection"
        r"\bbackend ready\b",               # "Backend ready. Waiting for UI connection"
        r"\bparam_value\b",                 # MAVLink PARAM_VALUE parameter read
        r"\brequested stream rate\b",       # MAVLink stream rate negotiation
    ],
    "success_corrector_min_score": 0.50,

    # FIX-2
    # S4-1 FIX: session_check_routes now ships with only generic, universally-safe
    # patterns (/healthz, /ping). App-specific routes (/api/user, /api/me,
    # /api/auth/status, /api/notifications/unread, /api/session) have been removed —
    # on a different app, those same paths may represent genuine unauthorised-access
    # attempts and must NOT be score-capped by default.
    # Per-deployment configs should extend this list with their own session-polling
    # endpoints as needed (pass as cfg={"session_check_routes": [...]}).
    "session_check_401_cap": True,
    "session_check_routes": [
        r"/healthz?\b",
        r"/ping\b",
    ],
    "session_check_401_max_score": 0.34,

    # ACC-5
    # S4-8 (updated): provider-aware LLM config.
    # acc5_llm_model is the Anthropic model used when llm_provider="anthropic".
    # acc5_ollama_model / acc5_ollama_base_url are used when llm_provider="ollama".
    # These are overridden at runtime by config.py via llm_provider/ollama_model/
    # ollama_base_url keys injected into STAGE4_SETTINGS.
    "acc5_llm_model":        "claude-haiku-4-5-20251001",  # Anthropic fallback model
    "acc5_ollama_model":     "llama3.2",                   # Ollama primary model
    "acc5_ollama_base_url":  "http://localhost:11434",      # Ollama server URL
    "acc5_hard_error_keywords": [        r"\bexception\b", r"\bpanic\b", r"\bfatal\b", r"\babort(?:ed)?\b",
        r"\boom\b", r"\bout.of.memory\b", r"\bsegfault\b", r"\bsegmentation.fault\b",
        r"\bkilled\b", r"\bcore.dump(?:ed)?\b", r"\bstack.overflow\b",
        r"\bnull.pointer\b", r"\bbus.error\b", r"\bdeadlock\b",
    ],
    "acc5_soft_retry_keywords": [
        r"\bretry(?:ing)?\b", r"\breconnect(?:ing)?\b",
        r"\battempt\s+\d+\s+of\s+\d+\b", r"\bwill\s+retry\b",
        r"\bback.?off\b", r"\btransient\b", r"\btemporarily\s+unavailable\b",
        r"\bwaiting\s+to\s+reconnect\b",
    ],
    "acc5_soft_retry_error_pct_max": 30.0,
    "acc5_enabled": True,
    # S4-8 (provider-aware): LLM-assisted borderline classification for ACC-5
    "acc5_llm_enabled":     True,
    "acc5_llm_max_tokens":  50,
    "acc5_llm_cache":       {},   # in-memory template hash cache (populated at runtime)

    # ACC-6
    "acc6_rarity_floor_pct_threshold": 0.50,
    "acc6_rarity_score_floor"        : 0.05,
    "acc6_enabled"                   : True,

    # ── S4-ML: ML anomaly scorer config ──────────────────────────────
    # S4-ML-1/2: enable/disable each scorer independently
    "ml_isolation_forest_enabled": True,
    "ml_autoencoder_enabled":      True,

    # S4-ML-1: Isolation Forest params
    "if_n_estimators":    100,
    "if_contamination":   "auto",  # overridden per-service; "auto" = sklearn default
    "if_random_state":    42,
    "if_min_rows":        15,      # S4-2: was 20 — allows ML for smaller services

    # S4-ML-2: Autoencoder params
    "ae_hidden_dim":        64,    # spec: 64 (was 32)
    "ae_latent_dim":        32,    # spec: 32 (was 8)
    "ae_epochs":            30,
    "ae_lr":                1e-3,
    "ae_batch_size":        32,
    "ae_min_rows":          30,    # S4-2: was 50 — allows AE for smaller services
    "ae_incremental_steps": 5,     # Issue 7: was 2

    # S4-ML-3: Ensemble fusion weights (S4-5: rebalanced from 40/40/20)
    "ensemble_if_weight":       0.30,   # S4-5: was 0.40
    "ensemble_ae_weight":       0.25,   # S4-5: was 0.40
    "ensemble_formula_weight":  0.35,   # S4-5: was 0.20 — formula carries semantic load
    "ensemble_semantic_weight": 0.10,   # S4-5: new — semantic embedding score

    # S4-ML-4: Post-deployment grace period
    "post_deploy_new_tid_threshold": 0.30,
    "post_deploy_score_multiplier":  0.75,
    "post_deploy_grace_enabled":     True,

    # S4-ML-5: Incremental AE update
    "ae_incremental_top_n": 50,

    # S4-ML-6: Ensemble disagreement threshold (spec: 0.35)
    "ensemble_disagreement_threshold": 0.35,

    # S4-ML-7: Model persistence
    # S4-1: enabled by default so trained models persist across runs.
    # Set to None to disable caching entirely.
    "ml_model_cache_dir": "models/stage4_cache",

    # S4-3: Semantic embedding score
    "semantic_embedding_enabled": True,
    "semantic_embedding_weight":  0.10,
    "known_anomaly_patterns": [
        "Connection refused",
        "OutOfMemoryError",
        "NullPointerException",
        "Segmentation fault",
        "FATAL error in",
        "Service unavailable",
        "Disk full",
        "Authentication failed",
        "Certificate expired",
        "Deadlock detected",
        "Stack overflow",
        "Timeout after",
        "Connection reset by peer",
        "No space left on device",
    ],

    # Feature version — increment when _ML_FEATURE_COLS changes so that
    # persisted models with wrong input_dim are rejected at load time.
    "ml_feature_version": 3,  # S4-4: was 2

    # ── False-positive prevention (S4-FP patches) ────────────────────
    "burst_no_error_damping":  0.30,   # S4-FP-1: burstiness multiplier when error_pct==0 + INFO/DEBUG
    "warn_routine_score_cap":  0.34,   # S4-FP-2: ceiling for is_warn_routine clusters (below MEDIUM=0.35)
    "tfidf_max_label":         "MEDIUM",  # S4-FP-5: max label when TF-IDF fallback active
}

# Module-level mirror of the resolved config from the most recent run_stage4()
# call.  Notebook accuracy-audit cells that reference `full_cfg` directly will
# find this populated after run_stage4() returns.  Accessing it before the first
# call returns an empty dict — guard with `full_cfg or STAGE4_CONFIG` if needed.
full_cfg: dict = {}


# ── STEP 0: COLUMN RESOLUTION ─────────────────────────────────────────
def _resolve_col(df, name, fallbacks=(), default=None):
    if name in df.columns:
        return name
    for fb in fallbacks:
        if fb in df.columns:
            print(f"  [col_resolve] '{name}' not found — using '{fb}' instead")
            return fb
    if default is not None:
        print(f"  [col_resolve] '{name}' not found — using constant '{default}'")
    else:
        print(f"  [col_resolve] WARNING: '{name}' not found and no fallback available")
    return None


def resolve_columns(df, cfg):
    col = {}
    col["service"]        = _resolve_col(df, cfg["col_service"],
                                          fallbacks=["source", "svc", "host"])
    col["event_label"]    = _resolve_col(df, cfg["col_event_label"],
                                          fallbacks=["cluster_label", "label", "cluster_name"])
    col["event_id"]       = _resolve_col(df, cfg["col_event_id"],
                                          fallbacks=[
                                              "semantic_cluster_id",
                                              cfg["col_cluster_id"],
                                              "cluster_id",
                                              "template_id",
                                          ])
    # S4-4C: severity is guaranteed normalised from Stage 1. If the primary
    # column is absent, log a warning before falling back to alternatives.
    if cfg["col_severity"] not in df.columns:
        _s4_logger.warning(
            "S4: primary severity column '%s' not found — "
            "using fallback. Verify Stage 1 ran correctly.",
            cfg["col_severity"]
        )
    col["severity"]       = _resolve_col(df, cfg["col_severity"],
                                          fallbacks=["level", "log_level"])
    col["timestamp"]      = _resolve_col(df, cfg["col_timestamp"],
                                          fallbacks=["timestamp_parsed", "ts", "time",
                                                     "datetime", "log_time", "timestamp"])
    col["message"]        = _resolve_col(df, cfg["col_message"],
                                          fallbacks=["msg", "raw_line", "normalized_message"])
    col["is_noise"]       = _resolve_col(df, cfg["col_is_noise"])
    col["event_template"] = _resolve_col(df, cfg["col_event_template"],
                                          fallbacks=["template"])
    col["singleton_class"]= _resolve_col(df, cfg["col_singleton_class"])
    # S4-4A: domain is guaranteed from Stage 2 — no silent fallback to non-domain columns.
    # If "domain" is absent _resolve_col prints a WARNING, which is the correct signal
    # that Stage 2 did not run correctly.
    if cfg["col_domain"] not in df.columns:
        _s4_logger.warning(
            "S4: primary domain column '%s' not found in DataFrame — "
            "Stage 2 domain classification may not have run correctly. "
            "No fallback will be applied.",
            cfg["col_domain"]
        )
    col["domain"]         = _resolve_col(df, cfg["col_domain"])
    col["cluster_id"]     = _resolve_col(df, cfg["col_cluster_id"],
                                          fallbacks=["semantic_cluster_id", "event_id"])
    return col


# ── FIX-C: JOIN cluster_label FROM cluster_summary ────────────────────
def enrich_with_cluster_summary(df, cluster_summary_df):
    if cluster_summary_df is None or cluster_summary_df.empty:
        return df

    left_key = right_key = None
    for lk, rk in [
        ("semantic_cluster_id", "cluster_id"),
        ("event_id",            "event_id"),
        ("cluster_id",          "cluster_id"),
        ("template_id",         "template_id"),
    ]:
        if lk in df.columns and rk in cluster_summary_df.columns:
            left_key, right_key = lk, rk
            break

    if left_key is None:
        print("  [enrich] No common join key — skipping")
        return df

    # S4-ROUTINE: include is_routine and anomaly_signal from stage3 so that
    # run_stage4 can pre-split routine clusters before scoring, and
    # _split_routine_clusters can use the flag as a direct bypass condition.
    #
    # S4-4A: domain is now guaranteed from Stage 2 on every row. Exclude "domain"
    # from the gap-fill join — df["domain"] must not be overwritten by the
    # cluster_summary value. A consistency check is performed below instead.
    candidate_cols = ["cluster_label", "sample_template", "services",
                      "is_routine", "anomaly_signal", "is_warn_routine"]  # S3-FP-2: propagate warn_routine flag
    missing_cols   = [c for c in candidate_cols
                      if c in cluster_summary_df.columns and c not in df.columns]
    if not missing_cols:
        print("  [enrich] cluster_label / sample_template already present — no join needed")
        return df

    summary_slim = (
        cluster_summary_df[[right_key] + missing_cols]
        .drop_duplicates(subset=[right_key])
        .copy()
    )
    # Normalise join key dtype to plain object/str to avoid pandas StringDtype vs object mismatches
    summary_slim[right_key] = summary_slim[right_key].astype(str)
    df = df.copy()
    df[left_key] = df[left_key].astype(str)

    if left_key == right_key:
        enriched = df.merge(summary_slim, on=left_key, how="left")
    else:
        enriched = df.merge(
            summary_slim.rename(columns={right_key: left_key}),
            on=left_key, how="left",
        )
    # Expose sample_template as sample_message so the frontend has a stable field name
    if "sample_template" in enriched.columns and "sample_message" not in enriched.columns:
        enriched["sample_message"] = enriched["sample_template"]

    # S4-4A: Domain consistency check — domain is written by Stage 2 and is final.
    # Log a WARNING if cluster_summary_df carries a different domain value for the
    # same cluster_id. Do NOT overwrite enriched["domain"] with the summary value.
    if "domain" in df.columns and "domain" in cluster_summary_df.columns:
        _sum_domain = (
            cluster_summary_df[[right_key, "domain"]]
            .drop_duplicates(subset=[right_key])
            .rename(columns={right_key: left_key, "domain": "_summary_domain"})
            .copy()
        )
        _sum_domain[left_key] = _sum_domain[left_key].astype(str)
        _check = enriched[[left_key, "domain"]].merge(_sum_domain, on=left_key, how="left")
        _mismatch = _check[
            _check["_summary_domain"].notna() &
            (_check["domain"] != _check["_summary_domain"])
        ]
        if not _mismatch.empty:
            _s4_logger.warning(
                "S4-4A: domain mismatch between Stage 2 rows and cluster_summary_df "
                "for %d cluster(s): %s. Keeping Stage 2 values — do not override.",
                _mismatch[left_key].nunique(),
                _mismatch[left_key].unique()[:5].tolist(),
            )

    print(f"  [enrich] Joined {missing_cols} via {left_key}→{right_key} "
          f"({summary_slim[right_key].nunique()} unique keys in summary)")
    return enriched


# ── STEP 1: SEVERITY NORMALISATION ────────────────────────────────────
def normalise_severity(sev_series, cfg):
    char_map = cfg["sev_char_map"]
    def _norm(v):
        if pd.isna(v):
            return "UNKNOWN"
        v = str(v).strip()
        if v in char_map:
            return char_map[v]
        v_upper = v.upper()
        if v_upper == "WARNING":
            return "WARN"
        return v_upper if v_upper in {"ERROR", "WARN", "INFO", "DEBUG"} else "UNKNOWN"
    return sev_series.apply(_norm)


# ── STEP 2: BUILD CLEAN WORKING DATAFRAME ─────────────────────────────
def build_clean_df(df, col, cfg):
    mask = pd.Series(True, index=df.index)
    if col["is_noise"]:
        mask &= ~df[col["is_noise"]].fillna(False).astype(bool)
    if col["domain"] and col["domain"] in df.columns:
        mask &= df[col["domain"]].fillna("other") != "noise"
    if col["event_id"]:
        mask &= df[col["event_id"]].notna()
        if col["event_template"]:
            mask &= ~df[col["event_template"]].isin(["<NOISE>", "<EMPTY>"])

    df_c = df[mask].copy()

    if col["severity"]:
        df_c["_severity_norm"] = normalise_severity(df_c[col["severity"]], cfg)
    else:
        df_c["_severity_norm"] = "UNKNOWN"

    # FIX-L: tz-aware / tz-naive coercion guard
    ts_col = col["timestamp"]
    if ts_col and ts_col in df_c.columns:
        try:
            parsed = pd.to_datetime(df_c[ts_col], errors="coerce", utc=True)
            if parsed.dt.tz is not None:
                df_c["_ts"] = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
            else:
                df_c["_ts"] = parsed
        except Exception:
            df_c["_ts"] = pd.to_datetime(df_c[ts_col], errors="coerce", utc=False)
            if hasattr(df_c["_ts"], "dt") and df_c["_ts"].dt.tz is not None:
                df_c["_ts"] = df_c["_ts"].dt.tz_localize(None)
        ts_ok = df_c["_ts"].notna().sum()
        print(f"  Timestamp coverage: {ts_ok}/{len(df_c)} ({ts_ok/len(df_c)*100:.1f}%)")
    else:
        df_c["_ts"] = pd.NaT
        print("  Timestamps: column not found — burst/trend detection disabled")

    return df_c


# ── 4a. FREQUENCY ANALYSIS ────────────────────────────────────────────
def compute_frequency(df_c, col):
    eid_col = col["event_id"]
    msg_col = col["message"]

    if not eid_col:
        raise RuntimeError("compute_frequency: no event_id column resolved.")

    total = len(df_c)
    freq  = df_c.groupby(eid_col).size().reset_index(name="count")

    # LOW-3 FIX: incorporate repeat_count from Stage 1 dedup so that
    # consecutively collapsed duplicate lines are counted in burst detection.
    if "repeat_count" in df_c.columns:
        repeat_totals = (
            df_c[df_c[eid_col].notna()]
            .groupby(eid_col)["repeat_count"]
            .sum()
            .reset_index(name="repeat_total")
        )
        freq = freq.merge(repeat_totals, on=eid_col, how="left")
        freq["repeat_total"] = freq["repeat_total"].fillna(0).astype(int)
        # Effective count includes all collapsed duplicates
        freq["effective_count"] = freq["count"] + freq["repeat_total"]
    else:
        freq["effective_count"] = freq["count"]

    # FIX-K: explicit denominator
    freq["pct_of_total"] = (freq["count"] / total * 100).round(2)

    log_counts           = np.log1p(freq["count"])
    freq["rarity_score"] = 1.0 - (log_counts / max(float(log_counts.max()), 1e-9))

    # FIX-J: distinct message count per cluster
    if msg_col and msg_col in df_c.columns:
        distinct = (
            df_c.groupby(eid_col)[msg_col]
            .nunique()
            .reset_index(name="distinct_msg_count")
        )
        freq = freq.merge(distinct, on=eid_col, how="left")
        freq["distinct_msg_count"] = freq["distinct_msg_count"].fillna(1).astype(int)
    else:
        freq["distinct_msg_count"] = 1

    return freq


# ── 4b. SEVERITY ANALYSIS ─────────────────────────────────────────────
def compute_severity(df_c, col, cfg):
    eid_col    = col["event_id"]
    sev_counts = (
        df_c.groupby([eid_col, "_severity_norm"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    sev_counts.columns.name = None
    for s in ["ERROR", "WARN", "INFO", "DEBUG", "UNKNOWN"]:
        if s not in sev_counts.columns:
            sev_counts[s] = 0

    sev_cols = [s for s in ["ERROR", "WARN", "INFO", "DEBUG", "UNKNOWN"]
                if s in sev_counts.columns]
    sev_counts["total"] = sev_counts[sev_cols].sum(axis=1)

    weights = cfg["sev_weight"]
    sev_counts["severity_score"] = sum(
        sev_counts.get(s, 0) * w
        for s, w in weights.items()
        if s in sev_counts.columns
    ) / (sev_counts["total"] + 1e-9)

    sev_counts["dominant_severity"] = sev_counts[sev_cols].idxmax(axis=1)
    sev_counts["error_pct"] = (
        sev_counts.get("ERROR", 0) / (sev_counts["total"] + 1e-9) * 100
    ).round(1)

    keep = [eid_col, "severity_score", "dominant_severity",
            "error_pct", "ERROR", "WARN", "INFO"]
    keep = [c for c in keep if c in sev_counts.columns]
    return sev_counts[keep]


# ── 4c. TEMPORAL ANALYSIS ─────────────────────────────────────────────
def compute_temporal(df_c, col, cfg):
    eid_col = col["event_id"]

    # FIX-O: guard against empty / insufficient input
    if "_ts" not in df_c.columns or df_c["_ts"].notna().sum() < 10:
        print("  Temporal analysis: skipped (insufficient parsed timestamps)")
        return pd.DataFrame(columns=[eid_col, "first_seen", "last_seen",
                                      "burst_detected", "peak_window_count",
                                      "burst_window_minutes"])

    df_ts = df_c[df_c["_ts"].notna()].copy()

    # FIX-O: single-row edge case
    if len(df_ts) < 2:
        print("  Temporal analysis: skipped (fewer than 2 timestamped rows)")
        return pd.DataFrame(columns=[eid_col, "first_seen", "last_seen",
                                      "burst_detected", "peak_window_count",
                                      "burst_window_minutes"])

    span_seconds = (df_ts["_ts"].max() - df_ts["_ts"].min()).total_seconds()
    target_bins  = cfg["burst_target_bins"]
    min_window_s = cfg["burst_min_window_s"]
    window_s     = max(min_window_s, span_seconds / max(target_bins, 1))
    window_str   = f"{int(window_s)}s"
    window_min   = round(window_s / 60, 1)
    mult         = cfg["burst_multiplier"]
    min_peak     = cfg.get("burst_min_peak_count", 3)
    print(f"  Burst detection window: {window_min} min  "
          f"(span={span_seconds/3600:.1f}h, bins≈{span_seconds/window_s:.0f})")
    rows = []
    for eid, grp in df_ts.groupby(eid_col):
        ts_sorted  = grp["_ts"].sort_values()
        first_seen = ts_sorted.min()
        last_seen  = ts_sorted.max()
        hourly     = grp.set_index("_ts").resample(window_str).size()
        peak_count = int(hourly.max()) if len(hourly) > 0 else 0
        burst      = False
        if peak_count >= min_peak and len(hourly) >= 4:
            roll_mean = hourly.shift(1).rolling(window=3, min_periods=2).mean()
            if (hourly > roll_mean * mult).any():
                burst = True
        rows.append({
            eid_col               : eid,
            "first_seen"          : first_seen,
            "last_seen"           : last_seen,
            "burst_detected"      : burst,
            "peak_window_count"   : peak_count,
            "burst_window_minutes": window_min,
        })
    return pd.DataFrame(rows)


# ── 4d. SOURCE DISTRIBUTION ───────────────────────────────────────────
def compute_source(df_c, col):
    eid_col = col["event_id"]
    svc_col = col["service"]
    if not svc_col or df_c[svc_col].isna().all():
        print("  Source analysis: skipped (no service/source column)")
        return pd.DataFrame(columns=[eid_col, "top_source",
                                      "source_spread", "spread_score", "is_isolated"])
    n_services = df_c[svc_col].nunique()
    rows = []
    for eid, grp in df_c.groupby(eid_col):
        sources = grp[svc_col].dropna()
        if len(sources) == 0:
            continue
        vc          = sources.value_counts()
        top_source  = vc.index[0]
        top_pct     = vc.iloc[0] / len(sources)
        spread      = sources.nunique()
        is_isolated = (top_pct >= 0.90 and spread == 1)
        spread_score = (spread - 1) / max(n_services - 1, 1)
        rows.append({
            eid_col        : eid,
            "top_source"   : top_source,
            "source_spread": spread,
            "spread_score" : round(spread_score, 3),
            "is_isolated"  : is_isolated,
        })
    return pd.DataFrame(rows)


# ── FIX-E: TREND DIRECTION ────────────────────────────────────────────
def compute_trend_direction(df_c, col, cfg):
    eid_col   = col["event_id"]
    min_count = cfg.get("trend_min_count", 5)
    rising_r  = cfg.get("trend_rising_ratio", 1.5)
    falling_r = cfg.get("trend_falling_ratio", 0.67)

    # FIX-O: guard
    if "_ts" not in df_c.columns or df_c["_ts"].notna().sum() < 10:
        print("  Trend analysis: skipped (insufficient timestamps)")
        return pd.DataFrame(columns=[eid_col, "trend_direction",
                                      "trend_early_count", "trend_late_count",
                                      "trend_ratio"])

    df_ts = df_c[df_c["_ts"].notna()].copy()

    if len(df_ts) < 2:
        print("  Trend analysis: skipped (fewer than 2 timestamped rows)")
        return pd.DataFrame(columns=[eid_col, "trend_direction",
                                      "trend_early_count", "trend_late_count",
                                      "trend_ratio"])

    rows  = []
    for eid, grp in df_ts.groupby(eid_col):
        n = len(grp)
        if n < min_count:
            rows.append({eid_col: eid, "trend_direction": "stable",
                         "trend_early_count": n, "trend_late_count": n,
                         "trend_ratio": 1.0})
            continue
        ts_sorted = grp["_ts"].sort_values()
        t_min     = ts_sorted.min()
        t_max     = ts_sorted.max()
        span      = (t_max - t_min).total_seconds()
        if span < 1:
            rows.append({eid_col: eid, "trend_direction": "stable",
                         "trend_early_count": n, "trend_late_count": n,
                         "trend_ratio": 1.0})
            continue
        third   = span / 3.0
        t1      = t_min + pd.Timedelta(seconds=third)
        t2      = t_min + pd.Timedelta(seconds=2 * third)
        early_n = (ts_sorted <= t1).sum()
        late_n  = (ts_sorted >= t2).sum()
        ratio   = late_n / max(early_n, 1)
        direction = ("rising" if ratio >= rising_r else
                     "falling" if ratio <= falling_r else "stable")
        rows.append({
            eid_col             : eid,
            "trend_direction"   : direction,
            "trend_early_count" : int(early_n),
            "trend_late_count"  : int(late_n),
            "trend_ratio"       : round(ratio, 3),
        })
    result = pd.DataFrame(rows)
    if len(result):
        counts = result["trend_direction"].value_counts()
        print(f"  Trend distribution — "
              + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return result


# ── FIX-F: CROSS-SERVICE CASCADE DETECTION ───────────────────────────
def compute_cascade_flags(df_c, col, cfg):
    from collections import defaultdict

    eid_col   = col["event_id"]
    svc_col   = col["service"]
    window_s  = cfg.get("cascade_window_s", 300)
    min_occur = cfg.get("cascade_min_occurrences", 2)

    empty = pd.DataFrame(columns=[eid_col, "cascade_source",
                                   "cascade_target_services"])
    if not svc_col or "_ts" not in df_c.columns:
        print("  Cascade detection: skipped (need service + timestamps)")
        return empty
    if df_c[svc_col].isna().all() or df_c["_ts"].notna().sum() < 10:
        print("  Cascade detection: skipped (too few valid rows)")
        return empty

    df_events = (
        df_c[[eid_col, svc_col, "_ts"]]
        .dropna()
        .sort_values("_ts")
        .rename(columns={svc_col: "_svc"})
        .reset_index(drop=True)
    )
    services = df_events["_svc"].unique()
    if len(services) < 2:
        # S4-7: Intra-service cascade — detect temporal ordering of anomaly
        # clusters within a single service. Replaces the old early-return.
        svc = services[0] if len(services) == 1 else "unknown"
        svc_events = df_events[df_events["_svc"] == svc].copy()
        if len(svc_events) >= 2 and "_ts" in svc_events.columns:
            svc_events = svc_events.sort_values("_ts")
            eids    = svc_events[eid_col].tolist()
            ts_list = svc_events["_ts"].tolist()
            cascade_rows = []
            window = cfg.get("cascade_window_s", 300)
            for _i in range(len(eids) - 1):
                _dt = (ts_list[_i + 1] - ts_list[_i]).total_seconds()
                if 0 < _dt <= window:
                    _lag = int(_dt)
                    _label_i = str(
                        svc_events.iloc[_i].get("cluster_label", eids[_i])
                        if hasattr(svc_events.iloc[_i], "get")
                        else eids[_i]
                    )
                    _label_j = str(
                        svc_events.iloc[_i + 1].get("cluster_label", eids[_i + 1])
                        if hasattr(svc_events.iloc[_i + 1], "get")
                        else eids[_i + 1]
                    )
                    _chain = (
                        f"{svc}.{_label_i}(+0s) → {svc}.{_label_j}(+{_lag}s)"
                    )
                    cascade_rows.append({
                        eid_col                  : eids[_i + 1],
                        "cascade_source"         : eids[_i],
                        "cascade_target_services": svc,
                        "cascade_chain_intra"    : _chain,
                    })
            if cascade_rows:
                _intra_df = pd.DataFrame(cascade_rows)
                print(
                    f"  Cascade detection (intra-service): found "
                    f"{len(cascade_rows)} temporal pairs for service '{svc}'"
                )
                return _intra_df
        print("  Cascade detection: skipped (only one service, no temporal pairs)")
        return empty
    # HIGH-20 FIX: cap at top 30 most-frequent services to avoid O(n²) blowup
    if len(services) > 30:
        top_svcs = df_events["_svc"].value_counts().head(30).index.tolist()
        services = [s for s in services if s in top_svcs]
        print(f"  Cascade detection: capped to top 30 services "
              f"({df_events['_svc'].nunique()} total in data)")

    window_td    = pd.Timedelta(seconds=window_s)
    pair_records = []
    for svc_a in services:
        upstream = df_events[df_events["_svc"] == svc_a][["_ts", eid_col]].copy()
        upstream = upstream.rename(columns={"_ts": "_ts_up", eid_col: "_eid_up"})
        for svc_b in services:
            if svc_b == svc_a:
                continue
            downstream = df_events[df_events["_svc"] == svc_b][["_ts", eid_col]].copy()
            downstream = downstream.rename(
                columns={"_ts": "_ts_dn", eid_col: "_eid_dn"})
            if upstream.empty or downstream.empty:
                continue
            merged = pd.merge_asof(
                downstream.sort_values("_ts_dn"),
                upstream.sort_values("_ts_up"),
                left_on="_ts_dn", right_on="_ts_up",
                direction="backward", tolerance=window_td,
            ).dropna(subset=["_eid_up"])
            if len(merged) < min_occur:
                continue
            pair_records.append({
                "svc_a"  : svc_a,
                "svc_b"  : svc_b,
                "count"  : len(merged),
                "up_eids": set(merged["_eid_up"].unique()),
                "dn_eids": set(merged["_eid_dn"].unique()),
            })

    if not pair_records:
        print(f"  Cascade detection: no cascade patterns found "
              f"(window={window_s}s, min_occurrences={min_occur})")
        return empty

    print(f"  Cascade detection: found {len(pair_records)} upstream→downstream pairs")
    for rec in sorted(pair_records, key=lambda x: -x["count"])[:10]:
        print(f"    {rec['svc_a']} → {rec['svc_b']}  ({rec['count']} co-occurrences)")

    from collections import defaultdict
    downstream_to_sources = defaultdict(set)
    upstream_to_targets   = defaultdict(set)
    for rec in pair_records:
        for eid in rec["dn_eids"]:
            downstream_to_sources[eid].add(rec["svc_a"])
        for eid in rec["up_eids"]:
            upstream_to_targets[eid].add(rec["svc_b"])

    all_eids = set(downstream_to_sources) | set(upstream_to_targets)
    result_rows = [
        {
            eid_col                  : eid,
            "cascade_source"         : ", ".join(sorted(downstream_to_sources.get(eid, set()))) or None,
            "cascade_target_services": ", ".join(sorted(upstream_to_targets.get(eid, set()))) or None,
        }
        for eid in all_eids
    ]
    return pd.DataFrame(result_rows)


# ── FIX-G: BASELINE COMPARISON ───────────────────────────────────────
def compute_baseline_comparison(anomaly_df, col, cfg):
    gt_path      = cfg.get("master_gt_path", "validation/master_ground_truth.csv")
    gt_label_col = cfg.get("gt_label_col",    "expected_anomaly_label")
    gt_tmpl_col  = cfg.get("gt_template_col", "expected_template_id")
    anomaly_lbls = cfg.get("gt_anomaly_labels", {"HIGH", "CRITICAL"})
    min_rows     = cfg.get("gt_min_rows", 3)
    elev_thresh  = cfg.get("gt_elevated_threshold",   0.25)
    supp_thresh  = cfg.get("gt_suppressed_threshold", 0.25)
    eid_col      = col["event_id"]

    anomaly_df["gt_historical_rate"] = float("nan")
    anomaly_df["gt_comparison"]      = "no_gt"

    gt_file = Path(gt_path)
    if not gt_file.exists():
        # S4-4 FIX: Emit a visible INFO message rather than silently skipping.
        # The silent skip meant operators had no way to know GT scoring was absent.
        # Note: master_gt_path is a per-deployment config point — update it in
        # STAGE4_CONFIG or pass cfg={"master_gt_path": "..."} for your environment.
        print(
            f"  [FIX-G] INFO: Ground-truth file not found at '{gt_path}'. "
            f"GT baseline comparison and score boosts/suppression will be skipped. "
            f"Set master_gt_path in your deployment config to enable this feature."
        )
        return anomaly_df

    try:
        gt = pd.read_csv(gt_file, dtype=str)
    except Exception as exc:
        print(f"  [FIX-G] Could not read GT file: {exc} — skipping")
        return anomaly_df

    if gt_label_col not in gt.columns or gt_tmpl_col not in gt.columns:
        print(f"  [FIX-G] GT file missing required columns — skipping")
        return anomaly_df

    gt["_is_anomaly"] = gt[gt_label_col].isin(anomaly_lbls)
    gt_stats = (
        gt.groupby(gt_tmpl_col)["_is_anomaly"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "anomaly_count", "count": "total_count",
                         gt_tmpl_col: eid_col})
    )
    gt_stats = gt_stats[gt_stats["total_count"] >= min_rows].copy()
    gt_stats["gt_historical_rate"] = (
        gt_stats["anomaly_count"] / gt_stats["total_count"]
    ).round(3)

    tmpl_col_in_anomaly = None
    for candidate in ["template_id", eid_col, "event_id", "semantic_cluster_id"]:
        if candidate and candidate in anomaly_df.columns:
            tmpl_col_in_anomaly = candidate
            break
    if tmpl_col_in_anomaly is None:
        print("  [FIX-G] Cannot find matching template column — skipping")
        return anomaly_df

    gt_lookup = gt_stats[[eid_col, "gt_historical_rate"]].copy()
    if tmpl_col_in_anomaly != eid_col:
        gt_lookup = gt_lookup.rename(columns={eid_col: tmpl_col_in_anomaly})

    if "gt_historical_rate" in anomaly_df.columns:
        anomaly_df = anomaly_df.drop(columns=["gt_historical_rate"])
    anomaly_df = anomaly_df.merge(gt_lookup, on=tmpl_col_in_anomaly, how="left")

    hist_rate    = anomaly_df["gt_historical_rate"]
    label_col    = anomaly_df.get("anomaly_label", pd.Series("LOW", index=anomaly_df.index))
    current_rate = label_col.isin(anomaly_lbls).astype(float)
    diff         = current_rate - hist_rate.fillna(0.0)

    anomaly_df["gt_comparison"] = np.select(
        [hist_rate.isna(), diff >= elev_thresh, diff <= -supp_thresh],
        ["new_event", "elevated", "suppressed"],
        default="baseline",
    )

    matched    = hist_rate.notna().sum()
    new_events = (anomaly_df["gt_comparison"] == "new_event").sum()
    elevated   = (anomaly_df["gt_comparison"] == "elevated").sum()
    suppressed = (anomaly_df["gt_comparison"] == "suppressed").sum()
    print(f"  [FIX-G] GT baseline: {matched} matched, {new_events} new, "
          f"{elevated} elevated, {suppressed} suppressed")
    return anomaly_df


# ── FIX-B: WARN SINGLETON GUARD ──────────────────────────────────────
def _apply_warn_singleton_guard(anomaly_df, df_c, col, cfg):
    keywords = cfg.get("warn_singleton_anomaly_keywords", set())
    eid_col  = col["event_id"]
    tmpl_col = col["event_template"]

    if tmpl_col and tmpl_col in df_c.columns:
        tmpl_map = (
            df_c[[eid_col, tmpl_col]]
            .dropna(subset=[eid_col])
            .drop_duplicates(subset=eid_col)
            .set_index(eid_col)[tmpl_col]
        )
    else:
        tmpl_map = pd.Series(dtype=str)

    candidate_mask = (
        (anomaly_df["count"] == 1) &
        (anomaly_df.get("dominant_severity", pd.Series(dtype=str)) == "WARN") &
        (anomaly_df.get("singleton_class", pd.Series(dtype=str)) == "true_anomaly")
    )

    downgraded = 0
    if candidate_mask.any() and keywords:
        pattern    = "|".join(re.escape(kw) for kw in keywords)
        cand_eids  = anomaly_df.loc[candidate_mask, eid_col]
        tmpl_texts = cand_eids.map(tmpl_map).fillna("").str.lower()
        has_keyword = tmpl_texts.str.contains(pattern, regex=True, na=False)
        cand_positions = anomaly_df.index[candidate_mask]
        downgrade_idx  = cand_positions[~has_keyword.values]
        anomaly_df.loc[downgrade_idx, "singleton_class"] = "unseen_variant"
        downgraded = len(downgrade_idx)

    if downgraded:
        print(f"  [FIX-B] Downgraded {downgraded} WARN singletons → unseen_variant")
    return anomaly_df


# ── ACC-1: Volume-severity score helper ──────────────────────────────
def _compute_volume_severity_score(anomaly_df, cfg):
    sev_weights = cfg["sev_weight"]
    norm_pct    = cfg.get("volume_norm_percentile", 75)
    counts      = anomaly_df["count"].fillna(0)
    p_norm      = float(np.percentile(counts, norm_pct))
    log_norm    = np.log1p(max(p_norm, 1.0))
    volume_norm = (np.log1p(counts) / log_norm).clip(upper=1.0)
    dom_sev     = anomaly_df.get("dominant_severity",
                                  pd.Series("UNKNOWN", index=anomaly_df.index))
    sev_raw     = dom_sev.map(lambda s: sev_weights.get(str(s), 0.05))
    score       = (volume_norm * sev_raw).clip(upper=1.0).round(4)
    print(f"  [ACC-1] volume_severity_score computed "
          f"(norm_pct=p{norm_pct}, p{norm_pct}_count={p_norm:.0f})")
    return score


# ── FIX-N: ROBUST EID RESOLUTION ─────────────────────────────────────
def _resolve_eid_col(anomaly_df, eid_col_name=None):
    resolved = None
    candidates = []
    if eid_col_name:
        candidates.append(eid_col_name)
    candidates += ["semantic_cluster_id", "event_id", "cluster_id"]
    for c in candidates:
        if c in anomaly_df.columns:
            resolved = c
            break
    return resolved


# ── ACC-4: Success-signal severity corrector ──────────────────────────
def _apply_success_signal_corrector(anomaly_df, sample_msg_map, cfg, eid_col_name=None):
    patterns   = cfg.get("success_signal_patterns", [])
    min_score  = cfg.get("success_corrector_min_score", 0.50)
    sev_weight = cfg.get("sev_weight", {})
    w_vs       = cfg.get("weight_volume_severity", 0.10)
    w_rar      = cfg.get("weight_rarity",     0.35) * (1.0 - w_vs)
    w_sev      = cfg.get("weight_severity",   0.30) * (1.0 - w_vs)
    w_burst    = cfg.get("weight_burstiness", 0.20) * (1.0 - w_vs)
    w_spread   = cfg.get("weight_spread",     0.05) * (1.0 - w_vs)

    if not patterns:
        return anomaly_df

    compiled = re.compile("|".join(patterns), re.IGNORECASE | re.MULTILINE)
    anomaly_df["corrected_by_acc4"] = False

    resolved_eid_col = _resolve_eid_col(anomaly_df, eid_col_name)
    if resolved_eid_col is None:
        print("  [ACC-4] WARNING: cannot resolve EID column — skipping corrector")
        return anomaly_df

    candidates = (
        (anomaly_df.get("dominant_severity", pd.Series(dtype=str)) == "ERROR") &
        (anomaly_df["anomaly_score"] >= min_score)
    )
    n_corrected = 0

    for idx in anomaly_df.index[candidates]:
        eid = anomaly_df.at[idx, resolved_eid_col]
        msg = str(sample_msg_map.get(eid, "")).strip()
        if not msg:
            continue

        if compiled.search(msg):
            info_sev_score = sev_weight.get("INFO", 0.1)

            anomaly_df.at[idx, "dominant_severity"] = "INFO"
            anomaly_df.at[idx, "error_pct"]         = 0.0
            anomaly_df.at[idx, "severity_score"]    = info_sev_score

            rarity  = anomaly_df.at[idx, "rarity_score"]
            burst   = anomaly_df.at[idx, "burstiness_score"]
            spread  = anomaly_df.at[idx, "spread_score"]
            vol_sev = anomaly_df.at[idx, "volume_severity_score"]
            bonus   = anomaly_df.at[idx, "_singleton_bonus"]

            new_base = (
                rarity         * w_rar   +
                info_sev_score * w_sev   +
                burst          * w_burst +
                spread         * w_spread +
                vol_sev        * w_vs
            )
            anomaly_df.at[idx, "_base_score"]   = round(new_base, 4)
            anomaly_df.at[idx, "anomaly_score"] = round(min(new_base + bonus, 1.0), 3)
            anomaly_df.at[idx, "corrected_by_acc4"] = True
            n_corrected += 1

    tc = cfg["threshold_critical"]
    th = cfg["threshold_high"]
    tm = cfg["threshold_medium"]
    if n_corrected > 0:
        corrected_idx = anomaly_df[anomaly_df["corrected_by_acc4"]].index
        anomaly_df.loc[corrected_idx, "anomaly_label"] = _anomaly_label_from_score(
            anomaly_df.loc[corrected_idx, "anomaly_score"], tc, th, tm
        )

    print(f"  [ACC-4] Success-signal corrector: {n_corrected} ERROR clusters "
          f"re-scored as INFO (min_score={min_score})")
    if n_corrected > 0:
        corrected = anomaly_df[anomaly_df["corrected_by_acc4"]]
        for _, r in corrected.iterrows():
            eid_val = r.get(resolved_eid_col, "?")
            print(f"    ↳ {eid_val}: score {r['anomaly_score']:.3f} → {r['anomaly_label']}")

    return anomaly_df


# ── S4-8 / ACC-5 LLM-assisted borderline classification ──────────────

def _get_anthropic_client():
    """Return an Anthropic client if the package is available, else None."""
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None


def _llm_classify_borderline(template_text: str, cfg: dict) -> str:
    """
    S4-8: Ask the LLM to classify a borderline log cluster as
    'error', 'benign', or 'uncertain'.

    Provider-aware: reads cfg['llm_provider'] to choose between Ollama
    (default) and Anthropic (fallback / production path).

    Results are cached by MD5 hash of the full prompt context (not just the
    template text) to avoid repeated API calls for identical inputs within a
    run.  The cache key now incorporates all scoring fields so two clusters
    with the same template text but different severity / burst / score values
    are correctly treated as distinct classification requests.

    MPCD §3.4.1 fix: The prompt now includes dominant_severity, count,
    burst_detected, anomaly_score, and error_pct — all the context that
    makes a borderline decision meaningful.  The caller passes these via
    the 'acc5_row_context' key in cfg (a temporary shallow dict populated
    per-row by _apply_severity_context_correction).

    MPCD §4.1 fix: Exponential backoff (2 retries, 2s / 4s delays) before
    falling through to 'uncertain'.  The deterministic fallback is only
    triggered after all retries are exhausted.

    Returns one of: 'error', 'benign', 'uncertain'.
    Never raises — all failures degrade gracefully to 'uncertain'.

    Time complexity : O(1) per call (single API round-trip or cache lookup).
    Space complexity: O(C) where C = number of unique prompt contexts seen
                      in the run (the MD5 cache).  Bounded by cluster count.
    """
    if not cfg.get("acc5_llm_enabled", True):
        return "uncertain"

    # ── Build the context-rich prompt (MPCD §3.4.1) ──────────────────
    # Extract per-row scoring context injected by _apply_severity_context_correction.
    # Fall back to safe defaults for every field so old call-sites that don't
    # provide the context dict still work without crashing.
    row_ctx = cfg.get("acc5_row_context", {})
    dominant_severity = row_ctx.get("dominant_severity", "UNKNOWN")
    count             = row_ctx.get("count", "unknown")
    burst_flag        = row_ctx.get("burst_detected", False)
    anomaly_score     = row_ctx.get("anomaly_score", "unknown")
    error_pct         = row_ctx.get("error_pct", 0)

    # Format error_pct safely — may be a float or the string "unknown"
    try:
        error_pct_str = f"{float(error_pct):.1f}%"
    except (TypeError, ValueError):
        error_pct_str = str(error_pct)

    prompt = (
        "You are an SRE anomaly classifier. Classify the log cluster below as:\n"
        "  error   — this is a genuine anomaly that warrants investigation\n"
        "  benign  — this is routine or expected behaviour\n"
        "  uncertain — genuinely ambiguous, insufficient signal\n\n"
        "Context:\n"
        f"  Template    : {template_text[:300]}\n"
        f"  Severity    : {dominant_severity}\n"
        f"  Count       : {count}\n"
        f"  Burst flag  : {burst_flag}\n"
        f"  Anomaly score: {anomaly_score}\n"
        f"  Error pct   : {error_pct_str}\n\n"
        "Reply with exactly one word: error, benign, or uncertain."
    )

    # ── MD5 cache keyed on the full prompt (includes all context fields) ──
    # Caching on the full prompt (rather than just template_text) means two
    # clusters with the same template but different scores get independent
    # verdicts.  Cost: slightly more cache misses than template-only keying,
    # but always correct.  Cache size is O(unique clusters) — acceptable.
    cache = cfg.setdefault("acc5_llm_cache", {})
    key   = hashlib.md5(prompt.encode("utf-8", errors="replace")).hexdigest()
    if key in cache:
        return cache[key]

    provider = cfg.get("llm_provider", "ollama")

    # ── Shared backoff helper (MPCD §4.1) ────────────────────────────
    # Defined inline to avoid adding a module-level function that changes
    # the public API surface.  max_retries=2 gives delays of 2s then 4s
    # before giving up — matches the spec exactly.
    import time as _time

    def _call_with_backoff(call_fn, max_retries: int = 2, base_delay: float = 2.0):
        """
        Call call_fn(); retry up to max_retries times with exponential backoff.
        Returns the result on success.  Re-raises the final exception so the
        outer try/except can catch it and return 'uncertain'.

        Time complexity : O(max_retries) = O(1) — constant retries.
        The total wall-clock wait is base_delay * (2^max_retries - 1) seconds
        in the worst case (2 + 4 = 6 s for the defaults).
        """
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return call_fn()
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 2s, 4s
                    _s4_logger.warning(
                        "S4-8: LLM call attempt %d/%d failed (%s) — "
                        "retrying in %.0fs",
                        attempt + 1, max_retries + 1, exc, delay,
                    )
                    _time.sleep(delay)
        raise last_exc  # re-raise after all retries exhausted

    # ── Ollama path ───────────────────────────────────────────────────
    if provider == "ollama":
        try:
            import requests as _requests
            ollama_base_url = cfg.get("acc5_ollama_base_url",
                                      cfg.get("ollama_base_url", "http://localhost:11434"))
            ollama_model    = cfg.get("acc5_ollama_model",
                                      cfg.get("ollama_model", "llama3.2"))

            def _ollama_call():
                resp = _requests.post(
                    f"{ollama_base_url}/api/generate",
                    json={"model": ollama_model, "prompt": prompt, "stream": False},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip().lower()

            raw_text = _call_with_backoff(_ollama_call)
            verdict  = raw_text if raw_text in ("error", "benign") else "uncertain"
            cache[key] = verdict
            _s4_logger.debug(
                "S4-8 [ollama]: classified '%s' → %s", template_text[:60], verdict
            )
            return verdict
        except Exception as exc:
            _s4_logger.warning(
                "S4-8 [ollama]: borderline classification failed after retries "
                "(%s) — returning 'uncertain'",
                exc,
            )
            return "uncertain"

    # ── Anthropic path (fallback / production) ────────────────────────
    try:
        client = _get_anthropic_client()
        if client is None:
            return "uncertain"

        def _anthropic_call():
            resp = client.messages.create(
                model=cfg.get("acc5_llm_model", "claude-haiku-4-5-20251001"),
                max_tokens=cfg.get("acc5_llm_max_tokens", 50),
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip().lower()

        result  = _call_with_backoff(_anthropic_call)
        verdict = result if result in ("error", "benign") else "uncertain"
        cache[key] = verdict
        _s4_logger.debug(
            "S4-8 [anthropic]: classified '%s' → %s", template_text[:60], verdict
        )
        return verdict
    except Exception as exc:
        _s4_logger.warning(
            "S4-8 [anthropic]: borderline classification failed after retries "
            "(%s) — returning 'uncertain'", exc
        )
        return "uncertain"


# ── ACC-5: Severity-context contradiction detector ────────────────────
def _apply_severity_context_correction(anomaly_df, sample_msg_map, cfg, eid_col_name=None):
    if not cfg.get("acc5_enabled", True):
        return anomaly_df

    hard_error_kws    = cfg.get("acc5_hard_error_keywords", [])
    soft_retry_kws    = cfg.get("acc5_soft_retry_keywords", [])
    retry_err_pct_max = cfg.get("acc5_soft_retry_error_pct_max", 30.0)
    sev_weight        = cfg.get("sev_weight", {})

    w_vs     = cfg.get("weight_volume_severity", 0.10)
    w_rar    = cfg.get("weight_rarity",     0.35) * (1.0 - w_vs)
    w_sev    = cfg.get("weight_severity",   0.30) * (1.0 - w_vs)
    w_burst  = cfg.get("weight_burstiness", 0.20) * (1.0 - w_vs)
    w_spread = cfg.get("weight_spread",     0.05) * (1.0 - w_vs)

    if not hard_error_kws and not soft_retry_kws:
        return anomaly_df

    hard_re  = re.compile("|".join(hard_error_kws),  re.IGNORECASE) if hard_error_kws  else None
    retry_re = re.compile("|".join(soft_retry_kws),  re.IGNORECASE) if soft_retry_kws  else None

    resolved_eid_col = _resolve_eid_col(anomaly_df, eid_col_name)
    if resolved_eid_col is None:
        print("  [ACC-5] WARNING: cannot resolve EID column — skipping")
        return anomaly_df

    anomaly_df["corrected_by_acc5"] = ""
    n_promoted = 0
    n_demoted  = 0

    tc = cfg["threshold_critical"]
    th = cfg["threshold_high"]
    tm = cfg["threshold_medium"]

    for idx in anomaly_df.index:
        eid = anomaly_df.at[idx, resolved_eid_col]
        msg = str(sample_msg_map.get(eid, "")).strip()
        if not msg:
            continue

        dom_sev   = anomaly_df.at[idx, "dominant_severity"]
        error_pct = anomaly_df.at[idx, "error_pct"]

        # Class A: WARN + hard-error body → promote (keyword pre-filter)
        if hard_re and dom_sev == "WARN" and hard_re.search(msg):
            # S4-8 / MPCD §3.4.1: inject per-row scoring context into cfg so
            # _llm_classify_borderline can build the full context-rich prompt.
            # We use a temporary key 'acc5_row_context' that is overwritten
            # each iteration and removed after the loop — no persistent mutation
            # of the shared cfg dict across calls.
            cfg["acc5_row_context"] = {
                "dominant_severity": dom_sev,
                "count":             anomaly_df.at[idx, "count"]
                                     if "count" in anomaly_df.columns else "unknown",
                "burst_detected":    bool(anomaly_df.at[idx, "burst_detected"])
                                     if "burst_detected" in anomaly_df.columns else False,
                "anomaly_score":     anomaly_df.at[idx, "anomaly_score"]
                                     if "anomaly_score" in anomaly_df.columns else "unknown",
                "error_pct":         error_pct,
            }
            _llm_verdict = _llm_classify_borderline(msg, cfg)
            if _llm_verdict == "benign":
                # LLM says benign — skip promotion
                continue

            err_sev_score = sev_weight.get("ERROR", 1.0)
            new_error_pct = max(float(error_pct), 80.0)

            anomaly_df.at[idx, "dominant_severity"] = "ERROR"
            anomaly_df.at[idx, "error_pct"]         = new_error_pct
            anomaly_df.at[idx, "severity_score"]    = err_sev_score

            rarity    = anomaly_df.at[idx, "rarity_score"]
            burst     = anomaly_df.at[idx, "burstiness_score"]
            spread    = anomaly_df.at[idx, "spread_score"]
            vol_sev   = anomaly_df.at[idx, "volume_severity_score"]
            err_vol   = anomaly_df.at[idx, "error_volume"]
            bonus     = anomaly_df.at[idx, "_singleton_bonus"]

            new_base = (
                err_sev_score * 0.40 +
                err_vol       * 0.35 +
                burst         * 0.15 +
                spread        * 0.05 +
                rarity        * 0.05
            )
            anomaly_df.at[idx, "_base_score"]   = round(new_base, 4)
            anomaly_df.at[idx, "anomaly_score"] = round(min(new_base + bonus, 1.0), 3)
            anomaly_df.at[idx, "corrected_by_acc5"] = "promoted"
            n_promoted += 1

        # Class B: ERROR + soft-retry body + low error_pct → demote (keyword pre-filter)
        elif retry_re and dom_sev == "ERROR" and error_pct < retry_err_pct_max and retry_re.search(msg):
            # S4-8 / MPCD §3.4.1: inject per-row scoring context (same pattern as Class A).
            cfg["acc5_row_context"] = {
                "dominant_severity": dom_sev,
                "count":             anomaly_df.at[idx, "count"]
                                     if "count" in anomaly_df.columns else "unknown",
                "burst_detected":    bool(anomaly_df.at[idx, "burst_detected"])
                                     if "burst_detected" in anomaly_df.columns else False,
                "anomaly_score":     anomaly_df.at[idx, "anomaly_score"]
                                     if "anomaly_score" in anomaly_df.columns else "unknown",
                "error_pct":         error_pct,
            }
            _llm_verdict = _llm_classify_borderline(msg, cfg)
            if _llm_verdict == "error":
                # LLM says genuine error — skip demotion
                continue

            warn_sev_score = sev_weight.get("WARN", 0.5)

            anomaly_df.at[idx, "dominant_severity"] = "WARN"
            anomaly_df.at[idx, "error_pct"]         = 0.0
            anomaly_df.at[idx, "severity_score"]    = warn_sev_score

            rarity  = anomaly_df.at[idx, "rarity_score"]
            burst   = anomaly_df.at[idx, "burstiness_score"]
            spread  = anomaly_df.at[idx, "spread_score"]
            vol_sev = anomaly_df.at[idx, "volume_severity_score"]
            bonus   = anomaly_df.at[idx, "_singleton_bonus"]

            new_base = (
                rarity         * w_rar   +
                warn_sev_score * w_sev   +
                burst          * w_burst +
                spread         * w_spread +
                vol_sev        * w_vs
            )
            anomaly_df.at[idx, "_base_score"]   = round(new_base, 4)
            anomaly_df.at[idx, "anomaly_score"] = round(min(new_base + bonus, 1.0), 3)
            anomaly_df.at[idx, "corrected_by_acc5"] = "demoted"
            n_demoted += 1

    # Clean up the temporary per-row context key so it doesn't linger in cfg
    # between pipeline stages or across multiple run_stage4() calls.
    cfg.pop("acc5_row_context", None)

    corrected_mask = anomaly_df["corrected_by_acc5"] != ""
    if corrected_mask.any():
        anomaly_df.loc[corrected_mask, "anomaly_label"] = _anomaly_label_from_score(
            anomaly_df.loc[corrected_mask, "anomaly_score"], tc, th, tm
        )

    print(f"  [ACC-5] Severity-context correction: "
          f"{n_promoted} WARN→ERROR promoted, {n_demoted} ERROR→WARN demoted")
    return anomaly_df


# ── 4e. ANOMALY SCORING ───────────────────────────────────────────────
def compute_anomaly_score(freq_df, sev_df, temporal_df, source_df,
                          singleton_df, col, cfg,
                          domain_confidence_df=None):
    eid_col = col["event_id"]
    w       = cfg

    _freq_cols = [c for c in [eid_col, "count", "pct_of_total",
                               "rarity_score", "distinct_msg_count", "effective_count"]
                  if c in freq_df.columns]
    anomaly = freq_df[_freq_cols].copy()
    if "effective_count" not in anomaly.columns:
        anomaly["effective_count"] = anomaly["count"]

    # ACC-6: Rarity score floor
    if cfg.get("acc6_enabled", True):
        floor_pct_thresh = cfg.get("acc6_rarity_floor_pct_threshold", 0.50)
        rarity_floor     = cfg.get("acc6_rarity_score_floor", 0.05)
        total_lines      = anomaly["count"].sum()
        dominant_mask    = (anomaly["count"] / max(total_lines, 1)) >= floor_pct_thresh
        n_floored        = dominant_mask.sum()
        if n_floored > 0:
            anomaly.loc[dominant_mask, "rarity_score"] = anomaly.loc[
                dominant_mask, "rarity_score"
            ].clip(lower=rarity_floor)
            print(f"  [ACC-6] Rarity floor applied to {n_floored} cluster(s) "
                  f"(≥{floor_pct_thresh*100:.0f}% of lines, floor={rarity_floor})")

    anomaly = anomaly.merge(
        sev_df[[eid_col, "severity_score", "dominant_severity", "error_pct"]],
        on=eid_col, how="left",
    )

    if len(temporal_df) > 0 and "burst_detected" in temporal_df.columns:
        anomaly = anomaly.merge(
            temporal_df[[eid_col, "burst_detected", "peak_window_count"]],
            on=eid_col, how="left",
        )
        anomaly["burst_detected"]    = anomaly["burst_detected"].fillna(False)
        anomaly["peak_window_count"] = anomaly["peak_window_count"].fillna(0)
        max_peak = anomaly["peak_window_count"].max()
        anomaly["burstiness_score"] = np.where(
            anomaly["burst_detected"],
            0.70 + (anomaly["peak_window_count"] / (max_peak + 1e-9)) * 0.30,
            (anomaly["peak_window_count"] / (max_peak + 1e-9)) * 0.40,
        )

        # ── S4-FP-1: Burstiness gate for 0%-error INFO/DEBUG clusters ─────────
        # A temporal spike in INFO/DEBUG logs with 0 errors is NOT anomalous.
        # Before this gate, burst_detected=True assigned burstiness_score 0.70–1.0
        # which fed into warn_base at weight ~0.20, pushing pure INFO bursts
        # (WebSocket heartbeats, parameter reads, serial port events) past the
        # MEDIUM threshold (0.35). The gate damps the score by 0.30× so their
        # contribution is ~0.06 instead of ~0.20 — firmly LOW territory.
        # An audit column _burst_damped=True is written for dashboard transparency
        # and for S4-FP-4 (FIX-M override) to consume.
        _burst_damping  = cfg.get("burst_no_error_damping", 0.30)
        _dom_sev_fp1    = anomaly.get(
            "dominant_severity", pd.Series("UNKNOWN", index=anomaly.index)
        ).fillna("UNKNOWN")
        _no_error_burst_gate = (
            (anomaly["error_pct"].fillna(0.0) == 0.0) &
            _dom_sev_fp1.isin(["INFO", "DEBUG", "UNKNOWN"])
        )
        anomaly["_burst_damped"] = _no_error_burst_gate & anomaly["burst_detected"].fillna(False)
        anomaly["burstiness_score"] = np.where(
            _no_error_burst_gate,
            anomaly["burstiness_score"] * _burst_damping,
            anomaly["burstiness_score"],
        )
        _n_burst_damped = int(anomaly["_burst_damped"].sum())
        if _n_burst_damped > 0:
            _s4_logger.info(
                "S4-FP-1: burstiness gate — %d INFO/DEBUG 0%%-error clusters "
                "damped by %.0f%%",
                _n_burst_damped, (1 - _burst_damping) * 100,
            )
        # ── end S4-FP-1 ───────────────────────────────────────────────────────
    else:
        anomaly["burst_detected"]    = False
        anomaly["peak_window_count"] = 0
        anomaly["burstiness_score"]  = 0.0
        anomaly["_burst_damped"]     = False  # S4-FP-1: audit column always present

    if len(source_df) > 0 and "spread_score" in source_df.columns:
        anomaly = anomaly.merge(
            source_df[[eid_col, "top_source", "spread_score", "source_spread"]],
            on=eid_col, how="left",
        )
        anomaly["spread_score"]  = anomaly["spread_score"].fillna(0.0)
        anomaly["source_spread"] = anomaly["source_spread"].fillna(1)
        anomaly["top_source"]    = anomaly["top_source"].fillna("unknown")
    else:
        anomaly["spread_score"]  = 0.0
        anomaly["source_spread"] = 1
        anomaly["top_source"]    = "unknown"

    if singleton_df is not None and len(singleton_df) > 0:
        anomaly = anomaly.merge(singleton_df, on=eid_col, how="left")
        anomaly["singleton_class"] = anomaly["singleton_class"].fillna("")
    else:
        anomaly["singleton_class"] = ""

    # S4-4B FIX: merge domain_confidence per cluster so _build_feature_matrix
    # can derive domain_confidence_score.  Without this merge the column is
    # absent from anomaly_df and the ML feature is silently zeroed for every
    # service, degrading ensemble scoring.
    if domain_confidence_df is not None and len(domain_confidence_df) > 0:
        anomaly = anomaly.merge(domain_confidence_df, on=eid_col, how="left")
    if "domain_confidence" not in anomaly.columns:
        anomaly["domain_confidence"] = 0.0
    anomaly["domain_confidence"] = (
        pd.to_numeric(anomaly["domain_confidence"], errors="coerce").fillna(0.0)
    )

    anomaly["rarity_score"]   = anomaly["rarity_score"].fillna(0.5)
    anomaly["severity_score"] = anomaly["severity_score"].fillna(0.1)
    anomaly["error_pct"]      = anomaly["error_pct"].fillna(0.0)

    # FIX-A: percentile-normalised error_volume
    is_error_dominant_mask = anomaly["error_pct"] >= 80.0
    error_dominant_counts  = anomaly.loc[is_error_dominant_mask, "count"]
    pct = cfg.get("error_volume_percentile", 95)
    if len(error_dominant_counts) >= 2:
        p95_error_count = float(np.percentile(error_dominant_counts, pct))
        log_norm_base   = np.log1p(max(p95_error_count, 1.0))
        print(f"  [FIX-A] error_volume normalised against "
              f"p{pct} of ERROR-dominant counts = {p95_error_count:.0f}")
    else:
        log_norm_base = np.log1p(anomaly["count"].max())

    anomaly["error_volume"] = (
        np.log1p(anomaly["count"]) / log_norm_base
    ).clip(upper=1.0)

    is_error_dominant = anomaly["error_pct"] >= 80.0
    error_base = (
        anomaly["severity_score"]   * 0.40 +
        anomaly["error_volume"]     * 0.35 +
        anomaly["burstiness_score"] * 0.15 +
        anomaly["spread_score"]     * 0.05 +
        anomaly["rarity_score"]     * 0.05
    )

    # ACC-1: volume_severity_score blended into warn_base
    w_vs     = cfg.get("weight_volume_severity", 0.10)
    w_rar    = w["weight_rarity"]     * (1.0 - w_vs)
    w_sev    = w["weight_severity"]   * (1.0 - w_vs)
    w_burst  = w["weight_burstiness"] * (1.0 - w_vs)
    w_spread = w["weight_spread"]     * (1.0 - w_vs)

    anomaly["volume_severity_score"] = _compute_volume_severity_score(anomaly, cfg)

    warn_base = (
        anomaly["rarity_score"]          * w_rar   +
        anomaly["severity_score"]        * w_sev   +
        anomaly["burstiness_score"]      * w_burst +
        anomaly["spread_score"]          * w_spread +
        anomaly["volume_severity_score"] * w_vs
    )

    anomaly["_base_score"] = np.where(is_error_dominant, error_base, warn_base)

    bonus_map = w["singleton_bonus"]
    sc = anomaly["singleton_class"]
    ep = anomaly["error_pct"]
    _dom_sev = anomaly.get(
        "dominant_severity",
        pd.Series("INFO", index=anomaly.index),
    )
    # S4-BONUS-FIX: the unseen_variant bonus (+0.05) must only fire when there
    # is genuine severity signal.  INFO/DEBUG clusters with 0% error_pct that
    # land here as "unseen_variant" (because classify_singletons ran on a
    # cold-start run without a known-normal safelist) would otherwise receive
    # the bonus on top of an already-borderline base score and tip into MEDIUM/HIGH.
    # Gate: dominant_severity must be WARN or above, OR error_pct > 0.
    _unseen_has_signal = (
        ~_dom_sev.isin(["INFO", "DEBUG", "UNKNOWN"])
    ) | (ep > 0)
    # S4-FP-3: also block the +0.05 bonus for is_warn_routine clusters.
    # These are already capped by S4-FP-2, but blocking the bonus ensures the
    # pre-cap score is lower and the cap fires less aggressively.
    _is_warn_routine_fp3 = anomaly.get(
        "is_warn_routine", pd.Series(False, index=anomaly.index)
    ).fillna(False).astype(bool)
    anomaly["_singleton_bonus"] = np.where(
        sc == "true_anomaly",
        np.where(ep >= 80.0,
                 bonus_map.get("true_anomaly_error", 0.35),
                 bonus_map.get("true_anomaly_warn",  0.20)),
        np.where(sc == "impossible_attempt_count",
                 bonus_map.get("impossible_attempt_count", 0.25),
        np.where(
            (sc == "unseen_variant") & _unseen_has_signal & ~_is_warn_routine_fp3,  # S4-FP-3
            bonus_map.get("unseen_variant", 0.05),
            0.0,
        ))
    )

    anomaly["anomaly_score"] = (
        anomaly["_base_score"] + anomaly["_singleton_bonus"]
    ).clip(upper=1.0).round(3)

    tc = w["threshold_critical"]
    th = w["threshold_high"]
    tm = w["threshold_medium"]
    anomaly["anomaly_label"] = _anomaly_label_from_score(
        anomaly["anomaly_score"], tc, th, tm
    )

    # ── S4-FP-2: Hard score ceiling for is_warn_routine clusters ──────────
    # Clusters flagged is_warn_routine=True by Stage 3 (S3-FP-2) are WARN-level
    # with 0% errors and no critical keywords. They stay visible at LOW but must
    # never reach MEDIUM/HIGH — they have no genuine error signal to justify it.
    # Ceiling = 0.34, just below MEDIUM threshold (0.35).
    if "is_warn_routine" in anomaly.columns:
        _wr_mask = anomaly["is_warn_routine"].fillna(False).astype(bool)
        _wr_cap  = float(cfg.get("warn_routine_score_cap", 0.34))
        _n_wr_capped = int((_wr_mask & (anomaly["anomaly_score"] > _wr_cap)).sum())
        if _n_wr_capped > 0:
            anomaly.loc[_wr_mask, "anomaly_score"] = (
                anomaly.loc[_wr_mask, "anomaly_score"].clip(upper=_wr_cap)
            )
            anomaly.loc[_wr_mask, "anomaly_label"] = "LOW"
            _s4_logger.info(
                "S4-FP-2: %d is_warn_routine clusters capped at %.2f → LOW",
                _n_wr_capped, _wr_cap,
            )
    # ── end S4-FP-2 ───────────────────────────────────────────────────────

    # ── S4-FP-5: TF-IDF degraded-mode ceiling ────────────────────────────
    # When Stage 3 ran in TF-IDF fallback mode, cluster boundaries are
    # unreliable (TF-IDF groups by shared vocabulary, not semantics). Cap all
    # output at MEDIUM so HIGH/CRITICAL are never fired on unreliable clusters.
    if cfg.get("_tfidf_mode", False):
        _tfidf_max   = cfg.get("tfidf_max_label", "MEDIUM")
        _label_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        _max_ix      = _label_order.index(_tfidf_max) if _tfidf_max in _label_order else 1
        _cap_labels  = _label_order[_max_ix + 1:]
        _score_caps  = {"MEDIUM": 0.59, "HIGH": 0.79}
        _score_cap   = _score_caps.get(_tfidf_max, 0.59)
        if _cap_labels:
            _tfidf_mask = anomaly["anomaly_label"].isin(_cap_labels)
            _n_tfidf    = int(_tfidf_mask.sum())
            anomaly["tfidf_capped"] = _tfidf_mask
            if _n_tfidf > 0:
                anomaly.loc[_tfidf_mask, "anomaly_label"] = _tfidf_max
                anomaly.loc[_tfidf_mask, "anomaly_score"] = (
                    anomaly.loc[_tfidf_mask, "anomaly_score"].clip(upper=_score_cap)
                )
                _s4_logger.warning(
                    "S4-FP-5: TF-IDF mode — %d clusters capped at %s (score≤%.2f)",
                    _n_tfidf, _tfidf_max, _score_cap,
                )
        else:
            anomaly["tfidf_capped"] = False
    # ── end S4-FP-5 ───────────────────────────────────────────────────────

    # ── S4-FP-6: Error-pct gate — kill 0%-error HIGH/MEDIUM false positives ──
    #
    # Root-cause: clusters whose dominant severity is WARN (not ERROR) and whose
    # error_pct is exactly 0 can still accumulate a HIGH/MEDIUM score via the
    # rarity + burstiness + volume_severity terms in warn_base.  This is the
    # primary source of the false positives visible in the dashboard screenshots
    # (HIGH "app" score=60, MEDIUM "express" score=58, both with ERROR%=0%).
    #
    # Gate logic:
    #   IF error_pct == 0.0                           (no error lines at all)
    #   AND dominant_severity NOT in {ERROR, CRITICAL} (pure WARN/INFO/DEBUG)
    #   AND singleton_class NOT in {true_anomaly,      (no hard severity signal)
    #                               impossible_attempt_count}
    #   THEN cap anomaly_score at fp6_zero_error_score_cap  (default 0.34 → LOW)
    #
    # Exemptions (cluster PASSES through unmodified):
    #   • singleton_class == "true_anomaly"           → template matched a known
    #     anomaly keyword; low count but genuine signal — must not be suppressed.
    #   • singleton_class == "impossible_attempt_count" → structural invariant
    #     violation; severity independent of error_pct.
    #   • dominant_severity in {ERROR, CRITICAL}      → error_pct == 0 is
    #     contradictory (ACC-4 will usually fix these), but if they survive
    #     we keep them; a severity-context contradiction is itself a signal.
    #   • cfg["fp6_zero_error_gate_enabled"] == False → kill-switch for operators
    #     who have a use-case where high-rarity 0%-error WARN clusters matter.
    #
    # False-negative safety proof:
    #   • true_anomaly clusters are explicitly whitelisted → no FN for critical
    #     logs with low count but high severity (requirement from spec).
    #   • ERROR/CRITICAL dominant severity clusters are whitelisted → genuine
    #     error clusters that happen to report 0% (e.g. ACC-4 not yet run on
    #     this row) survive the gate.
    #   • The cap is 0.34 (just below MEDIUM threshold 0.35), not 0.0 — the
    #     cluster remains visible at LOW so operators still see it.
    #
    # Audit: writes fp6_capped=True on affected rows for dashboard transparency
    # and for the regression assertion below.
    #
    if cfg.get("fp6_zero_error_gate_enabled", True):
        _fp6_cap = float(cfg.get("fp6_zero_error_score_cap", 0.34))
        # Exempt singleton classes — these carry genuine severity signal
        _fp6_exempt_classes = {
            "true_anomaly",
            "impossible_attempt_count",
        }
        # Exempt severity levels — ERROR/CRITICAL dominant with 0% error_pct
        # is a severity-contradiction signal; do not suppress it.
        _fp6_exempt_severities = {"ERROR", "CRITICAL"}

        _ep_col     = anomaly["error_pct"].fillna(0.0)
        _sc_col     = anomaly["singleton_class"].fillna("")
        _dsev_col   = anomaly.get(
            "dominant_severity",
            pd.Series("UNKNOWN", index=anomaly.index),
        ).fillna("UNKNOWN")

        _fp6_gate_mask = (
            (_ep_col == 0.0) &
            (~_dsev_col.isin(_fp6_exempt_severities)) &
            (~_sc_col.isin(_fp6_exempt_classes)) &
            (anomaly["anomaly_score"] > _fp6_cap)          # only rows that need capping
        )

        _n_fp6 = int(_fp6_gate_mask.sum())
        anomaly["fp6_capped"] = _fp6_gate_mask

        if _n_fp6 > 0:
            anomaly.loc[_fp6_gate_mask, "anomaly_score"] = (
                anomaly.loc[_fp6_gate_mask, "anomaly_score"].clip(upper=_fp6_cap)
            )
            anomaly.loc[_fp6_gate_mask, "anomaly_label"] = "LOW"
            _s4_logger.info(
                "S4-FP-6: error_pct gate — %d 0%%-error non-true-anomaly clusters "
                "capped at %.2f → LOW  (exempt: true_anomaly, impossible_attempt_count, "
                "ERROR/CRITICAL dominant)",
                _n_fp6, _fp6_cap,
            )
            # Debug: log each capped cluster for traceability
            _eid_dbg = col.get("event_id", "event_id")
            for _idx in anomaly.index[_fp6_gate_mask]:
                _s4_logger.debug(
                    "S4-FP-6 capped: %s  sev=%s  rarity=%.2f  burst=%.2f  "
                    "score_before_cap=%.3f  singleton_class=%s",
                    anomaly.at[_idx, _eid_dbg] if _eid_dbg in anomaly.columns else _idx,
                    anomaly.at[_idx, "dominant_severity"] if "dominant_severity" in anomaly.columns else "?",
                    anomaly.at[_idx, "rarity_score"],
                    anomaly.at[_idx, "burstiness_score"],
                    _fp6_cap,    # score_before_cap was already overwritten; use cap as proxy
                    anomaly.at[_idx, "singleton_class"],
                )
        else:
            anomaly["fp6_capped"] = False
            _s4_logger.debug("S4-FP-6: no clusters required capping (all clear)")
    else:
        anomaly["fp6_capped"] = False
        _s4_logger.info("S4-FP-6: zero-error gate disabled via config")
    # ── end S4-FP-6 ───────────────────────────────────────────────────────

    anomaly["singleton_class"] = (
        anomaly["singleton_class"].replace("", None).fillna("normal")
    )
    return anomaly.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


# ── ACC-2: Routine telemetry filter ──────────────────────────────────
def _split_routine_clusters(anomaly_df, cfg):
    if not cfg.get("suppress_routine", True):
        return anomaly_df, pd.DataFrame(columns=anomaly_df.columns)

    # FIX-O: guard against empty DataFrame
    if len(anomaly_df) == 0:
        return anomaly_df, pd.DataFrame(columns=anomaly_df.columns)

    pct_thresh   = cfg.get("routine_count_percentile", 90)
    burst_max    = cfg.get("routine_burst_max", 0.10)
    # ACC-2: Routine telemetry filter — count threshold uses SIGNAL clusters only.
    # Using np.percentile on the full anomaly_df inflates p90 whenever INFO/DEBUG
    # clusters are still present (they are high-volume by nature), raising the bar
    # so high that genuine signal clusters never exceed it and never get suppressed.
    # We compute p90 from WARN/ERROR-dominant clusters only; INFO/DEBUG clusters
    # always fail the dom_sev guard so the threshold value doesn't matter for them,
    # but using them to set the bar was the source of the false-suppression bug.
    _signal_counts_for_pct = anomaly_df.loc[
        ~anomaly_df.get(
            "dominant_severity",
            pd.Series("UNKNOWN", index=anomaly_df.index)
        ).isin({"INFO", "DEBUG"}),
        "count",
    ]
    if len(_signal_counts_for_pct) >= 2:
        count_thresh = float(np.percentile(_signal_counts_for_pct, pct_thresh))
    else:
        count_thresh = float(np.percentile(anomaly_df["count"], pct_thresh))

    routine_sev = {"DEBUG", "INFO"}
    dom_sev     = anomaly_df.get("dominant_severity",
                                  pd.Series("UNKNOWN", index=anomaly_df.index))
    trend_col   = anomaly_df.get("trend_direction",
                                  pd.Series("stable", index=anomaly_df.index))
    burst_col   = anomaly_df.get("burstiness_score",
                                  pd.Series(0.0, index=anomaly_df.index))

    # Original count-based mask: high-volume, stable, low-burst, INFO/DEBUG
    count_routine_mask = (
        dom_sev.isin(routine_sev) &
        (anomaly_df["count"] > count_thresh) &
        (trend_col == "stable") &
        (burst_col < burst_max)
    )

    # S4-ROUTINE-FIX: also suppress any cluster that stage3 explicitly flagged
    # as is_routine=True — these are INFO/DEBUG clusters whose anomaly_signal is
    # "none" regardless of count.  Without this, the p90 count gate leaves 90%
    # of INFO clusters in anomaly_df where they dilute the CRITICAL/HIGH signal
    # and make the domain map look all-red.
    #
    # Crucially, run_stage4 pre-splits is_routine=True clusters *before* scoring,
    # so they never reach this function — this mask is a belt-and-suspenders guard
    # for any that slip through (e.g. when cluster_summary_df was not provided and
    # the flag was not joined).
    s3_routine_mask = pd.Series(False, index=anomaly_df.index)
    if "is_routine" in anomaly_df.columns:
        s3_routine_mask = anomaly_df["is_routine"].fillna(False).astype(bool)
        n_s3_flagged = int(s3_routine_mask.sum())
        if n_s3_flagged:
            print(f"  [ACC-2/S4-ROUTINE] {n_s3_flagged} clusters carry is_routine=True "
                  f"from Stage 3 — added to routine mask unconditionally")

    routine_mask = count_routine_mask | s3_routine_mask

    if "corrected_by_acc4" in anomaly_df.columns:
        routine_mask &= ~anomaly_df["corrected_by_acc4"]

    # FIX-M: Never suppress clusters escalated to MEDIUM or above
    if "anomaly_label" in anomaly_df.columns:
        escalated_mask = anomaly_df["anomaly_label"].isin(["MEDIUM", "HIGH", "CRITICAL"])
        routine_mask  &= ~escalated_mask

    # S4-FP-4: FIX-M interaction guard.
    # FIX-M above protects MEDIUM+ clusters from suppression — correctly.
    # But if a cluster reached MEDIUM ONLY because burst inflation was not fully
    # damped by S4-FP-1 (e.g. a very high peak count on a 0%-error INFO cluster),
    # FIX-M locks in that false positive. We identify these residual cases via the
    # _burst_damped audit column written by S4-FP-1, and allow the count-based
    # routine mask to still suppress them even though they are at MEDIUM.
    # Triple gate: _burst_damped=True AND error_pct==0 AND label==MEDIUM
    # — all three must hold to override FIX-M, preventing accidental suppression
    # of genuine MEDIUM anomalies that happen to have had a burst.
    if "_burst_damped" in anomaly_df.columns:
        _fixm_override = (
            anomaly_df["_burst_damped"].fillna(False).astype(bool) &
            (anomaly_df.get("error_pct", pd.Series(0.0)).fillna(0.0) == 0.0) &
            (anomaly_df.get("anomaly_label", pd.Series("LOW")) == "MEDIUM")
        )
        if _fixm_override.any():
            routine_mask = routine_mask | (count_routine_mask & _fixm_override)
            _s4_logger.info(
                "S4-FP-4: FIX-M override — %d burst-damped 0%%-error MEDIUM "
                "clusters allowed into routine suppression",
                int(_fixm_override.sum()),
            )

    n_routine = routine_mask.sum()
    n_total   = len(anomaly_df)
    print(f"  [ACC-2] Routine filter: {n_routine}/{n_total} clusters suppressed "
          f"(count > p{pct_thresh}={count_thresh:.0f}, stable, "
          f"burstiness < {burst_max}, severity DEBUG/INFO, or is_routine=True from S3)")

    signal_df  = anomaly_df[~routine_mask].reset_index(drop=True)
    routine_df = anomaly_df[routine_mask].reset_index(drop=True)
    return signal_df, routine_df


# ── ACC-3: Stuck-state priority escalation ────────────────────────────
def _apply_stuck_state_escalation(anomaly_df, df_c, col, cfg):
    tokens = cfg.get("stuck_state_tokens", set())
    if not tokens:
        return anomaly_df

    eid_col         = col["event_id"]
    msg_col         = col["message"]
    no_304_escalate = cfg.get("http_304_never_escalate", True)

    if msg_col and msg_col in df_c.columns:
        sample_map = (
            df_c[df_c[eid_col].notna()]
            .groupby(eid_col)[msg_col]
            .first()
        )
    else:
        sample_map = pd.Series(dtype=str)

    warn_mask   = anomaly_df.get("dominant_severity",
                                  pd.Series(dtype=str)) == "WARN"
    warn_counts = anomaly_df.loc[warn_mask, "count"]
    # MEDIUM-22 FIX: add an absolute floor so we never escalate clusters with
    # very few occurrences just because the WARN median is also very low.
    count_threshold = max(
        float(warn_counts.median()) if len(warn_counts) > 0
        else float(np.percentile(anomaly_df["count"], 50)),
        10.0,   # never escalate a MEDIUM cluster seen fewer than 10 times
    )

    pattern = re.compile(
        "|".join(re.escape(t.lower()) for t in tokens), re.IGNORECASE
    )

    candidates = anomaly_df["anomaly_label"] == "MEDIUM"
    n_promoted = 0

    for idx in anomaly_df.index[candidates]:
        row   = anomaly_df.loc[idx]
        eid   = row[eid_col]
        count = row["count"]
        if count < count_threshold:
            continue

        msg = str(sample_map.get(eid, "")).lower()
        if not msg:
            lbl_col = col.get("event_label")
            if lbl_col and lbl_col in anomaly_df.columns:
                msg = str(row.get(lbl_col, "")).lower()

        if no_304_escalate and " 304 " in msg:
            continue

        if pattern.search(msg):
            anomaly_df.at[idx, "anomaly_label"] = "HIGH"
            th = cfg["threshold_high"]
            current_score = anomaly_df.at[idx, "anomaly_score"]
            if current_score < th:
                anomaly_df.at[idx, "anomaly_score"] = round(
                    th + 0.01 + (current_score * 0.05), 3
                )
            n_promoted += 1

    print(f"  [ACC-3/FIX-I] Stuck-state escalation: {n_promoted} MEDIUM → HIGH "
          f"(count_threshold={count_threshold:.0f}, http_304_guard={no_304_escalate})")
    return anomaly_df


# ── FIX-2: Session-check 401 cap ─────────────────────────────────────
def _apply_session_check_401_cap(anomaly_df, sample_msg_map, cfg, eid_col_name=None):
    if not cfg.get("session_check_401_cap", True):
        return anomaly_df

    routes    = cfg.get("session_check_routes", [])
    max_score = cfg.get("session_check_401_max_score", 0.34)
    if not routes:
        return anomaly_df

    route_re      = re.compile("|".join(routes), re.IGNORECASE)
    status_401_re = re.compile(r"\b401\b")

    resolved_eid_col = _resolve_eid_col(anomaly_df, eid_col_name)
    n_capped = 0
    if resolved_eid_col is None:
        print(f"  [FIX-2] Session-check 401 cap: skipped (no EID column resolved)")
        return anomaly_df

    # LOW-23 FIX: vectorised mask instead of row-by-row Python loop
    msg_series = anomaly_df[resolved_eid_col].map(
        lambda eid: str(sample_msg_map.get(eid, "")).strip()
    )
    cap_mask = (
        (anomaly_df["error_pct"] == 0.0) &
        (anomaly_df["anomaly_score"] > max_score) &
        msg_series.str.contains(r"\b401\b", regex=True, na=False) &
        msg_series.str.contains(
            "|".join(routes), regex=True, flags=re.IGNORECASE, na=False
        )
    )
    n_capped = int(cap_mask.sum())
    if n_capped > 0:
        anomaly_df.loc[cap_mask, "anomaly_score"] = max_score
        anomaly_df.loc[cap_mask, "anomaly_label"] = "LOW"

    print(f"  [FIX-2] Session-check 401 cap: {n_capped} cluster(s) capped at LOW "
          f"(max_score={max_score}, requires error_pct=0)")
    return anomaly_df


# ── APPLY POST-SCORING BOOSTS ─────────────────────────────────────────
def apply_post_scoring_boosts(anomaly_df, trend_df, cascade_df, col, cfg,
                               df_c=None, sample_msg_map=None):
    if df_c is None:
        df_c = pd.DataFrame()
    if sample_msg_map is None:
        sample_msg_map = {}

    eid_col = col["event_id"]
    tc = cfg["threshold_critical"]
    th = cfg["threshold_high"]
    tm = cfg["threshold_medium"]

    # FIX-E: Merge trend data
    if trend_df is not None and len(trend_df) > 0 and eid_col in trend_df.columns:
        trend_cols = [c for c in [eid_col, "trend_direction", "trend_ratio"]
                      if c in trend_df.columns]
        anomaly_df = anomaly_df.merge(trend_df[trend_cols], on=eid_col, how="left")
        anomaly_df["trend_direction"] = anomaly_df["trend_direction"].fillna("stable")

        rising_boost = cfg.get("trend_rising_boost", 0.05)
        rising_mask  = (
            (anomaly_df["trend_direction"] == "rising") &
            (anomaly_df["anomaly_label"].isin(["HIGH", "CRITICAL"]))
        )
        n_rising = rising_mask.sum()
        if n_rising > 0:
            anomaly_df.loc[rising_mask, "anomaly_score"] = (
                anomaly_df.loc[rising_mask, "anomaly_score"] + rising_boost
            ).clip(upper=1.0).round(3)
            print(f"  [FIX-E] Rising-trend boost (+{rising_boost}) → {n_rising} events")
    else:
        anomaly_df["trend_direction"] = "stable"

    # FIX-F: Merge cascade data
    if cascade_df is not None and len(cascade_df) > 0 and eid_col in cascade_df.columns:
        cascade_cols = [c for c in [eid_col, "cascade_source", "cascade_target_services"]
                        if c in cascade_df.columns]
        anomaly_df   = anomaly_df.merge(cascade_df[cascade_cols], on=eid_col, how="left")
        cascade_boost = cfg.get("cascade_score_boost", 0.08)
        cascade_mask  = anomaly_df["cascade_source"].notna()
        n_cascade     = cascade_mask.sum()
        if n_cascade > 0:
            anomaly_df.loc[cascade_mask, "anomaly_score"] = (
                anomaly_df.loc[cascade_mask, "anomaly_score"] + cascade_boost
            ).clip(upper=1.0).round(3)
            print(f"  [FIX-F] Cascade boost (+{cascade_boost}) → {n_cascade} events")
    else:
        anomaly_df["cascade_source"]          = None
        anomaly_df["cascade_target_services"] = None

    # FIX-G: Baseline comparison
    anomaly_df = compute_baseline_comparison(anomaly_df, col, cfg)
    gt_boost   = cfg.get("gt_elevated_boost", 0.05)
    elev_mask  = anomaly_df["gt_comparison"] == "elevated"
    n_elev     = elev_mask.sum()
    if n_elev > 0:
        anomaly_df.loc[elev_mask, "anomaly_score"] = (
            anomaly_df.loc[elev_mask, "anomaly_score"] + gt_boost
        ).clip(upper=1.0).round(3)
        print(f"  [FIX-G] Elevated-baseline boost (+{gt_boost}) → {n_elev} events")

    # ACC-3 + FIX-I: Stuck-state escalation
    anomaly_df = _apply_stuck_state_escalation(anomaly_df, df_c, col, cfg)

    # FIX-2: Cap benign session-check 401 clusters
    anomaly_df = _apply_session_check_401_cap(
        anomaly_df, sample_msg_map, cfg, eid_col_name=eid_col
    )

    # ACC-5: Severity-context contradiction corrector (before ACC-4)
    anomaly_df = _apply_severity_context_correction(
        anomaly_df, sample_msg_map, cfg, eid_col_name=eid_col
    )

    # ACC-4: Success-signal severity corrector (after all boosts)
    anomaly_df = _apply_success_signal_corrector(
        anomaly_df, sample_msg_map, cfg, eid_col_name=eid_col
    )

    # HIGH-19 FIX: do NOT blanket re-label here. Each corrector (ACC-3, ACC-4,
    # ACC-5) already re-labels the rows it modifies. A blanket re-label here
    # would silently undo ACC-3 stuck-state escalations that set the label
    # directly by name rather than by score bump.
    # Only re-label rows that had their score changed by FIX-E or FIX-F boosts
    # (trend and cascade) — those are the only boosts in this function that
    # change score without updating label.
    boosted_mask = (
        anomaly_df.get("trend_direction", pd.Series("stable", index=anomaly_df.index)) == "rising"
    ) | anomaly_df.get("cascade_source", pd.Series(dtype=object)).notna()
    if boosted_mask.any():
        anomaly_df.loc[boosted_mask, "anomaly_label"] = _anomaly_label_from_score(
            anomaly_df.loc[boosted_mask, "anomaly_score"], tc, th, tm
        )

    # S4-FP-2 re-enforcement: trend/cascade/GT/stuck-state boosts can push a
    # is_warn_routine cluster back above the 0.34 ceiling applied in
    # compute_anomaly_score. Re-apply the ceiling here so no post-scoring
    # boost can ever elevate a warn_routine cluster to MEDIUM or above.
    if "is_warn_routine" in anomaly_df.columns:
        _wr_re = anomaly_df["is_warn_routine"].fillna(False).astype(bool)
        _wr_cap_re = float(cfg.get("warn_routine_score_cap", 0.34))
        _n_wr_re = int((_wr_re & (anomaly_df["anomaly_score"] > _wr_cap_re)).sum())
        if _n_wr_re > 0:
            anomaly_df.loc[_wr_re, "anomaly_score"] = (
                anomaly_df.loc[_wr_re, "anomaly_score"].clip(upper=_wr_cap_re)
            )
            anomaly_df.loc[_wr_re, "anomaly_label"] = "LOW"
            _s4_logger.info(
                "S4-FP-2 [post_boost re-enforce]: %d warn_routine clusters "
                "re-capped at %.2f after post-scoring boosts",
                _n_wr_re, _wr_cap_re,
            )

    return anomaly_df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_stage4(df, cluster_summary_df=None, cfg=None):
    """
    Stage 4 — Anomaly Scoring.

    Parameters
    ----------
    df : pd.DataFrame
        Per-line classified DataFrame from Stage 3.
    cluster_summary_df : pd.DataFrame, optional
        Cluster summary from Stage 3 (for cluster_label / domain enrichment).
    cfg : dict, optional
        Overrides for STAGE4_CONFIG.

    Returns
    -------
    dict with keys:
        anomaly_df, routine_df, freq_df, sev_df, temporal_df,
        trend_df, cascade_df, source_df, verification_table,
        col_map, config_used, sample_msg_map
    """
    if cfg is None:
        cfg = {}
    global full_cfg  # expose to notebook audit cells that reference the bare name
    full_cfg = {**STAGE4_CONFIG, **cfg}

    # S4-9: Load confirmed anomaly pool written by Stage 5 from the previous run.
    # Clusters whose sample_message matches a confirmed-anomaly template are
    # excluded from the normal training pool, preventing the ML models from
    # learning that past incidents were normal behaviour.
    _confirmed_pool_path = Path(
        full_cfg.get("confirmed_anomaly_pool_path", "models/confirmed_anomaly_pool.json")
    )
    _confirmed_templates: set = set()
    if _confirmed_pool_path.exists():
        try:
            with open(_confirmed_pool_path, "r", encoding="utf-8") as _f:
                _pool_data = json.load(_f)
            _confirmed_templates = set(_pool_data.get("anomaly_templates", []))
            _s4_logger.info(
                "S4-9: loaded %d confirmed anomaly templates from pool (%s)",
                len(_confirmed_templates), _confirmed_pool_path,
            )
        except Exception as _exc:
            _s4_logger.warning(
                "S4-9: could not load confirmed_anomaly_pool (%s) — skipping", _exc
            )
    # Pass the confirmed templates into full_cfg so compute_ml_anomaly_scores
    # can access them when building the normal mask.
    full_cfg["_confirmed_templates"] = _confirmed_templates

    print("Stage 4 — Anomaly Scoring (ACC-5, ACC-6, FIX-L, FIX-M, FIX-N, FIX-O)\n")

    if cluster_summary_df is not None:
        print("[Step 0a] Enriching with cluster_summary (FIX-C)...")
        df = enrich_with_cluster_summary(df, cluster_summary_df)
        print()

    # S4-FP-5: Detect TF-IDF fallback mode from Stage 3 stats.
    # When TF-IDF is active, cluster boundaries are unreliable (vocabulary
    # overlap ≠ semantic similarity). Set _tfidf_mode flag in full_cfg so
    # compute_anomaly_score can apply the degraded-mode ceiling later.
    _tfidf_mode = False
    if cluster_summary_df is not None and not cluster_summary_df.empty:
        if "tfidf_fallback_active" in cluster_summary_df.columns:
            _tfidf_mode = bool(cluster_summary_df["tfidf_fallback_active"].fillna(False).any())
        elif "embedding_model" in cluster_summary_df.columns:
            _tfidf_mode = bool(
                cluster_summary_df["embedding_model"].fillna("").str.contains("tfidf", case=False).any()
            )
    full_cfg["_tfidf_mode"] = _tfidf_mode
    if _tfidf_mode:
        print(
            "\n  ⚠  [S4-FP-5] TF-IDF embedding fallback detected.\n"
            "     Cluster quality is degraded — HIGH and CRITICAL labels are\n"
            "     suppressed to MEDIUM. Install sentence-transformers for full accuracy.\n"
        )
    # Clusters with is_routine=True have anomaly_signal="none" and are INFO/DEBUG
    # baseline activity (health checks, successful logins, stock updates, invoices,
    # etc.).  Letting them enter the scoring pipeline inflates WARN/INFO scores,
    # causes the p90 count threshold in ACC-2 to shift upward (since these
    # high-volume routine clusters raise the percentile), and makes every domain
    # look red because there's no green baseline to dilute it.
    #
    # We extract them here, build a minimal scored row with anomaly_score=0.0
    # and anomaly_label="ROUTINE", and re-join them into routine_df at the end.
    # The scoring path never sees them, so ACC-2's percentile-based threshold
    # is computed only from genuinely anomalous clusters.
    #
    # FALLBACK-PRESPLIT: if is_routine was not joined (e.g. cluster_summary_df
    # absent or join key mismatch), derive it directly from the per-line severity
    # distribution.  A cluster whose rows are entirely INFO or DEBUG with no
    # singleton_class set is indistinguishable from a Stage-3 routine cluster and
    # should be pre-split with the same treatment.  Without this guard, 100% of
    # INFO/DEBUG clusters enter scoring and poison both the S4-3 percentile
    # calibration and the ACC-2 p90 count gate.
    _presplit_routine_df = pd.DataFrame()
    if "is_routine" not in df.columns:
        # Build a per-cluster severity summary on the raw df to derive the flag.
        _sev_col = STAGE4_CONFIG["col_severity"]
        _eid_col = (
            "semantic_cluster_id" if "semantic_cluster_id" in df.columns
            else "event_id" if "event_id" in df.columns
            else None
        )
        if _eid_col and _sev_col in df.columns:
            _sev_norm = df[_sev_col].str.strip().str.upper().replace(
                {"WARNING": "WARN", "E": "ERROR", "W": "WARN",
                 "I": "INFO",  "D": "DEBUG", "e": "ERROR",
                 "w": "WARN",  "i": "INFO"}
            )
            _routine_sev = {"INFO", "DEBUG"}
            _sc_col = STAGE4_CONFIG["col_singleton_class"]
            _has_sc = _sc_col in df.columns

            _cluster_sev = (
                df.assign(_sev_n=_sev_norm)
                .groupby(_eid_col)["_sev_n"]
                .apply(lambda s: s.str.upper().isin({"ERROR", "WARN"}).any())
                .rename("_has_signal_sev")
            )
            if _has_sc:
                _cluster_sc = (
                    df[df[_sc_col].notna() & (df[_sc_col] != "")]
                    .groupby(_eid_col)[_sc_col]
                    .count()
                    .rename("_n_singleton")
                )
                _cluster_meta = _cluster_sev.to_frame().join(
                    _cluster_sc, how="left"
                ).fillna(0)
                _cluster_meta["_is_routine_fallback"] = (
                    ~_cluster_meta["_has_signal_sev"] &
                    (_cluster_meta["_n_singleton"] == 0)
                )
            else:
                _cluster_meta = _cluster_sev.to_frame()
                _cluster_meta["_is_routine_fallback"] = ~_cluster_meta["_has_signal_sev"]

            _routine_eids = set(
                _cluster_meta[_cluster_meta["_is_routine_fallback"]].index.astype(str)
            )
            df = df.copy()
            df["is_routine"] = df[_eid_col].astype(str).isin(_routine_eids)
            n_fallback = int(df["is_routine"].sum())
            print(
                f"[Step 0b-pre] FALLBACK-PRESPLIT: is_routine not joined from "
                f"cluster_summary — derived from per-line severity. "
                f"{n_fallback} INFO/DEBUG-only clusters flagged as routine."
            )

    if "is_routine" in df.columns:
        _routine_flag = df["is_routine"].fillna(False).astype(bool)
        n_presplit = int(_routine_flag.sum())
        if n_presplit > 0:
            print(f"[Step 0b-pre] Pre-splitting {n_presplit} is_routine=True clusters "
                  f"before scoring (S4-ROUTINE-FIX)...")
            _presplit_routine_df = df[_routine_flag].copy()
            # S4-ML-NORMAL-FIX: retain routine rows as the ML normal training pool
            # BEFORE removing them from df.  compute_ml_anomaly_scores needs confirmed-
            # normal rows to train Isolation Forest + AE.  Removing them first left
            # X_normal empty for every service, forcing 100% formula/statistical
            # fallback and zeroing ML ensemble scoring entirely.
            _ml_normal_pool_df = _presplit_routine_df.copy()
            df = df[~_routine_flag].copy()
            print(f"  Remaining for scoring : {len(df)} rows")
            print(f"  ML normal pool        : {len(_ml_normal_pool_df)} routine rows "
                  f"retained for IF/AE training\n")
        else:
            _ml_normal_pool_df = pd.DataFrame()
    else:
        _ml_normal_pool_df = pd.DataFrame()

    # MEDIUM-12 FIX: normalise the cluster ID column name so all downstream
    # resolution is unambiguous and Stage 5 Assert 1 receives a consistent key.
    if "semantic_cluster_id" in df.columns and "event_id" not in df.columns:
        df = df.rename(columns={"semantic_cluster_id": "event_id"})

    print("[Step 0b] Resolving column names...")
    col = resolve_columns(df, full_cfg)

    if col["event_id"] is None:
        raise RuntimeError(
            f"\n\n⚠  Stage 4: cannot resolve a groupby key (event_id).\n"
            f"   Available columns: {list(df.columns)}"
        )
    print()

    print("[Step 1] Filtering noise and parsing timestamps...")
    df_c  = build_clean_df(df, col, full_cfg)
    total = len(df_c)
    n_ev  = df_c[col["event_id"]].nunique() if col["event_id"] else 0
    print(f"  Clean logs  : {total}")
    print(f"  Event types : {n_ev}\n")

    # FIX-O: abort gracefully if nothing to score
    if total == 0:
        print("  ⚠  No non-noise logs remain after filtering — Stage 4 skipped.")
        empty = pd.DataFrame()
        return {
            "anomaly_df": empty, "routine_df": empty,
            "freq_df": empty, "sev_df": empty,
            "temporal_df": empty, "trend_df": empty,
            "cascade_df": empty, "source_df": empty,
            "verification_table": empty,
            "col_map": col, "config_used": full_cfg,
            "sample_msg_map": {},
        }

    eid_col = col["event_id"]
    msg_col = col["message"]
    if msg_col and msg_col in df_c.columns:
        # MEDIUM-21 FIX: use the highest-severity message per cluster, not
        # the first occurrence. This ensures ACC-4/5/FIX-2 see a representative
        # error message rather than a potentially benign INFO/DEBUG first line.
        _sev_rank_map = {"ERROR": 3, "WARN": 2, "INFO": 1, "DEBUG": 0, "UNKNOWN": 0}
        _df_for_map = df_c[df_c[eid_col].notna()].copy()
        _df_for_map["_sev_rank"] = _df_for_map["_severity_norm"].map(_sev_rank_map).fillna(0)
        sample_msg_map = (
            _df_for_map
            .sort_values("_sev_rank", ascending=False)
            .groupby(eid_col)[msg_col]
            .first()
            .to_dict()
        )
        del _df_for_map
    else:
        sample_msg_map = {}

    print("[Step 2] Frequency analysis (FIX-J: distinct_msg_count, ACC-6: rarity floor)...")
    freq_df = compute_frequency(df_c, col)

    # S3.7 — Guard against empty freq_df.
    # compute_frequency() groups by event_id.  If every non-noise row has a
    # null event_id (e.g. all template_ids were None after Stage 3), groupby
    # produces an empty DataFrame.  compute_anomaly_score() then crashes on
    # freq_df["count"].max() and similar operations.
    # Mirror the total == 0 early-exit pattern used above.
    if freq_df.empty:
        print("  ⚠  No scoreable clusters after frequency analysis "
              "(all event_id values are null) — Stage 4 skipped.")
        empty = pd.DataFrame()
        return {
            "anomaly_df": empty, "routine_df": empty,
            "freq_df": freq_df, "sev_df": empty,
            "temporal_df": empty, "trend_df": empty,
            "cascade_df": empty, "source_df": empty,
            "verification_table": empty,
            "col_map": col, "config_used": full_cfg,
            "sample_msg_map": sample_msg_map,
        }

    print("[Step 3] Severity analysis...")
    sev_df = compute_severity(df_c, col, full_cfg)

    print("[Step 4] Temporal analysis (FIX-L: tz guard, FIX-O: edge guards)...")
    temporal_df = compute_temporal(df_c, col, full_cfg)

    print("[Step 4b] Trend direction analysis (FIX-E, FIX-O)...")
    trend_df = compute_trend_direction(df_c, col, full_cfg)

    print("[Step 5] Source distribution...")
    source_df = compute_source(df_c, col)

    singleton_df = None
    if col["event_id"] and col["singleton_class"]:
        sc_col = col["singleton_class"]
        singleton_df = (
            df_c[[eid_col, sc_col]]
            .dropna(subset=[sc_col])
            .drop_duplicates(subset=eid_col)
            .rename(columns={sc_col: "singleton_class"})
        )

    # S4-4B FIX: aggregate domain_confidence per cluster from df_c so it is
    # available on anomaly_df for _build_feature_matrix (domain_confidence_score
    # ML feature).  Use mean across member rows — consistent with Stage 3's
    # cluster_summary aggregation strategy (stage3.py line ~1452).
    domain_confidence_df = None
    if "domain_confidence" in df_c.columns:
        domain_confidence_df = (
            df_c.groupby(eid_col)["domain_confidence"]
            .mean()
            .reset_index()
            .rename(columns={"domain_confidence": "domain_confidence"})
        )
        domain_confidence_df["domain_confidence"] = (
            pd.to_numeric(domain_confidence_df["domain_confidence"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
    else:
        _s4_logger.warning(
            "S4-4B: domain_confidence absent from df_c — "
            "domain_confidence_score feature will be zeroed for all clusters. "
            "Check Stage 2 output."
        )

    print("[Step 6] Anomaly scoring (ACC-6: rarity floor in scoring)...\n")
    anomaly_df = compute_anomaly_score(
        freq_df, sev_df, temporal_df, source_df,
        singleton_df, col, full_cfg,
        domain_confidence_df=domain_confidence_df,
    )

    # FIX-B: WARN singleton guard
    print()
    anomaly_df = _apply_warn_singleton_guard(anomaly_df, df_c, col, full_cfg)

    # Re-compute bonuses after FIX-B downgrade
    bonus_map = full_cfg["singleton_bonus"]
    sc = anomaly_df["singleton_class"]
    ep = anomaly_df["error_pct"]
    anomaly_df["_singleton_bonus"] = np.where(
        sc == "true_anomaly",
        np.where(ep >= 80.0,
                 bonus_map.get("true_anomaly_error", 0.35),
                 bonus_map.get("true_anomaly_warn",  0.20)),
        np.where(sc == "impossible_attempt_count",
                 bonus_map.get("impossible_attempt_count", 0.25),
        np.where(sc == "unseen_variant",
                 bonus_map.get("unseen_variant", 0.05),
                 0.0))
    )
    anomaly_df["anomaly_score"] = (
        anomaly_df["_base_score"] + anomaly_df["_singleton_bonus"]
    ).clip(upper=1.0).round(3)

    tc = full_cfg["threshold_critical"]
    th = full_cfg["threshold_high"]
    tm = full_cfg["threshold_medium"]

    # ── S4-ML-3: ML ensemble scoring ─────────────────────────────────
    # Runs AFTER formula scoring (so formula scores are available as a
    # feature and as the fallback weight), BEFORE threshold calibration
    # and post-scoring boosts (so the final anomaly_score reflects ML).
    #
    # known_normal_tids comes from stage3's config (populated by pipeline.py
    # from the previous run's run_info.json).  If absent, post-deploy grace
    # period detection is skipped (safe — just slightly more false positives
    # in the first run after a deployment).
    _known_normal_tids = full_cfg.get("known_normal_tids", [])
    print("\n[Step 6-ML] ML ensemble anomaly scoring (S4-ML-1/2/3)...")
    try:
        anomaly_df = compute_ml_anomaly_scores(
            anomaly_df, full_cfg,
            known_normal_tids=_known_normal_tids,
            normal_pool_df=_ml_normal_pool_df if not _ml_normal_pool_df.empty else None,
        )
        # Replace anomaly_score with ml_ensemble_score where ML was used
        # (keeps formula_fallback rows on their original formula score).
        ml_mask = anomaly_df["anomaly_source"] != "formula_fallback"
        if ml_mask.any():
            anomaly_df.loc[ml_mask, "anomaly_score"] = (
                anomaly_df.loc[ml_mask, "ml_ensemble_score"]
            )
            _s4_logger.info(
                "S4-ML: ensemble score applied to %d/%d clusters",
                int(ml_mask.sum()), len(anomaly_df),
            )
        n_disagree = int(anomaly_df.get("ensemble_disagreement",
                                        pd.Series([False])).sum())
        if n_disagree:
            print(f"  [S4-ML-6] Ensemble disagreement flagged on "
                  f"{n_disagree} cluster(s) — review recommended.")
        post_deploy_active = (
            anomaly_df.get("anomaly_source", pd.Series([""])) == "post_deploy_caution"
        ).any()
        if post_deploy_active:
            print("  [S4-ML-4] ⚠  Post-deployment grace period ACTIVE — "
                  "ML scores dampened to reduce false positives.")
    except Exception as exc:
        _s4_logger.error(
            "S4-ML: ensemble scoring failed (%s) — "
            "formula scores retained for all clusters.", exc
        )
        # Ensure output columns exist even on failure
        for col_name, default in [
            ("ml_if_score",           0.5),
            ("ml_ae_score",           0.5),
            ("ml_ensemble_score",     anomaly_df["anomaly_score"]),
            ("ensemble_disagreement", False),
            ("anomaly_source",        "formula_fallback"),
            ("ml_confidence",         0.0),
        ]:
            if col_name not in anomaly_df.columns:
                anomaly_df[col_name] = default
    print()

    # S4-3 FIX: Percentile-based threshold calibration.
    # Fixed thresholds (0.80/0.60/0.35) were tuned against one specific log file.
    # On a low-variance log (e.g. replica warnings all cluster ~0.35) almost nothing
    # reaches HIGH. On a noisy gateway log, routine errors flood CRITICAL.
    # Instead, derive thresholds from the actual score distribution, with the fixed
    # values as a fallback when there are too few anomalies to calibrate.
    #
    # SIGNAL-ONLY CALIBRATION: calibrate against clusters whose dominant severity
    # is WARN or ERROR only.  If INFO/DEBUG-dominant clusters survive this far
    # (fallback pre-split missed some or cluster_summary was absent), including
    # them in the calibration pool drags all percentiles down so that genuinely
    # benign INFO clusters end up above MEDIUM threshold.
    _MIN_ANOMALIES_FOR_CALIBRATION = 20
    _signal_mask = anomaly_df.get(
        "dominant_severity", pd.Series("UNKNOWN", index=anomaly_df.index)
    ).isin({"ERROR", "WARN", "UNKNOWN"})
    _scores = anomaly_df.loc[_signal_mask, "anomaly_score"].dropna()
    if len(_scores) < _MIN_ANOMALIES_FOR_CALIBRATION:
        # fall back to all scores if signal-only pool is too small
        _scores = anomaly_df["anomaly_score"].dropna()
    if len(_scores) >= _MIN_ANOMALIES_FOR_CALIBRATION:
        tc_cal = float(np.percentile(_scores, 95))
        th_cal = float(np.percentile(_scores, 85))
        tm_cal = float(np.percentile(_scores, 60))
        # Safety: ensure calibrated thresholds are ordered and won't collapse all
        # events into one bucket (e.g. if all scores are identical).
        _spread = tc_cal - tm_cal
        if _spread >= 0.05 and th_cal > tm_cal and tc_cal > th_cal:
            tc, th, tm = round(tc_cal, 3), round(th_cal, 3), round(tm_cal, 3)
            # Write back so apply_post_scoring_boosts and all correctors pick up
            # the calibrated values without needing explicit parameters.
            full_cfg["threshold_critical"] = tc
            full_cfg["threshold_high"]     = th
            full_cfg["threshold_medium"]   = tm
            print(
                f"  [S4-3] Percentile-calibrated thresholds: "
                f"CRITICAL≥{tc}, HIGH≥{th}, MEDIUM≥{tm} "
                f"(n={len(_scores)}, p95/p85/p60)"
            )
        else:
            print(
                f"  [S4-3] Score distribution too narrow for calibration "
                f"(spread={_spread:.3f}) — using fixed thresholds "
                f"CRITICAL≥{tc}, HIGH≥{th}, MEDIUM≥{tm}"
            )
    else:
        print(
            f"  [S4-3] Too few anomalies for calibration (n={len(_scores)}, "
            f"need≥{_MIN_ANOMALIES_FOR_CALIBRATION}) — using fixed thresholds "
            f"CRITICAL≥{tc}, HIGH≥{th}, MEDIUM≥{tm}"
        )

    anomaly_df["anomaly_label"] = _anomaly_label_from_score(
        anomaly_df["anomaly_score"], tc, th, tm
    )
    # A12-FIX: Write calibrated thresholds as constant columns so validate_pipeline.py
    # can read them back from stage4_anomaly.csv without needing full_cfg.
    # Values are identical on every row — they are run-level constants.
    anomaly_df["threshold_critical"] = tc
    anomaly_df["threshold_high"]     = th
    anomaly_df["threshold_medium"]   = tm
    anomaly_df = anomaly_df.sort_values(
        "anomaly_score", ascending=False
    ).reset_index(drop=True)

    print("\n[Step 6b] Cross-service cascade detection (FIX-F)...")
    cascade_df = compute_cascade_flags(df_c, col, full_cfg)

    print("\n[Step 6c] Post-scoring boosts + correctors "
          "(FIX-E/F/G, ACC-3/FIX-I, FIX-2, ACC-5, ACC-4)...")
    anomaly_df = apply_post_scoring_boosts(
        anomaly_df, trend_df, cascade_df, col, full_cfg,
        df_c=df_c, sample_msg_map=sample_msg_map,
    )

    # ACC-2: Routine suppression (after all correctors, FIX-M guard inside)
    print("\n[Step 6d] Routine telemetry suppression (ACC-2, FIX-M)...")
    anomaly_df, routine_df = _split_routine_clusters(anomaly_df, full_cfg)

    # S4-ROUTINE-FIX: merge the pre-split Stage-3 routine clusters back into
    # routine_df now that scoring is complete.  They get anomaly_score=0.0 and
    # anomaly_label="ROUTINE" so the dashboard can render them in the baseline
    # panel with their true counts intact.
    if not _presplit_routine_df.empty:
        # Assign minimal scoring columns so routine_df has a consistent schema
        _presplit_routine_df = _presplit_routine_df.copy()
        for _col_name, _default in [
            ("anomaly_score",      0.0),
            ("anomaly_label",      "ROUTINE"),
            ("_base_score",        0.0),
            ("_singleton_bonus",   0.0),
            ("burst_detected",     False),
            ("trend_direction",    "stable"),
            ("cascade_source",     None),
            ("corrected_by_acc4",  False),
            ("corrected_by_acc5",  ""),
            ("gt_comparison",      "no_gt"),
        ]:
            if _col_name not in _presplit_routine_df.columns:
                _presplit_routine_df[_col_name] = _default
        routine_df = pd.concat(
            [routine_df, _presplit_routine_df], ignore_index=True
        )
        print(f"  [S4-ROUTINE] Merged {len(_presplit_routine_df)} pre-split "
              f"routine clusters into routine_df "
              f"(total routine: {len(routine_df)})")

    # ── Summary ───────────────────────────────────────────────────────
    label_dist = anomaly_df["anomaly_label"].value_counts()
    print("\nAnomaly distribution (signal clusters only):")
    icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    for lbl, cnt in label_dist.items():
        print(f"  {icons.get(lbl, '  ')} {lbl:<10}: {cnt} events")
    if len(routine_df) > 0:
        print(f"  ⬜ ROUTINE   : {len(routine_df)} events (suppressed)")

    acc4_count = anomaly_df.get("corrected_by_acc4", pd.Series([False])).sum()
    acc5_promo = (anomaly_df.get("corrected_by_acc5", pd.Series([""])) == "promoted").sum()
    acc5_demot = (anomaly_df.get("corrected_by_acc5", pd.Series([""])) == "demoted").sum()
    if acc4_count:
        print(f"\n  ⚠  ACC-4 corrected {acc4_count} clusters from ERROR → INFO severity")
    if acc5_promo or acc5_demot:
        print(f"  ⚠  ACC-5 severity-context: {acc5_promo} promoted, {acc5_demot} demoted")

    # S4-ML summary
    _src_counts = anomaly_df.get("anomaly_source", pd.Series(dtype=str)).value_counts()
    _n_disagree = int(anomaly_df.get("ensemble_disagreement", pd.Series([False])).sum())
    print(f"\n  ML anomaly source breakdown:")
    for src_val, src_cnt in _src_counts.items():
        print(f"    {src_val:<26}: {src_cnt}")
    if _n_disagree:
        print(f"  ⚠  Ensemble disagreement     : {_n_disagree} clusters (IF ≠ AE)")

    # Determine primary anomaly_model_source for dashboard / pipeline.py
    _ensemble_count = int(
        anomaly_df.get("anomaly_source", pd.Series(dtype=str))
        .isin(["ae_if_ensemble", "ml_if_only", "ml_ae_only"])
        .sum()
    )
    _anomaly_model_source = "AE+IF" if _ensemble_count > 0 else "formula"

    print(f"\nTop 15 most anomalous events:")
    hdr = (f"{'Event':<16} {'Score':>6}  {'Label':<10}  "
           f"{'Sev':>5}  {'Err%':>5}  {'Burst':>6}  "
           f"{'Trend':>8}  {'Singleton'}")
    print(hdr)
    print("-" * 90)
    for _, row in anomaly_df.head(15).iterrows():
        burst   = "✓" if row.get("burst_detected", False) else "—"
        trend   = str(row.get("trend_direction", "stable"))[:4]
        print(
            f"{str(row[eid_col]):<16} "
            f"{row['anomaly_score']:>6.3f}  "
            f"{row['anomaly_label']:<10}  "
            f"{str(row.get('dominant_severity', '?')):>5}  "
            f"{row.get('error_pct', 0):>4.1f}%  "
            f"{burst:>6}  "
            f"{trend:>8}  "
            f"{str(row.get('singleton_class', ''))}"
        )

    print(f"\n{'='*60}")
    print(f"Stage 4 complete")
    print(f"  Total signal events   : {len(anomaly_df)}")
    print(f"  CRITICAL              : {(anomaly_df['anomaly_label']=='CRITICAL').sum()}")
    print(f"  HIGH                  : {(anomaly_df['anomaly_label']=='HIGH').sum()}")
    print(f"  MEDIUM                : {(anomaly_df['anomaly_label']=='MEDIUM').sum()}")
    print(f"  LOW                   : {(anomaly_df['anomaly_label']=='LOW').sum()}")
    print(f"  ROUTINE (suppressed)  : {len(routine_df)}")
    print(f"  ACC-4 sev-corrected   : {acc4_count}")
    print(f"  ACC-5 promoted        : {acc5_promo}")
    print(f"  ACC-5 demoted         : {acc5_demot}")
    print(f"  Burst events          : "
          f"{anomaly_df.get('burst_detected', pd.Series([False])).sum()}")
    print(f"  ML ensemble clusters  : "
          f"{int((anomaly_df.get('anomaly_source', pd.Series([''])).isin(['ae_if_ensemble', 'ml_if_only', 'ml_ae_only'])).sum())}")
    print(f"  Ensemble disagreements: {_n_disagree}")
    print(f"  S4-FP-6 capped (0%% err): "
          f"{int(anomaly_df.get('fp6_capped', pd.Series([False])).sum())}")
    print(f"{'='*60}")

    # ── S4-REGRESSION: Post-scoring assertions ────────────────────────────
    # Run after all scoring, boosts, and ceiling fixes are complete.
    # Logs errors but does NOT raise — the pipeline continues regardless.
    _regression_failures = []
    try:
        # Assertion 1: No 0%-error INFO/DEBUG cluster should be MEDIUM or above
        if all(c in anomaly_df.columns for c in ["error_pct", "dominant_severity", "anomaly_label"]):
            _fp1_violations = anomaly_df[
                (anomaly_df["error_pct"].fillna(0.0) == 0.0) &
                anomaly_df["dominant_severity"].isin(["INFO", "DEBUG"]) &
                anomaly_df["anomaly_label"].isin(["MEDIUM", "HIGH", "CRITICAL"])
            ]
            if len(_fp1_violations) > 0:
                msg = (
                    f"S4-REGRESSION [S4-FP-1]: {len(_fp1_violations)} INFO/DEBUG 0%%-error "
                    f"clusters reached MEDIUM+ after all fixes."
                )
                _regression_failures.append(msg)
                _s4_logger.error(msg)

        # Assertion 5 (S4-FP-6): No 0%-error non-ERROR/CRITICAL cluster without
        # a hard severity signal (true_anomaly / impossible_attempt_count) should
        # reach MEDIUM or above when the FP-6 gate is enabled.
        # This is the primary regression guard for the false-positive fix.
        if all(c in anomaly_df.columns for c in ["error_pct", "dominant_severity",
                                                   "anomaly_label", "singleton_class"]):
            if full_cfg.get("fp6_zero_error_gate_enabled", True):
                _fp6_exempt_cls_a = {"true_anomaly", "impossible_attempt_count"}
                _fp6_exempt_sev_a = {"ERROR", "CRITICAL"}
                _fp6_violations_a = anomaly_df[
                    (anomaly_df["error_pct"].fillna(0.0) == 0.0) &
                    (~anomaly_df["dominant_severity"].isin(_fp6_exempt_sev_a)) &
                    (~anomaly_df["singleton_class"].isin(_fp6_exempt_cls_a)) &
                    anomaly_df["anomaly_label"].isin(["MEDIUM", "HIGH", "CRITICAL"])
                ]
                if len(_fp6_violations_a) > 0:
                    msg = (
                        f"S4-REGRESSION [S4-FP-6]: {len(_fp6_violations_a)} 0%%-error "
                        f"non-critical clusters reached MEDIUM+ after the FP-6 gate. "
                        f"Scores: {list(_fp6_violations_a['anomaly_score'].round(3))[:5]}"
                    )
                    _regression_failures.append(msg)
                    _s4_logger.error(msg)

        # Assertion 2: No is_warn_routine cluster should exceed warn_routine_score_cap
        if "is_warn_routine" in anomaly_df.columns:
            _wr_cap_assert = float(full_cfg.get("warn_routine_score_cap", 0.34))
            _fp2_violations = anomaly_df[
                anomaly_df["is_warn_routine"].fillna(False) &
                (anomaly_df["anomaly_score"] > _wr_cap_assert + 1e-4)
            ]
            if len(_fp2_violations) > 0:
                msg = (
                    f"S4-REGRESSION [S4-FP-2]: {len(_fp2_violations)} is_warn_routine "
                    f"clusters exceed cap {_wr_cap_assert}. "
                    f"Max score: {_fp2_violations['anomaly_score'].max():.3f}"
                )
                _regression_failures.append(msg)
                _s4_logger.error(msg)

        # Assertion 3: No cluster should have anomaly_score > 1.0
        _over_one = int((anomaly_df["anomaly_score"] > 1.0 + 1e-6).sum())
        if _over_one > 0:
            msg = f"S4-REGRESSION: {_over_one} clusters have anomaly_score > 1.0"
            _regression_failures.append(msg)
            _s4_logger.error(msg)

        # Assertion 4: TF-IDF mode — no HIGH/CRITICAL should remain
        if full_cfg.get("_tfidf_mode", False):
            _tfidf_max_assert = full_cfg.get("tfidf_max_label", "MEDIUM")
            _label_order_assert = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            _max_ix_assert = _label_order_assert.index(_tfidf_max_assert) \
                if _tfidf_max_assert in _label_order_assert else 1
            _bad_labels_assert = _label_order_assert[_max_ix_assert + 1:]
            if _bad_labels_assert:
                _fp5_count = int(anomaly_df["anomaly_label"].isin(_bad_labels_assert).sum())
                if _fp5_count > 0:
                    msg = (
                        f"S4-REGRESSION [S4-FP-5]: TF-IDF mode active but "
                        f"{_fp5_count} clusters have label above {_tfidf_max_assert}"
                    )
                    _regression_failures.append(msg)
                    _s4_logger.error(msg)

        if not _regression_failures:
            _s4_logger.info("S4-REGRESSION: all assertions passed ✅")
    except Exception as _reg_exc:
        _s4_logger.warning("S4-REGRESSION: assertion check failed unexpectedly: %s", _reg_exc)
    # ── end S4-REGRESSION ─────────────────────────────────────────────────

    return {
        "anomaly_df"        : anomaly_df,
        "routine_df"        : routine_df,
        "freq_df"           : freq_df,
        "sev_df"            : sev_df,
        "temporal_df"       : temporal_df,
        "trend_df"          : trend_df,
        "cascade_df"        : cascade_df,
        "source_df"         : source_df,
        "verification_table": anomaly_df.copy(),  # Issue 9: independent copy, not alias
        "col_map"           : col,
        "config_used"       : full_cfg,
        "sample_msg_map"    : sample_msg_map,
        # S4-ML: new keys consumed by pipeline.py and dashboard
        "ml_stats": {
            "anomaly_source_counts":   _src_counts.to_dict(),
            "ensemble_disagreements":  _n_disagree,
            "anomaly_model_source":    _anomaly_model_source,
            "post_deploy_active":      bool(
                (anomaly_df.get("anomaly_source", pd.Series([""])) == "post_deploy_caution").any()
            ),
            "ml_enabled": (
                full_cfg.get("ml_isolation_forest_enabled", True) or
                full_cfg.get("ml_autoencoder_enabled", True)
            ),
        },
        "regression_failures": _regression_failures,  # S4-REGRESSION
    }


# ══════════════════════════════════════════════════════════════════════
# STANDALONE TEST ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python stage4.py <stage3_output.csv> [cluster_summary.csv]")
        print("       stage3_output.csv must be the CSV produced by stage3.py")
        sys.exit(1)

    from pathlib import Path

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    cluster_summary = None
    if len(sys.argv) >= 3:
        cs_path = Path(sys.argv[2])
        if cs_path.exists():
            cluster_summary = pd.read_csv(cs_path, dtype=str, low_memory=False)
            print(f"Loaded cluster summary from: {cs_path}")
        else:
            print(f"Warning: cluster summary not found: {cs_path}")

    print(f"Loading stage 3 output from: {csv_path}")
    df_s3 = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Restore bool/numeric types
    for col in ("is_noise", "is_merged", "burst_flag", "timestamp_parsed_ok"):
        if col in df_s3.columns:
            df_s3[col] = df_s3[col].map({"True": True, "False": False}).fillna(False)
    for col_name in ("line_no", "repeat_count", "burst_count"):
        if col_name in df_s3.columns:
            df_s3[col_name] = pd.to_numeric(df_s3[col_name], errors="coerce")
    if "timestamp_parsed" in df_s3.columns:
        df_s3["timestamp_parsed"] = pd.to_datetime(df_s3["timestamp_parsed"], errors="coerce", utc=True)

    print(f"Loaded {len(df_s3):,} rows")

    results = run_stage4(df_s3, cluster_summary_df=cluster_summary)

    anomaly_df = results["anomaly_df"]
    print(f"\nAnomaly clusters: {len(anomaly_df)}")

    out_path = csv_path.parent / "stage4_anomaly.csv"
    anomaly_df.to_csv(out_path, index=False)
    print(f"\n✅  stage4_anomaly.csv written to: {out_path}")

    routine_df = results["routine_df"]
    if len(routine_df) > 0:
        routine_path = csv_path.parent / "stage4_routine.csv"
        routine_df.to_csv(routine_path, index=False)
        print(f"✅  stage4_routine.csv written to: {routine_path}")