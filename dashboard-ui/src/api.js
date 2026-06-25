// ─── api.js ──────────────────────────────────────────────────────────────────
// All communication with the backend lives here.
// No component should ever call fetch() directly — import these functions instead.
//
// Usage:
//   import { uploadFile, pollStatus, getResults, listRuns, deleteRun } from './api';
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ─── Health ───────────────────────────────────────────────────────────────────

/**
 * GET /health
 * Returns true if the backend is reachable and healthy, false otherwise.
 * Call this once on mount to set apiStatus.
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

// ─── Upload ───────────────────────────────────────────────────────────────────

/**
 * POST /analyze
 * Uploads a log file and immediately gets back a run_id.
 * The pipeline runs async — this call returns in ~1 second.
 *
 * @param {File} file  — the File object from the drag-drop or <input>
 * @returns {{ run_id: string }}
 * @throws Error with a human-readable message on failure
 */
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  // IMPORTANT: do NOT set Content-Type manually.
  // The browser must set it so the boundary string is included.

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let detail = `Upload failed (${res.status})`;
    try {
      const body = await res.json();
      detail = body.detail || body.message || detail;
    } catch {}
    throw new Error(detail);
  }

  return res.json(); // { run_id: "2026-04-23_..." }
}

// ─── Status polling ───────────────────────────────────────────────────────────

/**
 * GET /runs/{run_id}/status
 * Returns the current pipeline status for a run.
 *
 * @param {string} runId
 * @returns {{
 *   status: 'processing' | 'complete' | 'failed',
 *   stage_progress: 'queued' | 'stage_1' | 'stage_2' | 'stage_3' | 'stage_4' | 'stage_5' | 'complete',
 *   stage_1_started_at?: string,  // ISO timestamps — one per stage
 *   stage_2_started_at?: string,
 *   stage_3_started_at?: string,
 *   stage_4_started_at?: string,
 *   stage_5_started_at?: string,
 *   completed_at?: string,
 *   error_message?: string,       // present when status === 'failed'
 *   error_hint?: string,
 *   failed_at?: string,
 * }}
 */
export async function pollStatus(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}/status`);
  if (!res.ok) throw new Error(`Status check failed (${res.status})`);
  return res.json();
}

// ─── Results ──────────────────────────────────────────────────────────────────

/**
 * GET /runs/{run_id}/results
 * Fetches the full analysis output once status === 'complete'.
 *
 * @param {string} runId
 * @returns {{
 *   run_id: string,
 *   filename: string,
 *   anomaly_count: number,
 *   incident_count: number,
 *   anomalies: Array<object>,
 *   incidents: Array<object>,
 *   run_info: {
 *     total_lines: number,
 *     parsed_ok: number,
 *     noise_lines: number,
 *     detected_encoding: string,
 *     unique_templates: number,
 *     total_clusters: number,
 *     anomaly_label_counts: object,
 *     incident_severity_counts: object,
 *     stage_1_started_at: string,
 *     completed_at: string,
 *     services_affected: string[],   // ← ARRAY, not pipe-delimited string
 *     recurring_services: string[],
 *     default_date_inferred: boolean,
 *   }
 * }}
 */
export async function getResults(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}/results`);
  if (!res.ok) throw new Error(`Could not fetch results (${res.status})`);
  return res.json();
}

// ─── Previous runs ────────────────────────────────────────────────────────────

/**
 * GET /runs
 * Returns a list of all previous runs (most recent first).
 * Each item is a run_info object — same shape as results.run_info.
 *
 * @returns {Array<{ run_id: string, filename: string, anomaly_count: number, ... }>}
 */
export async function listRuns() {
  const res = await fetch(`${API_BASE}/runs`);
  if (!res.ok) throw new Error(`Could not list runs (${res.status})`);
  return res.json();
}

/**
 * DELETE /runs/{run_id}
 * Permanently deletes a run and its output files from the server.
 * Call listRuns() again after this to refresh the UI.
 *
 * @param {string} runId
 * @returns {boolean} true on success
 */
export async function deleteRun(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`Delete failed (${res.status})`);
  return true;
}
// ─── Feedback (Ground Truth) ──────────────────────────────────────────────────

/**
 * POST /feedback
 * Submit a True Positive or False Positive label for a cluster.
 * If the cluster already has a label it is overwritten (user can change mind).
 *
 * @param {object} payload
 * @param {string} payload.cluster_ref      — e.g. "SC8DE1D88E8BB1"
 * @param {'true_positive'|'false_positive'} payload.label
 * @param {string} [payload.template_id]
 * @param {string} [payload.run_id]
 * @param {string} [payload.service]
 * @param {string} [payload.log_template]
 * @param {string} [payload.severity]
 * @param {number} [payload.anomaly_score]
 * @returns {object} the saved ground truth record
 */
export async function submitFeedback(payload) {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = `Feedback failed (${res.status})`;
    try { const b = await res.json(); detail = b.detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

/**
 * GET /feedback/{cluster_ref}
 * Get the current label for a single cluster, or null if unlabelled.
 *
 * @param {string} clusterRef
 * @returns {object|null}
 */
export async function getFeedback(clusterRef) {
  const res = await fetch(`${API_BASE}/feedback/${clusterRef}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`getFeedback failed (${res.status})`);
  return res.json();
}

/**
 * DELETE /feedback/{cluster_ref}
 * Remove a label so the cluster is treated as unlabelled again.
 *
 * @param {string} clusterRef
 * @returns {boolean}
 */
export async function deleteFeedback(clusterRef) {
  const res = await fetch(`${API_BASE}/feedback/${clusterRef}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`deleteFeedback failed (${res.status})`);
  return true;
}

/**
 * GET /feedback/stats
 * Returns { total, true_positive, false_positive } counts.
 */
export async function getFeedbackStats() {
  const res = await fetch(`${API_BASE}/feedback/stats`);
  if (!res.ok) throw new Error(`getFeedbackStats failed (${res.status})`);
  return res.json();
}