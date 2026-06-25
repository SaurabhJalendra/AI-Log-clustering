// ─── usePolling.js ────────────────────────────────────────────────────────────
// Custom React hook that owns the entire upload → poll → results state machine.
//
// Usage in dashboard.jsx:
//   import { usePolling } from './usePolling';
//
//   const {
//     apiStatus,       // 'checking' | 'ok' | 'error'
//     loading,         // true while uploading or polling
//     error,           // null | { message, hint, stage }
//     results,         // null | full results object from GET /results
//     stageProgress,   // 'queued' | 'stage_1' ... 'stage_5' | 'complete'
//     elapsedSeconds,  // how long the current stage has been running
//     previousRuns,    // array of past runs from GET /runs
//     handleUpload,    // (File) => void  — call this when user drops a file
//     loadRun,         // (run_id) => void — load a previous run's results
//     deleteRun,       // (run_id) => void — delete a previous run
//     clearError,      // () => void — reset error so user can try again
//     clearResults,    // () => void — go back to the empty/upload state
//   } = usePolling();
// ─────────────────────────────────────────────────────────────────────────────

import { useState, useEffect, useRef, useCallback } from "react";
import {
  checkHealth,
  uploadFile,
  pollStatus,
  getResults,
  listRuns,
  deleteRun as apiDeleteRun,
} from "./api";

// Human-readable label for each pipeline stage
export const STAGE_LABELS = {
  queued:   "Queued — waiting to start...",
  stage_1:  "Stage 1 — Parsing log format",
  stage_2:  "Stage 2 — Mining log templates",
  stage_3:  "Stage 3 — Semantic clustering  (slowest step, ~2 min)",
  stage_4:  "Stage 4 — Anomaly scoring",
  stage_5:  "Stage 5 — Root cause analysis",
  complete: "Complete",
};

const POLL_INTERVAL_MS = 3000; // poll every 3 seconds

export function usePolling() {
  // ── Core state (Section 3 of build doc) ───────────────────────────────────
  const [apiStatus,     setApiStatus]     = useState("checking"); // 'checking'|'ok'|'error'
  const [runId,         setRunId]         = useState(null);
  const [results,       setResults]       = useState(null);
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState(null);       // null | { message, hint, stage }

  // ── Stage progress (Section 7.1 — stage-aware spinner) ────────────────────
  const [stageProgress,   setStageProgress]   = useState(null);  // current stage string
  const [stageStartedAt,  setStageStartedAt]  = useState(null);  // ISO timestamp of current stage
  const [elapsedSeconds,  setElapsedSeconds]  = useState(0);

  // ── Previous runs (Section 4.3) ───────────────────────────────────────────
  const [previousRuns, setPreviousRuns] = useState([]);

  // ── Internal refs ──────────────────────────────────────────────────────────
  const pollIntervalRef  = useRef(null);  // holds the setInterval id
  const elapsedTimerRef  = useRef(null);  // holds the elapsed-seconds ticker
  const currentRunIdRef  = useRef(null);  // stable ref so interval closure can read it

  // ══════════════════════════════════════════════════════════════════════════
  // HELPERS
  // ══════════════════════════════════════════════════════════════════════════

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current)  clearInterval(pollIntervalRef.current);
    if (elapsedTimerRef.current)  clearInterval(elapsedTimerRef.current);
    pollIntervalRef.current = null;
    elapsedTimerRef.current = null;
  }, []);

  // Start a 1-second ticker so the UI can show "Elapsed: Xs"
  const startElapsedTicker = useCallback((stageStartISO) => {
    if (elapsedTimerRef.current) clearInterval(elapsedTimerRef.current);
    elapsedTimerRef.current = setInterval(() => {
      if (!stageStartISO) return;
      const secs = Math.round((Date.now() - new Date(stageStartISO)) / 1000);
      setElapsedSeconds(secs);
    }, 1000);
  }, []);

  // Update stage progress display from a status response
  const applyStageProgress = useCallback((statusData) => {
    const stage = statusData.stage_progress || "queued";
    setStageProgress(stage);

    // Find the started_at timestamp for the current stage
    const startKey = `${stage}_started_at`;
    const startISO = statusData[startKey] || null;
    setStageStartedAt(startISO);
    startElapsedTicker(startISO);
  }, [startElapsedTicker]);

  // ══════════════════════════════════════════════════════════════════════════
  // 1. HEALTH CHECK — runs once on mount (Section 4.1)
  // ══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    checkHealth().then((ok) => setApiStatus(ok ? "ok" : "error"));
  }, []);

  // ══════════════════════════════════════════════════════════════════════════
  // 2. LOAD PREVIOUS RUNS — runs once on mount (Section 4.3)
  // ══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    listRuns()
      .then(setPreviousRuns)
      .catch(() => setPreviousRuns([])); // silently fail — not critical
  }, []);

  // ══════════════════════════════════════════════════════════════════════════
  // 3. CLEANUP — stop all intervals when the component unmounts
  // ══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  // ══════════════════════════════════════════════════════════════════════════
  // 4. HANDLE FILE UPLOAD — two-step upload + poll (Section 4.2)
  // ══════════════════════════════════════════════════════════════════════════

  const handleUpload = useCallback(async (file) => {
    // Reset everything before starting a new run
    stopPolling();
    setLoading(true);
    setError(null);
    setResults(null);
    setRunId(null);
    setStageProgress("queued");
    setElapsedSeconds(0);

    // ── Step 1: Upload the file ──────────────────────────────────────────
    let newRunId;
    try {
      const data = await uploadFile(file);
      newRunId = data.run_id;
      setRunId(newRunId);
      currentRunIdRef.current = newRunId;
    } catch (e) {
      setError({ message: e.message, hint: "", stage: "upload" });
      setLoading(false);
      return;
    }

    // ── Step 2: Poll for completion ──────────────────────────────────────
    pollIntervalRef.current = setInterval(async () => {
      try {
        const statusData = await pollStatus(currentRunIdRef.current);

        // Update the stage progress display on every poll
        applyStageProgress(statusData);

        if (statusData.status === "complete") {
          stopPolling();
          try {
            const resultData = await getResults(currentRunIdRef.current);
            setResults(resultData);
          } catch (e) {
            setError({ message: "Pipeline finished but results could not be loaded.", hint: e.message, stage: "results" });
          }
          setLoading(false);
          // Refresh previous runs list so new run appears
          listRuns().then(setPreviousRuns).catch(() => {});

        } else if (statusData.status === "failed") {
          stopPolling();
          // Section 4.5 — richer error display
          setError({
            message: statusData.error_message || "Pipeline failed.",
            hint:    statusData.error_hint    || "",
            stage:   statusData.stage_progress || "unknown",
          });
          setLoading(false);
        }
        // If still 'processing' — do nothing, poll again in 3 seconds

      } catch (e) {
        // Network error mid-poll
        stopPolling();
        setError({ message: "Lost connection to API during processing.", hint: e.message, stage: stageProgress || "unknown" });
        setLoading(false);
      }
    }, POLL_INTERVAL_MS);

  }, [stopPolling, applyStageProgress, stageProgress]);

  // ══════════════════════════════════════════════════════════════════════════
  // 5. LOAD A PREVIOUS RUN (Section 4.3)
  // ══════════════════════════════════════════════════════════════════════════

  const loadRun = useCallback(async (id) => {
    stopPolling();
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await getResults(id);
      setResults(data);
      setRunId(id);
    } catch (e) {
      setError({ message: `Could not load run ${id}.`, hint: e.message, stage: "" });
    }
    setLoading(false);
  }, [stopPolling]);

  // ══════════════════════════════════════════════════════════════════════════
  // 6. DELETE A PREVIOUS RUN (Section 4.4)
  // ══════════════════════════════════════════════════════════════════════════

  const deleteRun = useCallback(async (id) => {
    try {
      await apiDeleteRun(id);
      // If we're currently viewing the deleted run, clear the results
      if (id === currentRunIdRef.current) {
        setResults(null);
        setRunId(null);
      }
      // Refresh the list
      listRuns().then(setPreviousRuns).catch(() => {});
    } catch (e) {
      setError({ message: `Could not delete run ${id}.`, hint: e.message, stage: "" });
    }
  }, []);

  // ══════════════════════════════════════════════════════════════════════════
  // 7. UI HELPERS
  // ══════════════════════════════════════════════════════════════════════════

  const clearError   = useCallback(() => setError(null),   []);
  const clearResults = useCallback(() => {
    setResults(null);
    setRunId(null);
    setStageProgress(null);
    setElapsedSeconds(0);
  }, []);

  // ══════════════════════════════════════════════════════════════════════════
  // RETURN — everything dashboard.jsx needs
  // ══════════════════════════════════════════════════════════════════════════

  return {
    // Status
    apiStatus,
    loading,
    error,
    results,
    // Stage progress (for the spinner)
    stageProgress,
    stageStartedAt,
    elapsedSeconds,
    // Previous runs
    previousRuns,
    runId,
    // Actions
    handleUpload,
    loadRun,
    deleteRun,
    clearError,
    clearResults,
  };
}