// ─── DASHBOARD — ROOT ORCHESTRATOR ───────────────────────────────────────────
// This file owns:
//   • Theme context + toggle
//   • Sidebar navigation
//   • Top-level app state (upload, polling, page routing)
//   • Shared inline components that only live here (FileUpload, StageSpinner,
//     ErrorBox, WarningBanners, RunInfoPanel, PreviousRunsPanel)
//
// All visual constants (font sizes, colours, opacity) come from ./theme.js
// All page-level components are imported from their own files.

import { useState, useEffect, useContext, createContext } from "react";
import { usePolling, STAGE_LABELS } from "./usePolling";

// ── Theme constants ────────────────────────────────────────────────────────────
import { READABILITY_STYLE, FONT_SIZE, FONT, SIGNAL_COLOR, OPACITY } from "./Theme";

// ── Shared UI ──────────────────────────────────────────────────────────────────
import { Badge, ClickableCard } from "./Components";

// ── Chart imports still used in RunInfoPanel ──────────────────────────────────
import AnomalyTimeline from "./charts/AnomalyTimeline";

// ── Page imports ──────────────────────────────────────────────────────────────
import OverviewPage        from "./pages/OverviewPage";
import AnomaliesPage       from "./pages/AnomaliesPage";
import ClusterExplorerPage from "./pages/ClusterExplorerPage";
import LogStreamPage       from "./pages/LogStreamPage";
import IncidentsPage       from "./pages/IncidentsPage";
import LogIntelligencePage from "./pages/LogIntelligencePage";
import TemplateIntelPage   from "./pages/TemplateIntelPage";

// ─── READABILITY STYLE INJECTION ──────────────────────────────────────────────

function ReadabilityStyles() {
  return <style>{READABILITY_STYLE}</style>;
}

// ─── THEME CONTEXT ────────────────────────────────────────────────────────────

const ThemeContext = createContext({ dark: false, toggle: () => {} });

function ThemeProvider({ children }) {
  const [dark, setDark] = useState(() => {
    try { return localStorage.getItem("skyai_theme") === "dark"; } catch { return false; }
  });

  useEffect(() => {
    document.body.classList.toggle("dark", dark);
    try { localStorage.setItem("skyai_theme", dark ? "dark" : "light"); } catch {}
  }, [dark]);

  const toggle = () => setDark(d => !d);
  return <ThemeContext.Provider value={{ dark, toggle }}>{children}</ThemeContext.Provider>;
}

function ThemeToggle() {
  const { dark, toggle } = useContext(ThemeContext);
  return (
    <button className="theme-toggle" onClick={toggle} title={dark ? "Switch to light mode" : "Switch to dark mode"}>
      {dark ? "☀" : "☾"}
    </button>
  );
}

// ─── FILE UPLOAD ──────────────────────────────────────────────────────────────

function FileUpload({ onUpload, disabled }) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) onUpload(file);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "60vh" }}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${dragging ? "var(--accent-blue)" : "var(--border-default)"}`,
          borderRadius: 12, padding: "48px 64px", textAlign: "center",
          background: dragging ? "var(--surface-info)" : "var(--bg-card)",
          transition: "all 0.2s", cursor: disabled ? "not-allowed" : "pointer",
          opacity: disabled ? OPACITY.dim : OPACITY.full, maxWidth: 480, width: "100%",
        }}
      >
        <div style={{ fontSize: 36, marginBottom: 16 }}>📂</div>
        <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-primary)", marginBottom: 8 }}>Drop a log file here</div>
        <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", marginBottom: 24 }}>or click to browse</div>
        <input
          type="file" accept=".log,.txt"
          disabled={disabled}
          onChange={(e) => e.target.files[0] && onUpload(e.target.files[0])}
          style={{ display: "none" }}
          id="file-input"
        />
        <label htmlFor="file-input" style={{
          background: "var(--surface-info)", border: "1px solid var(--accent-blue)", borderRadius: 6,
          color: "var(--accent-blue)", padding: "8px 24px", fontSize: FONT_SIZE.lg,
          cursor: disabled ? "not-allowed" : "pointer",
        }}>
          Browse files
        </label>
      </div>
      <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-muted)", marginTop: 16 }}>Supports .log and .txt files</div>
    </div>
  );
}

// ─── STAGE PROGRESS SPINNER ───────────────────────────────────────────────────

function StageSpinner({ stageProgress, elapsedSeconds }) {
  const stages     = ["stage_1", "stage_2", "stage_3", "stage_4", "stage_5"];
  const currentIdx = stages.indexOf(stageProgress);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "60vh", gap: 24 }}>
      <div style={{ fontSize: FONT_SIZE.lg, color: "var(--accent-blue)", fontFamily: "monospace", marginBottom: 8 }}>
        {STAGE_LABELS[stageProgress] || "Processing..."}
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        {stages.map((s, i) => (
          <div key={s} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{
              width: i === currentIdx ? 12 : 8,
              height: i === currentIdx ? 12 : 8,
              borderRadius: "50%",
              background: i < currentIdx ? "var(--accent-green)" : i === currentIdx ? "var(--accent-blue)" : "var(--border-default)",
              border: i === currentIdx ? "2px solid var(--accent-blue)" : "none",
              transition: "all 0.3s",
              boxShadow: i === currentIdx ? "0 0 8px var(--accent-blue)" : "none",
            }} />
            {i < stages.length - 1 && (
              <div style={{ width: 24, height: 1, background: i < currentIdx ? "var(--accent-green)" : "var(--border-default)" }} />
            )}
          </div>
        ))}
      </div>
      <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-muted)", fontFamily: "monospace" }}>
        {`Elapsed: ${elapsedSeconds}s`}
      </div>
      <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-muted)", opacity: OPACITY.muted }}>
        Do not close this tab — the pipeline is running on the server
      </div>
    </div>
  );
}

// ─── ERROR BOX ────────────────────────────────────────────────────────────────

function ErrorBox({ error, onRetry }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "60vh" }}>
      <div style={{
        background: "var(--surface-error)", border: "1px solid var(--surface-error-border)", borderRadius: 10,
        padding: "24px 32px", maxWidth: 520, width: "100%",
      }}>
        <div style={{ fontSize: FONT_SIZE.lg, color: "var(--accent-red)", fontWeight: 700, marginBottom: 12 }}>
          ⛔ Pipeline Error {error.stage ? `— failed at ${error.stage}` : ""}
        </div>
        <div style={{ fontSize: FONT_SIZE.lg, color: "var(--text-secondary)", marginBottom: error.hint ? 12 : 0, lineHeight: 1.6 }}>
          {error.message}
        </div>
        {error.hint && (
          <pre style={{
            fontSize: FONT_SIZE.base, color: "var(--text-muted)", background: "var(--bg-card-deep)",
            border: "1px solid var(--border-default)", borderRadius: 6,
            padding: "10px 12px", overflow: "auto", marginBottom: 0,
          }}>
            {error.hint}
          </pre>
        )}
        <button onClick={onRetry} style={{
          marginTop: 20, background: "var(--surface-info)", border: "1px solid var(--accent-green)",
          borderRadius: 6, color: "var(--accent-green)", padding: "8px 20px",
          fontSize: FONT_SIZE.md, cursor: "pointer",
        }}>
          ↺ Try again
        </button>
      </div>
    </div>
  );
}

// ─── WARNING BANNERS ──────────────────────────────────────────────────────────

function WarningBanners({ results }) {
  const runInfo  = results?.run_info || {};
  const anomalies = results?.anomalies || [];

  const showDateWarning  = runInfo.default_date_inferred === false;
  const otherDomainPct   = anomalies.length > 0
    ? Math.round(anomalies.filter(a => a.domain === "other").length / anomalies.length * 100) : 0;
  const showDomainWarning = otherDomainPct > 50;

  if (!showDateWarning && !showDomainWarning) return null;

  return (
    <div style={{ marginBottom: 20 }}>
      {showDateWarning && (
        <div style={{
          background: "var(--surface-warn)", border: "1px solid var(--surface-warn-border)",
          borderRadius: 6, padding: "10px 14px", marginBottom: 8,
          fontSize: FONT_SIZE.md, color: "var(--accent-yellow)",
        }}>
          ⚠ Log file has no datable lines in the first 500 rows. Timestamps may be inaccurate — install chardet for better detection.
        </div>
      )}
      {showDomainWarning && (
        <div style={{
          background: "var(--surface-info)", border: "1px solid var(--surface-info-border)",
          borderRadius: 6, padding: "10px 14px",
          fontSize: FONT_SIZE.md, color: "var(--accent-blue)",
        }}>
          ℹ {otherDomainPct}% of clusters were classified as "other" domain. Domain detection works best when log messages contain service-specific keywords.
        </div>
      )}
    </div>
  );
}

// ─── RUN INFO SUMMARY PANEL ───────────────────────────────────────────────────

function RunInfoPanel({ results, onNavigate }) {
  const ri    = results?.run_info || {};
  const tiles = [
    { label: "Total Lines",    value: (ri.total_lines || 0).toLocaleString(),      nav: null },
    { label: "Parsed OK",      value: (ri.parsed_ok || 0).toLocaleString(),        nav: null },
    { label: "Noise Filtered", value: (ri.noise_lines || 0).toLocaleString(),      nav: null },
    { label: "Encoding",       value: ri.detected_encoding || "—",                 nav: null },
    { label: "Templates",      value: (ri.unique_templates || 0).toLocaleString(), nav: null },
    { label: "Clusters",       value: (ri.total_clusters || 0).toLocaleString(),   nav: null },
    { label: "Anomalies",      value: results?.anomaly_count || 0,                 nav: "anomalies" },
    { label: "Incidents",      value: results?.incident_count || 0,                nav: "incidents" },
  ];

  const tileStyle = {
    background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "12px 14px",
  };

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 10, marginBottom: 10 }}>
        {tiles.map(({ label, value, nav }) => {
          const inner = (
            <>
              <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
              <div style={{ fontSize: FONT_SIZE.h2, fontWeight: 700, color: "var(--text-primary)", fontFamily: FONT.mono }}>{value}</div>
            </>
          );
          return nav && onNavigate
            ? <ClickableCard key={label} onClick={() => onNavigate(nav)} style={tileStyle}>{inner}</ClickableCard>
            : <div key={label} style={tileStyle}>{inner}</div>;
        })}
      </div>
      {ri.recurring_services?.length > 0 && (
        <div style={{ background: "var(--surface-warn)", border: "1px solid var(--surface-warn-border)", borderRadius: 6, padding: "8px 14px", fontSize: FONT_SIZE.md, color: "var(--accent-yellow)" }}>
          ⚠ Recurring services detected: {ri.recurring_services.join(", ")}
        </div>
      )}
    </div>
  );
}

// ─── PREVIOUS RUNS PANEL ─────────────────────────────────────────────────────

function PreviousRunsPanel({ runs, onLoad, onDelete, currentRunId }) {
  if (!runs || runs.length === 0) return null;
  return (
    <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", marginBottom: 20 }}>
      <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", marginBottom: 12, letterSpacing: "0.05em", textTransform: "uppercase" }}>
        Previous Runs
      </div>
      {runs.slice(0, 8).map((run) => {
        const rid       = run.run_id;
        const isCurrent = rid === currentRunId;
        return (
          <div key={rid} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: "1px solid var(--border-subtle)" }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: FONT_SIZE.md, color: isCurrent ? "var(--accent-blue)" : "var(--text-secondary)", fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {run.log_filename || rid}
              </div>
              <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-muted)" }}>
                {run.anomaly_count ?? "?"} anomalies · {run.incident_count ?? "?"} incidents
              </div>
            </div>
            <button onClick={() => onLoad(rid)} style={{ background: "var(--surface-info)", border: "1px solid var(--surface-info-border)", borderRadius: 4, color: "var(--accent-blue)", padding: "3px 10px", fontSize: FONT_SIZE.base, cursor: "pointer" }}>Load</button>
            <button onClick={() => onDelete(rid)} style={{ background: "var(--surface-error)", border: "1px solid var(--surface-error-border)", borderRadius: 4, color: "var(--accent-red)", padding: "3px 10px", fontSize: FONT_SIZE.base, cursor: "pointer" }}>Delete</button>
          </div>
        );
      })}
    </div>
  );
}

// ─── PAGES CONFIG ─────────────────────────────────────────────────────────────

const PAGES = [
  { id: "overview",   label: "Overview" },
  { id: "anomalies",  label: "Anomalies" },
  { id: "clusters",   label: "Cluster Explorer" },
  { id: "stream",     label: "Log Stream" },
  { id: "incidents",  label: "Incidents" },
  { id: "logintel",   label: "Log Intelligence" },
  { id: "templates",  label: "Template Intelligence" },
];

// ─── ROOT APP ─────────────────────────────────────────────────────────────────

function AppInner() {
  const [page, setPage] = useState("overview");

  const {
    apiStatus, loading, error, results,
    stageProgress, elapsedSeconds,
    previousRuns, runId,
    handleUpload, loadRun, deleteRun, clearError, clearResults,
  } = usePolling();

  const incidentCount = results?.incident_count || 0;
  const criticalCount = results?.run_info?.incident_severity_counts?.CRITICAL || 0;
  const runTimestamp  = results?.run_info?.run_timestamp || results?.run_info?.completed_at || null;

  return (
    <div style={{ display: "flex", height: "100vh", background: "var(--bg-app)", color: "var(--text-primary)", fontFamily: FONT.mono }}>

      {/* ── Sidebar ── */}
      <div style={{ width: 200, background: "var(--bg-sidebar)", borderRight: "1px solid var(--border-subtle)", display: "flex", flexDirection: "column", flexShrink: 0 }}>
        <div style={{ padding: "18px 16px", borderBottom: "1px solid var(--border-subtle)" }}>
          <div style={{ fontSize: FONT_SIZE.xl, fontWeight: 700, color: "var(--text-heading)", letterSpacing: "0.1em" }}>SKY AI</div>
          <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-secondary)", marginTop: 2, letterSpacing: "0.08em" }}>LOG INTELLIGENCE</div>
        </div>

        <nav style={{ padding: "10px 0" }}>
          {PAGES.map(p => (
            <button key={p.id} onClick={() => setPage(p.id)} style={{
              display: "block", width: "100%", textAlign: "left",
              padding: "9px 16px",
              background: page === p.id ? "var(--sidebar-active-bg)" : "none",
              borderLeft: `2px solid ${page === p.id ? "var(--sidebar-active-bar)" : "transparent"}`,
              border: "none",
              color: page === p.id ? "var(--sidebar-active-text)" : results ? "var(--sidebar-text)" : "var(--sidebar-text-dim)",
              fontSize: FONT_SIZE.md, cursor: results ? "pointer" : "default",
              letterSpacing: "0.03em",
              opacity: (!results && p.id !== "overview") ? OPACITY.dim : OPACITY.full,
            }}>
              {p.label}
            </button>
          ))}
        </nav>

        <div style={{ marginTop: "auto", padding: "12px 16px", borderTop: "1px solid var(--border-subtle)", fontSize: FONT_SIZE.xs, color: "var(--text-muted)" }}>
          <div style={{ marginBottom: 4 }}>
            API: <span style={{ color: apiStatus === "ok" ? "var(--accent-green)" : apiStatus === "error" ? "var(--accent-red)" : "var(--accent-yellow)" }}>
              {apiStatus}
            </span>
          </div>
          {results && (
            <div style={{ color: criticalCount > 0 ? "var(--accent-red)" : "var(--text-muted)", marginTop: 2 }}>
              {criticalCount > 0 ? `● ${criticalCount} CRITICAL` : `● ${incidentCount} incidents`}
            </div>
          )}
          {results && (
            <button onClick={clearResults} style={{
              marginTop: 8, width: "100%", background: "none",
              border: "1px solid var(--border-default)", borderRadius: 4,
              color: "var(--text-muted)", padding: "4px 0", fontSize: FONT_SIZE.xs, cursor: "pointer",
            }}>
              ↑ New upload
            </button>
          )}
        </div>
      </div>

      {/* ── Main content ── */}
      <div style={{ flex: 1, overflow: "auto", padding: "24px 28px" }}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", marginBottom: 24, gap: 12 }}>
          <h1 style={{ fontSize: FONT_SIZE.h3, fontWeight: 700, color: "var(--text-heading)", margin: 0 }}>
            {PAGES.find(p => p.id === page)?.label}
          </h1>
          {runTimestamp && (
            <span style={{ fontSize: FONT_SIZE.base, color: "var(--text-secondary)", marginLeft: "auto" }}>
              {new Date(runTimestamp).toLocaleString()}
            </span>
          )}
          {results && criticalCount > 0 && (
            <div style={{ background: "var(--surface-error)", border: "1px solid var(--surface-error-border)", borderRadius: 4, padding: "3px 10px", fontSize: FONT_SIZE.base, color: "var(--accent-red)" }}>
              {criticalCount} CRITICAL
            </div>
          )}
          <ThemeToggle />
        </div>

        {/* API error banner */}
        {apiStatus === "error" && (
          <div style={{ background: "var(--surface-error)", border: "1px solid var(--surface-error-border)", borderRadius: 8, padding: "14px 18px", marginBottom: 20, fontSize: FONT_SIZE.md, color: "var(--accent-red)" }}>
            ⛔ Cannot connect to API at localhost:8000. Make sure the backend is running:<br />
            <code style={{ fontSize: FONT_SIZE.base, color: "var(--text-secondary)" }}>uvicorn backend.api:app --reload --port 8000</code>
          </div>
        )}

        {error && <ErrorBox error={error} onRetry={clearError} />}

        {!error && loading && (
          <StageSpinner stageProgress={stageProgress} elapsedSeconds={elapsedSeconds} />
        )}

        {!error && !loading && !results && (
          <>
            <PreviousRunsPanel runs={previousRuns} onLoad={loadRun} onDelete={deleteRun} currentRunId={runId} />
            <FileUpload onUpload={handleUpload} disabled={apiStatus !== "ok"} />
          </>
        )}

        {!error && !loading && results && (
          <>
            <WarningBanners results={results} />
            <RunInfoPanel results={results} onNavigate={setPage} />

            {page === "overview"   && <OverviewPage        results={results} onNavigate={setPage} />}
            {page === "anomalies"  && <AnomaliesPage       results={results} runId={runId} />}
            {page === "clusters"   && <ClusterExplorerPage results={results} />}
            {page === "stream"     && <LogStreamPage       results={results} />}
            {page === "incidents"  && <IncidentsPage       results={results} />}
            {page === "logintel"   && <LogIntelligencePage results={results} />}
            {page === "templates"  && <TemplateIntelPage   results={results} />}
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <ReadabilityStyles />
      <AppInner />
    </ThemeProvider>
  );
}