import { useState } from "react";
import { submitFeedback } from "../api";
import AnomalyScatter from "../charts/AnomalyScatter";
import ServiceBar     from "../charts/ServiceBar";
import { SIGNAL_COLOR, SIGNAL_BG, FONT_SIZE, FONT } from "../Theme";
import { Badge, ScoreBar } from "../Components";

// ─── XAI HELPERS ──────────────────────────────────────────────────────────────

const rarityNote = (cluster) => {
  const r = cluster.rarity_score;
  if (r > 0.80) return "very rare — seen in <5% of similar log files";
  if (r > 0.60) return "uncommon pattern for this service";
  if (r > 0.40) return "moderately infrequent";
  return "common pattern";
};

const burstNote = (cluster) => {
  if (!cluster.burst_detected) return "no burst";
  const peak = cluster.peak_window_count;
  return `peak of ${peak} occurrences in a single time window`;
};

const SINGLETON_EXPLANATIONS = {
  true_anomaly:             "Template matched a known anomaly pattern (critical keyword, exception signature, or failure indicator)",
  unseen_variant:           "Template not seen in previous runs — may indicate a new failure mode",
  impossible_attempt_count: "Attempt counter exceeded maximum threshold (e.g. 5/3 retries)",
  known_normal:             "Previously confirmed as routine behaviour",
  normal:                   "No anomaly signal detected in template text",
};

// ─── SUB-COMPONENTS ───────────────────────────────────────────────────────────

function XaiScoreRow({ label, score, note }) {
  const pct = Math.round((score || 0) * 100);
  const barColor = pct > 75 ? "#dc322f" : pct > 50 ? "#b58900" : pct > 25 ? "#268bd2" : "var(--text-muted)";
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 3 }}>
        <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontFamily: "monospace" }}>
          {label}
        </span>
        <span style={{ fontSize: FONT_SIZE.xs, color: barColor, fontFamily: "monospace", fontWeight: 600 }}>
          {pct}%
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ flex: 1, height: 4, background: "var(--border-default)", borderRadius: 2, overflow: "hidden" }}>
          <div style={{ width: `${pct}%`, height: "100%", background: barColor, borderRadius: 2, transition: "width 0.3s ease" }} />
        </div>
      </div>
      {note && (
        <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", fontFamily: "monospace", marginTop: 2, fontStyle: "italic" }}>
          {note}
        </div>
      )}
    </div>
  );
}

function TrendContext({ direction, windowCounts, peakCount, burstDetected }) {
  const arrow = direction === "rising" ? "↑" : direction === "falling" ? "↓" : "→";
  const arrowColor = direction === "rising" ? "#dc322f" : direction === "falling" ? "#3a7d44" : "var(--text-muted)";

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: FONT_SIZE.lg, color: arrowColor, fontFamily: "monospace", fontWeight: 700 }}>
          {arrow}
        </span>
        <span style={{ fontSize: FONT_SIZE.md, color: "var(--text-secondary)", fontFamily: "monospace" }}>
          {direction || "stable"}
        </span>
        {burstDetected && peakCount && (
          <span style={{ fontSize: FONT_SIZE.xs, color: "#dc322f", fontFamily: "monospace", background: "rgba(220,50,47,0.08)", border: "1px solid rgba(220,50,47,0.25)", borderRadius: 3, padding: "1px 6px" }}>
            peak: {peakCount} / window
          </span>
        )}
      </div>
      {Array.isArray(windowCounts) && windowCounts.length > 0 && (
        <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 32 }}>
          {windowCounts.map((count, i) => {
            const max = Math.max(...windowCounts, 1);
            const h = Math.round((count / max) * 28);
            return (
              <div key={i} title={`Window ${i + 1}: ${count}`} style={{
                flex: 1, height: h || 2, minHeight: 2,
                background: i === windowCounts.indexOf(Math.max(...windowCounts)) ? "#dc322f" : "var(--border-strong)",
                borderRadius: 2, transition: "height 0.2s ease",
              }} />
            );
          })}
        </div>
      )}
    </div>
  );
}

function SingletonExplanation({ singletonClass, anomalySignal, anomalyReason }) {
  const explanation = SINGLETON_EXPLANATIONS[singletonClass] ?? "Unknown classification";
  return (
    <div>
      <div style={{ fontFamily: "monospace", fontSize: FONT_SIZE.md, color: "var(--text-primary)", fontWeight: 600, marginBottom: 4 }}>
        {singletonClass || "—"}
      </div>
      <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-secondary)", lineHeight: 1.5, marginBottom: anomalyReason ? 6 : 0 }}>
        {explanation}
      </div>
      {anomalyReason && (
        <div style={{ fontSize: FONT_SIZE.xs, color: "#b58900", fontFamily: "monospace", background: "rgba(181,137,0,0.06)", border: "1px solid rgba(181,137,0,0.2)", borderRadius: 3, padding: "4px 8px", marginTop: 4 }}>
          Reason: {anomalyReason}
        </div>
      )}
    </div>
  );
}

// ─── WHY PANEL ────────────────────────────────────────────────────────────────

function WhyPanel({ row }) {
  return (
    <div style={{ background: "var(--bg-card-deep)", border: "1px solid var(--border-default)", borderRadius: 6, padding: "14px 16px", marginTop: 12 }}>
      <div style={{ fontSize: FONT_SIZE.base, color: "#dc322f", fontWeight: 600, marginBottom: 14, letterSpacing: "0.06em" }}>
        ⚡ XAI REASONING
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* ── 1. Score Breakdown ── */}
        <div>
          <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8, fontFamily: "monospace" }}>
            Score Breakdown
          </div>
          <XaiScoreRow label="Rarity"       score={row.rarity_score}      note={rarityNote(row)} />
          <XaiScoreRow label="Burstiness"   score={row.burstiness_score}  note={burstNote(row)} />
          <XaiScoreRow label="Severity"     score={row.severity_score}    note={row.dominant_severity} />
          <XaiScoreRow label="Spread"       score={row.spread_score}      note={row.source_spread != null ? `across ${row.source_spread} sources` : null} />
          <XaiScoreRow label="Error Volume" score={row.error_volume ?? 0} note={row.error_pct != null ? `${row.error_pct.toFixed(1)}% error lines` : null} />
        </div>

        {/* ── Right column ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

          {/* ── 2. Signal Classification ── */}
          <div>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8, fontFamily: "monospace" }}>
              Signal Classification
            </div>
            <SingletonExplanation
              singletonClass={row.singleton_class}
              anomalySignal={row.anomaly_signal}
              anomalyReason={row.anomaly_reason}
            />
          </div>

          {/* ── 2b. Sample Log Template ── */}
          <div>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8, fontFamily: "monospace" }}>
              Sample Log Template
            </div>
            {/* Field resolution priority — covers every known API response shape:
                template_string    → Stage 3 drain output (primary)
                representative_log → alt field name used in some pipeline versions
                event_template     → normalised name written by the anomaly scorer
                sample_template    → cluster_summary join field
                sample_message     → raw log line fallback                       */}
            {(row.template_string || row.representative_log || row.event_template || row.sample_template || row.sample_message) ? (
              <div style={{
                background: "var(--bg-card)",
                border: "1px solid var(--border-subtle)",
                borderLeft: "3px solid var(--border-strong)",
                borderRadius: 4,
                padding: "7px 10px",
                fontFamily: "monospace",
                fontSize: FONT_SIZE.xs,
                color: "var(--text-secondary)",
                lineHeight: 1.7,
                wordBreak: "break-all",
                whiteSpace: "pre-wrap",
                maxHeight: 120,
                overflowY: "auto",
              }}>
                {row.template_string || row.representative_log || row.event_template || row.sample_template || row.sample_message}
              </div>
            ) : (
              <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", fontFamily: "monospace", fontStyle: "italic" }}>
                No template available for this cluster
              </div>
            )}
            {row.template_id && (
              <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", fontFamily: "monospace", marginTop: 4 }}>
                template_id: <span style={{ color: "var(--text-secondary)" }}>{row.template_id}</span>
              </div>
            )}
          </div>

          {/* ── 3. Trend ── */}
          <div>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8, fontFamily: "monospace" }}>
              Trend
            </div>
            <TrendContext
              direction={row.trend_direction}
              windowCounts={row.window_counts}
              peakCount={row.peak_window_count}
              burstDetected={row.burst_detected}
            />
          </div>
        </div>
      </div>

      {/* ── 4. Representative Log Line ── */}
      {(row.sample_template || row.sample_message) && (
        <div style={{ marginTop: 14 }}>
          <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6, fontFamily: "monospace" }}>
            Representative Log Line
          </div>
          <div style={{
            background: "var(--bg-card)", border: "1px solid var(--border-subtle)",
            borderRadius: 4, padding: "8px 12px",
            fontFamily: "monospace", fontSize: FONT_SIZE.xs,
            color: "var(--text-secondary)", lineHeight: 1.6,
            wordBreak: "break-all", whiteSpace: "pre-wrap",
          }}>
            {row.sample_template || row.sample_message}
          </div>
        </div>
      )}

      {/* ── 5. Suppression / audit notes ── */}
      {(row._burst_damped || row.is_warn_routine) && (
        <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 6 }}>
          {row._burst_damped && (
            <div style={{ fontSize: FONT_SIZE.xs, fontFamily: "monospace", color: "#b58900", background: "rgba(181,137,0,0.06)", border: "1px solid rgba(181,137,0,0.2)", borderRadius: 3, padding: "5px 10px" }}>
              ⚠ Burst detected but damped — 0% error signal, score was reduced
            </div>
          )}
          {row.is_warn_routine && (
            <div style={{ fontSize: FONT_SIZE.xs, fontFamily: "monospace", color: "#268bd2", background: "rgba(38,139,210,0.06)", border: "1px solid rgba(38,139,210,0.2)", borderRadius: 3, padding: "5px 10px" }}>
              ℹ Scored as warn-routine — capped at LOW (no error signal)
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── ANOMALY CARD ─────────────────────────────────────────────────────────────

function AnomalyCard({ row, expanded, onToggle, runId }) {
  const [feedbackState, setFeedbackState] = useState(null);
  // null | 'true_positive' | 'false_positive' | 'saving' | 'error'

  const handleFeedback = async (label) => {
    // ── Guard: template_id is required for the safelist to work ──────
    // row.template_id comes from the pipeline's stage3/4 output.
    // If it is missing the backend will reject the record (400) and the
    // false-positive safelist will never suppress this cluster on reruns.
    const templateId = row.template_id || row.tid || row.cluster_template_id || "";
    if (!templateId) {
      console.error(
        "[handleFeedback] template_id is missing on row — available keys:",
        Object.keys(row),
        "row.event_id:", row.event_id,
      );
      setFeedbackState("error");
      setTimeout(() => setFeedbackState(null), 4000);
      return;          // abort — do NOT send a broken record to the DB
    }

    setFeedbackState("saving");
    try {
      await submitFeedback({
        cluster_ref:   row.event_id,
        label,
        template_id:   templateId,
        run_id:        runId              || "",
        service:       row.top_source     || "",
        log_template:  row.sample_message || "",
        severity:      row.dominant_severity || "",
        anomaly_score: row.anomaly_score   || 0,
      });
      setFeedbackState(label);
    } catch (err) {
      console.error("Feedback error:", err);
      setFeedbackState("error");
      setTimeout(() => setFeedbackState(null), 3000);
    }
  };

  return (
    <div
      onClick={onToggle}
      style={{
        background: "var(--bg-card)",
        border: `1px solid ${row.anomaly_score > 0.75 ? "var(--surface-error-border)" : "var(--border-default)"}`,
        borderLeft: `3px solid ${SIGNAL_COLOR[row.anomaly_label]}`,
        borderRadius: 8, padding: "14px 18px", marginBottom: 10,
        cursor: "pointer",
        transition: "background 0.15s ease",
      }}
      onMouseEnter={e => { e.currentTarget.style.background = "var(--bg-card-hover, var(--bg-card-deep))"; }}
      onMouseLeave={e => { e.currentTarget.style.background = "var(--bg-card)"; }}
    >
      {/* ── Header row ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
        <Badge label={row.anomaly_label} type={row.anomaly_label} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontFamily: "monospace", fontSize: FONT_SIZE.lg, color: "var(--text-primary)", fontWeight: 600 }}>
              {row.top_source || "unknown"}
            </span>
            <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", background: "var(--bg-card-deep)", border: "1px solid var(--border-subtle)", borderRadius: 3, padding: "1px 6px", fontFamily: "monospace" }}>
              {row.domain}
            </span>
            {row.burst_detected  && <span style={{ fontSize: FONT_SIZE.xs, color: "#dc322f", fontFamily: "monospace" }}>⚡ BURST</span>}
            {row.recurrence_flag && <span style={{ fontSize: FONT_SIZE.xs, color: "#b58900", fontFamily: "monospace" }}>↻ RECUR</span>}
          </div>
          <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", fontFamily: "monospace", marginTop: 2 }}>
            ref: {row.event_id}
          </div>
        </div>
        {/* Button kept for visual affordance; click is handled by the card wrapper */}
        <button
          onClick={e => { e.stopPropagation(); onToggle(); }}
          style={{ background: "none", border: "1px solid var(--border-default)", borderRadius: 4, color: "var(--text-muted)", padding: "3px 10px", fontSize: FONT_SIZE.base, cursor: "pointer", flexShrink: 0 }}
        >
          {expanded ? "▲ hide" : "▼ why?"}
        </button>
      </div>

      <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-secondary)", marginBottom: 10, fontFamily: "monospace", lineHeight: 1.5 }}>{row.sample_message}</div>

      {/* ── Metrics row ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
        {[
          { label: "Anomaly Score", v: <ScoreBar score={row.anomaly_score} /> },
          { label: "Error %",       v: <span style={{ color: "#dc322f", fontFamily: "monospace", fontSize: FONT_SIZE.md }}>{Math.round(row.error_pct || 0)}%</span> },
          { label: "Trend",         v: <span style={{ color: row.trend_direction === "rising" ? "#dc322f" : row.trend_direction === "falling" ? "#3a7d44" : "var(--text-muted)", fontSize: FONT_SIZE.md }}>
              {row.trend_direction === "rising" ? "↑" : row.trend_direction === "falling" ? "↓" : "→"} {row.trend_direction || "—"}
            </span> },
          { label: "Burst",         v: <span style={{ color: row.burst_detected ? "#dc322f" : "var(--text-muted)", fontSize: FONT_SIZE.md }}>{row.burst_detected ? "● detected" : "○ none"}</span> },
        ].map(({ label, v }) => (
          <div key={label}>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
            {v}
          </div>
        ))}
      </div>

      {/* ── Expanded XAI panel ── */}
      {expanded && (
        <div onClick={e => e.stopPropagation()}>
          <WhyPanel row={row} />

          {/* Feedback buttons */}
          <div style={{ display: "flex", gap: 8, marginTop: 12, alignItems: "center" }}>
            <button
              onClick={() => handleFeedback("true_positive")}
              disabled={feedbackState === "saving" || feedbackState === "true_positive"}
              style={{
                background: feedbackState === "true_positive" ? "#3a7d44" : "var(--surface-info)",
                border: "1px solid #3a7d44", borderRadius: 4,
                color: feedbackState === "true_positive" ? "#fff" : "#3a7d44",
                padding: "4px 12px", fontSize: FONT_SIZE.base,
                cursor: feedbackState === "saving" ? "wait" : feedbackState === "true_positive" ? "default" : "pointer",
                opacity: feedbackState === "false_positive" ? 0.4 : 1,
              }}
            >
              {feedbackState === "saving" ? "..." : feedbackState === "true_positive" ? "👍 Marked True Positive" : "👍 True Positive"}
            </button>
            <button
              onClick={() => handleFeedback("false_positive")}
              disabled={feedbackState === "saving" || feedbackState === "false_positive"}
              style={{
                background: feedbackState === "false_positive" ? "#dc322f" : "var(--surface-error)",
                border: "1px solid #dc322f", borderRadius: 4,
                color: feedbackState === "false_positive" ? "#fff" : "#dc322f",
                padding: "4px 12px", fontSize: FONT_SIZE.base,
                cursor: feedbackState === "saving" ? "wait" : feedbackState === "false_positive" ? "default" : "pointer",
                opacity: feedbackState === "true_positive" ? 0.4 : 1,
              }}
            >
              {feedbackState === "saving" ? "..." : feedbackState === "false_positive" ? "👎 Marked False Positive" : "👎 False Positive"}
            </button>
            {feedbackState === "error" && (
              <span style={{ fontSize: FONT_SIZE.xs, color: "#dc322f" }}>
                Save failed — template_id missing on this cluster (check console)
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── ANOMALIES PAGE ───────────────────────────────────────────────────────────

export default function AnomaliesPage({ results, runId }) {
  const anomalyDf = results?.anomalies || [];
  const [expanded, setExpanded] = useState(null);
  const [filter, setFilter]     = useState("ALL");
  const labels   = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"];
  const filtered = filter === "ALL" ? anomalyDf : anomalyDf.filter(r => r.anomaly_label === filter);

  return (
    <div>
      <AnomalyScatter anomalies={anomalyDf} />
      <ServiceBar anomalies={anomalyDf} />

      {/* Filter pills */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        {labels.map(l => (
          <button key={l} onClick={() => setFilter(l)} style={{
            background: filter === l ? (SIGNAL_BG[l] || "var(--bg-card)") : "var(--bg-card)",
            border: `1px solid ${filter === l ? (SIGNAL_COLOR[l] || "var(--border-strong)") : "var(--border-default)"}`,
            color: filter === l ? (SIGNAL_COLOR[l] || "var(--text-primary)") : "var(--text-muted)",
            borderRadius: 4, padding: "4px 14px", fontSize: FONT_SIZE.md, cursor: "pointer", fontFamily: "monospace",
          }}>
            {l}
          </button>
        ))}
        <span style={{ marginLeft: "auto", fontSize: FONT_SIZE.md, color: "var(--text-muted)", alignSelf: "center" }}>{filtered.length} events</span>
      </div>

      {filtered.length === 0 && (
        <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.lg, textAlign: "center", marginTop: 40 }}>No anomalies in this category</div>
      )}
      {filtered.map(row => (
        <AnomalyCard
          key={row.event_id}
          row={row}
          expanded={expanded === row.event_id}
          onToggle={() => setExpanded(expanded === row.event_id ? null : row.event_id)}
          runId={runId}
        />
      ))}
    </div>
  );
}