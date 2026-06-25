import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell,
} from "recharts";
import FormatBreakdown       from "../charts/FormatBreakdown";
import SeverityDistribution  from "../charts/SeverityDistribution";
import { SIGNAL_COLOR, FONT_SIZE, FONT } from "../Theme";
import { Badge } from "../Components";

// ─── LOG INTELLIGENCE PAGE ────────────────────────────────────────────────────

export default function LogIntelligencePage({ results }) {
  const ri             = results?.run_info || {};
  const totalLines     = ri.total_lines  || 0;
  const parsedOk       = ri.parsed_ok    || 0;
  const noiseLines     = ri.noise_lines  || 0;
  const parseRate      = totalLines > 0 ? Math.round(parsedOk / totalLines * 100) : 0;
  const formatCounts   = ri.format_counts   || {};
  const severityCounts = ri.severity_counts || {};
  const noiseReasons   = ri.noise_reasons   || {};
  const uniqueTemplates = ri.unique_templates || 0;
  // Build a cluster_id → cluster_summary lookup so we can enrich anomaly rows
  // with sample_template and cluster_label (stage 3 fields not present in stage 4).
  const csLookup = {};
  (results?.cluster_summary || []).forEach(cs => {
    csLookup[String(cs.cluster_id)] = cs;
  });

  const recentSample = (results?.anomalies || []).slice(0, 8).map(row => {
    const cs = csLookup[String(row.event_id)] || {};
    return {
      ...row,
      // Prefer stage3 sample_template, then stage4 sample_message, then cluster label
      resolvedMessage: cs.sample_template || row.sample_template || row.sample_message || cs.cluster_label || "—",
    };
  });

  const labelCounts = ri.anomaly_label_counts || {};
  const labelData   = Object.entries(labelCounts).map(([label, count]) => ({ label, count }));

  return (
    <div>
      {/* ── KPI row ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 20 }}>
        {[
          { label: "Total Lines",  value: totalLines.toLocaleString(), color: "var(--text-primary)" },
          { label: "Parse Rate",   value: `${parseRate}%`,             color: parseRate > 80 ? "#3a7d44" : parseRate > 60 ? "#b58900" : "#dc322f" },
          { label: "Encoding",     value: ri.detected_encoding || "—", color: "var(--accent-blue)" },
          { label: "Format Types", value: Object.keys(formatCounts).length || "—", color: "var(--accent-blue)" },
        ].map(({ label, value, color }) => (
          <div key={label} style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px" }}>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
            <div style={{ fontSize: FONT_SIZE.kpi, fontWeight: 700, color, fontFamily: FONT.mono }}>{value}</div>
          </div>
        ))}
      </div>

      {/* ── Format Breakdown + Severity Distribution ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <FormatBreakdown formatCounts={formatCounts} />
        <SeverityDistribution severityCounts={severityCounts} />
      </div>

      {/* ── Normalisation Pipeline ── */}
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", marginBottom: 16 }}>
        <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 12 }}>
          Normalisation Pipeline
        </div>
        <div className="norm-pipeline" style={{ marginBottom: 16 }}>
          {[
            { label: "Total Lines",        value: totalLines.toLocaleString(),             color: "var(--accent-blue)", border: "var(--accent-blue)" },
            { label: "Parsed OK",          value: parsedOk.toLocaleString(),              color: "#3a7d44",            border: "#3a7d44" },
            { label: "After Noise Filter", value: (parsedOk - noiseLines).toLocaleString(), color: "#b58900",           border: "#b58900" },
            { label: "Unique Templates",   value: uniqueTemplates.toLocaleString(),        color: "#6c71c4",            border: "#6c71c4" },
          ].map(({ label, value, color, border }, i, arr) => (
            <div key={label} style={{ display: "flex", alignItems: "center", flex: 1, gap: 6 }}>
              <div className="norm-step" style={{ color, borderColor: border, background: "var(--bg-card-deep)", flex: 1 }}>
                <div style={{ fontSize: FONT_SIZE.xl, fontWeight: 700, marginBottom: 2 }}>{value}</div>
                <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", letterSpacing: "0.04em" }}>{label}</div>
              </div>
              {i < arr.length - 1 && (
                <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.xl, flexShrink: 0 }}>→</div>
              )}
            </div>
          ))}
        </div>

        {/* Noise Reasons + Parse Stats */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div style={{ background: "var(--bg-card-deep)", border: "1px solid var(--border-subtle)", borderRadius: 6, padding: "10px 12px" }}>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>Noise Reasons</div>
            {Object.keys(noiseReasons).length === 0
              ? <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-muted)" }}>None recorded</div>
              : Object.entries(noiseReasons).map(([reason, count]) => (
                <div key={reason} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px solid var(--border-subtle)", fontSize: FONT_SIZE.base }}>
                  <span style={{ color: "var(--text-muted)", fontFamily: "monospace" }}>{reason}</span>
                  <span style={{ color: "var(--text-secondary)", fontFamily: "monospace" }}>{Number(count).toLocaleString()}</span>
                </div>
              ))
            }
          </div>
          <div style={{ background: "var(--bg-card-deep)", border: "1px solid var(--border-subtle)", borderRadius: 6, padding: "10px 12px" }}>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>Parse Stats</div>
            {[
              { label: "Timestamps parsed",  v: ri.ts_parsed_ok ?? "—" },
              { label: "Timestamp failures", v: ri.ts_failed     ?? "—" },
              { label: "JSON lines parsed",  v: ri.json_ok       ?? "—" },
            ].map(({ label, v }) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px solid var(--border-subtle)", fontSize: FONT_SIZE.base }}>
                <span style={{ color: "var(--text-muted)" }}>{label}</span>
                <span style={{ color: "var(--text-secondary)", fontFamily: "monospace" }}>{typeof v === "number" ? v.toLocaleString() : v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Recent Log Sample ── */}
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", marginBottom: 16 }}>
        <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 12 }}>
          Recent Log Sample
        </div>
        {recentSample.length === 0
          ? <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.md }}>No log data available</div>
          : <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: FONT_SIZE.base, fontFamily: "monospace" }}>
                <thead>
                  <tr style={{ background: "var(--bg-card-deep)", borderBottom: "1px solid var(--border-default)" }}>
                    {["Severity", "Service · Domain", "Message"].map(h => (
                      <th key={h} style={{ padding: "7px 12px", color: "var(--text-muted)", fontWeight: 400, textAlign: "left", fontSize: FONT_SIZE.xs, letterSpacing: "0.05em" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {recentSample.map((row, i) => (
                    <tr key={row.event_id || i} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                      <td style={{ padding: "6px 12px", whiteSpace: "nowrap" }}>
                        <Badge label={row.anomaly_label} type={row.anomaly_label} />
                      </td>
                      <td style={{ padding: "6px 12px", whiteSpace: "nowrap" }}>
                        <div style={{ color: "var(--text-primary)", fontSize: FONT_SIZE.md, fontWeight: 600 }}>{row.top_source || "—"}</div>
                        <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginTop: 1 }}>{row.domain}</div>
                      </td>
                      <td style={{ padding: "6px 12px", color: "var(--text-secondary)", maxWidth: 480, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                        title={row.resolvedMessage}>
                        {row.resolvedMessage}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
        }
      </div>

      {/* ── Pipeline Audit grid ── */}
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px" }}>
        <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 12 }}>
          Pipeline Audit
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12 }}>
          {[
            { stage: "Stage 1 — Ingestion & Parsing", color: "var(--accent-blue)",
              items: [
                { label: "Total lines",  v: totalLines.toLocaleString() },
                { label: "Encoding",     v: ri.detected_encoding || "—" },
                { label: "Format types", v: Object.keys(formatCounts).length || "—" },
              ] },
            { stage: "Stage 2 — Normalisation", color: "#3a7d44",
              items: [
                { label: "Parsed OK",        v: parsedOk.toLocaleString() },
                { label: "Parse rate",       v: `${parseRate}%` },
                { label: "Unique templates", v: uniqueTemplates.toLocaleString() },
              ] },
            { stage: "Stage 3 — Clustering", color: "#b58900",
              items: [
                { label: "Total clusters",   v: ri.total_clusters ?? "—" },
                { label: "Merged templates", v: ri.merged_template_count ?? "—" },
                { label: "Silhouette score", v: ri.silhouette_score != null ? Number(ri.silhouette_score).toFixed(3) : "—" },
              ] },
            { stage: "Stage 4 — Anomaly Detection", color: "#cb4b16",
              items: [
                { label: "Anomaly count",    v: (results?.anomaly_count ?? 0).toLocaleString() },
                { label: "CRITICAL",         v: (ri.anomaly_label_counts?.CRITICAL ?? 0).toLocaleString() },
                { label: "HIGH / MEDIUM",    v: `${ri.anomaly_label_counts?.HIGH ?? 0} / ${ri.anomaly_label_counts?.MEDIUM ?? 0}` },
              ] },
            { stage: "Stage 5 — Root Cause Analysis", color: "#dc322f",
              items: [
                { label: "Incidents",          v: (results?.incident_count ?? 0).toLocaleString() },
                { label: "CRITICAL incidents", v: (ri.incident_severity_counts?.CRITICAL ?? 0).toLocaleString() },
                { label: "Narrative status",   v: (results?.incident_count ?? 0) > 0 ? "generated" : "—" },
              ] },
          ].map(({ stage, color, items }) => (
            <div key={stage} style={{ background: "var(--bg-card-deep)", border: "1px solid var(--border-subtle)", borderRadius: 6, padding: "12px 14px" }}>
              <div style={{ fontSize: FONT_SIZE.xs, color, marginBottom: 8, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" }}>{stage}</div>
              {items.map(({ label, v }) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid var(--border-subtle)", fontSize: FONT_SIZE.base }}>
                  <span style={{ color: "var(--text-muted)" }}>{label}</span>
                  <span style={{ color: "var(--text-secondary)", fontFamily: "monospace" }}>{v}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}