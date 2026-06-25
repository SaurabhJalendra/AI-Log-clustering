// ─── SeverityDistribution.jsx ────────────────────────────────────────────────
// Two-part display: (1) proportional colored strip, (2) bar chart by severity.
// Props: severityCounts (object) — keys = severity labels, values = counts
// Source: results.run_info.severity_counts
// ─────────────────────────────────────────────────────────────────────────────

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const SEV_ORDER  = ["FATAL", "CRITICAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "TRACE"];
const SEV_COLOR  = {
  FATAL:    "#dc322f",
  CRITICAL: "#dc322f",
  ERROR:    "#cb4b16",
  WARN:     "#b58900",
  WARNING:  "#b58900",
  INFO:     "#268bd2",
  DEBUG:    "#555555",
  TRACE:    "#44475a",
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const val = payload[0]?.value ?? 0;
  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--border-default)",
      borderRadius: 6,
      padding: "8px 12px",
      fontSize: 11,
      fontFamily: "monospace",
    }}>
      <div style={{ color: SEV_COLOR[label] || "var(--text-secondary)", fontWeight: 600, marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ color: "var(--text-muted)" }}>
        Count: <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{val.toLocaleString()}</span>
      </div>
    </div>
  );
}

export default function SeverityDistribution({ severityCounts }) {
  if (!severityCounts || Object.keys(severityCounts).length === 0) {
    return (
      <div style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border-default)",
        borderRadius: 8,
        padding: "16px 20px",
        height: 180,
        display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-muted)", fontSize: 12,
      }}>
        No severity data
      </div>
    );
  }

  // Build ordered data — known severities first, then unknown keys
  const knownKeys  = SEV_ORDER.filter(k => severityCounts[k] !== undefined);
  const unknownKeys = Object.keys(severityCounts).filter(k => !SEV_ORDER.includes(k));
  const orderedKeys = [...knownKeys, ...unknownKeys];

  const data = orderedKeys.map(key => ({
    severity: key,
    count: Number(severityCounts[key] || 0),
  })).filter(d => d.count > 0);

  const total = data.reduce((s, d) => s + d.count, 0);

  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--border-default)",
      borderRadius: 8,
      padding: "16px 20px",
    }}>
      <div style={{
        fontSize: 12, color: "var(--text-muted)",
        letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 2,
      }}>
        Severity Distribution
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", opacity: 0.7, marginBottom: 12 }}>
        {total.toLocaleString()} total log lines across {data.length} severity level{data.length !== 1 ? "s" : ""}
      </div>

      {/* Proportional strip */}
      <div className="sev-strip" style={{ marginBottom: 14 }}>
        {data.map(d => (
          <div
            key={d.severity}
            className="sev-seg"
            title={`${d.severity}: ${d.count.toLocaleString()} (${total > 0 ? Math.round(d.count / total * 100) : 0}%)`}
            style={{
              flex: d.count,
              background: SEV_COLOR[d.severity] || "#586e75",
              minWidth: 2,
            }}
          />
        ))}
      </div>

      {/* Bar chart */}
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} margin={{ top: 4, right: 16, bottom: 0, left: -10 }} barCategoryGap="30%">
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" vertical={false} />
          <XAxis
            dataKey="severity"
            tick={{ fontSize: 9, fill: "var(--text-muted)", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tick={{ fontSize: 9, fill: "var(--text-muted)", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            allowDecimals={false}
            tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--border-subtle)", opacity: 0.4 }} />
          <Bar dataKey="count" maxBarSize={32} radius={[3, 3, 0, 0]}>
            {data.map(d => (
              <Cell key={d.severity} fill={SEV_COLOR[d.severity] || "#586e75"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend row */}
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 10 }}>
        {data.map(d => (
          <div key={d.severity} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: SEV_COLOR[d.severity] || "#586e75" }} />
            <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "monospace" }}>
              {d.severity} ({total > 0 ? Math.round(d.count / total * 100) : 0}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}