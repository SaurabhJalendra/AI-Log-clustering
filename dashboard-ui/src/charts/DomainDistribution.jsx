// ─── DomainDistribution.jsx ──────────────────────────────────────────────────
// Horizontal bar chart — domain on Y-axis, template count on X-axis.
// Uses SIGNAL_COLOR map to color bars by worst anomaly signal per domain.
// Props: clusterSummary (array) — results.cluster_summary
//        anomalies (array)      — results.anomalies (for signal lookup)
// ─────────────────────────────────────────────────────────────────────────────

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, LabelList,
} from "recharts";

const SIGNAL_COLOR = {
  none:     "#3a7d44",
  low:      "#b58900",
  medium:   "#cb4b16",
  high:     "#dc322f",
};
const SIG_RANK = { high: 3, medium: 2, low: 1, none: 0 };

function buildData(clusterSummary, anomalies) {
  // Build anomaly signal map: event_id → signal level
  const signalMap = {};
  (anomalies || []).forEach(a => {
    const label = a.anomaly_label || "INFO";
    signalMap[String(a.event_id)] = (
      label === "CRITICAL" || label === "HIGH" ? "high"
      : label === "MEDIUM" ? "medium"
      : label === "LOW" ? "low"
      : "none"
    );
  });

  const domainMap = {};
  (clusterSummary || []).forEach(cs => {
    const domain = cs.domain || "other";
    if (!domainMap[domain]) {
      domainMap[domain] = { domain, templateCount: 0, worstSignal: "none" };
    }
    domainMap[domain].templateCount += 1;

    const sig = signalMap[String(cs.cluster_id)] || "none";
    if ((SIG_RANK[sig] || 0) > (SIG_RANK[domainMap[domain].worstSignal] || 0)) {
      domainMap[domain].worstSignal = sig;
    }
  });

  return Object.values(domainMap)
    .sort((a, b) => b.templateCount - a.templateCount)
    .slice(0, 12);
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload || {};
  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--border-default)",
      borderRadius: 6,
      padding: "8px 12px",
      fontSize: 11,
      fontFamily: "monospace",
    }}>
      <div style={{ color: SIGNAL_COLOR[d.worstSignal] || "var(--text-secondary)", fontWeight: 600, marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ color: "var(--text-muted)" }}>
        Templates: <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{d.templateCount}</span>
      </div>
      <div style={{ color: "var(--text-muted)", marginTop: 2 }}>
        Signal: <span style={{ color: SIGNAL_COLOR[d.worstSignal] }}>{d.worstSignal}</span>
      </div>
    </div>
  );
}

export default function DomainDistribution({ clusterSummary, anomalies }) {
  const data = buildData(clusterSummary, anomalies);

  if (data.length === 0) {
    return (
      <div style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border-default)",
        borderRadius: 8,
        padding: "16px 20px",
        height: 200,
        display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-muted)", fontSize: 12,
      }}>
        No domain data
      </div>
    );
  }

  const chartHeight = Math.max(data.length * 40 + 40, 160);
  const xMax = Math.max(...data.map(d => d.templateCount), 1);

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
        Domain Distribution
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", opacity: 0.7, marginBottom: 12 }}>
        Templates per domain · top {data.length} · colored by anomaly signal
      </div>

      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 52, bottom: 0, left: 8 }}
          barCategoryGap="28%"
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, xMax]}
            tick={{ fontSize: 9, fill: "var(--text-muted)", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            allowDecimals={false}
          />
          <YAxis
            type="category"
            dataKey="domain"
            tick={{ fontSize: 11, fill: "var(--text-secondary)", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            width={80}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--border-subtle)", opacity: 0.4 }} />
          <Bar dataKey="templateCount" maxBarSize={20} radius={[0, 3, 3, 0]}>
            {data.map(d => (
              <Cell key={d.domain} fill={SIGNAL_COLOR[d.worstSignal] || "#586e75"} fillOpacity={0.85} />
            ))}
            <LabelList
              dataKey="templateCount"
              position="right"
              style={{ fontSize: 10, fontFamily: "monospace", fill: "var(--text-muted)" }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Signal legend */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 10 }}>
        {["none", "low", "medium", "high"].map(s => (
          <div key={s} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: SIGNAL_COLOR[s] }} />
            <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "monospace" }}>{s}</span>
          </div>
        ))}
      </div>
    </div>
  );
}