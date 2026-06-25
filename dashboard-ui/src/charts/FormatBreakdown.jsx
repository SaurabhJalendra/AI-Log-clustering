// ─── FormatBreakdown.jsx ──────────────────────────────────────────────────────
// Horizontal bar chart showing log line counts per format type.
// Props: formatCounts (object) — keys = format tags, values = line counts
// Source: results.run_info.format_counts
// ─────────────────────────────────────────────────────────────────────────────

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";

// Color by format family
function formatColor(key) {
  const k = (key || "").toLowerCase();
  if (k === "f1") return "#268bd2";
  if (k === "f2") return "#2aa198";
  if (k === "f3") return "#6c71c4";
  if (k.startsWith("f"))  return "#839496"; // other f-formats — muted blue-grey
  if (k === "access" || k === "nginx" || k === "apache") return "#3a7d44";
  if (k === "json")   return "#2a7a35";
  if (k === "noise" || k === "unknown") return "#44475a";
  return "#586e75"; // fallback
}

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
      <div style={{ color: "var(--text-secondary)", fontWeight: 600, marginBottom: 4 }}>{label}</div>
      <div style={{ color: "var(--text-muted)" }}>
        Lines: <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{val.toLocaleString()}</span>
      </div>
    </div>
  );
}

export default function FormatBreakdown({ formatCounts }) {
  if (!formatCounts || Object.keys(formatCounts).length === 0) {
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
        No format data
      </div>
    );
  }

  const data = Object.entries(formatCounts)
    .map(([format, count]) => ({ format, count: Number(count) }))
    .sort((a, b) => b.count - a.count);

  const chartHeight = Math.max(data.length * 36 + 40, 120);
  const xMax = Math.max(...data.map(d => d.count), 1);
  const maxLabelLen = Math.max(...data.map(d => (d.format || "").length), 4);
  const yAxisWidth  = Math.max(60, Math.min(maxLabelLen * 7, 140));

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
        Format Breakdown
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", opacity: 0.7, marginBottom: 12 }}>
        {data.length} format type{data.length !== 1 ? "s" : ""} detected
      </div>

      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 48, bottom: 0, left: 8 }}
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
            tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
          />
          <YAxis
            type="category"
            dataKey="format"
            tick={{ fontSize: 11, fill: "var(--text-secondary)", fontFamily: "monospace", textAnchor: "end" }}
            tickLine={false}
            axisLine={false}
            width={yAxisWidth}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--border-subtle)", opacity: 0.4 }} />
          <Bar dataKey="count" maxBarSize={18} radius={[0, 3, 3, 0]}>
            {data.map((d) => (
              <Cell key={d.format} fill={formatColor(d.format)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}