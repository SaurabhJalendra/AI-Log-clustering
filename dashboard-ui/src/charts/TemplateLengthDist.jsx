// ─── TemplateLengthDist.jsx ───────────────────────────────────────────────────
// Horizontal bar chart of template token-length buckets.
// Props: templateLengthDist (object|array, optional)
//        anomalies (array, optional) — fallback: derive from sample_template word counts
// Source: results.run_info.template_length_distribution (primary)
//         results.anomalies[].sample_template (fallback)
// ─────────────────────────────────────────────────────────────────────────────

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const BUCKETS = [
  { key: "1–3",  min: 1,  max: 3,  color: "#6c71c4" },
  { key: "4–7",  min: 4,  max: 7,  color: "#268bd2" },
  { key: "8–12", min: 8,  max: 12, color: "#2aa198" },
  { key: "13–20",min: 13, max: 20, color: "#3a7d44" },
  { key: "21+",  min: 21, max: Infinity, color: "#b58900" },
];

function wordCount(str) {
  return (str || "").trim().split(/\s+/).filter(Boolean).length;
}

function bucketsFromDist(dist) {
  // dist can be object { "1-3": 40, "4-7": 120, ... } or array [{ bucket, count }]
  if (Array.isArray(dist)) {
    return BUCKETS.map(b => {
      const match = dist.find(d =>
        String(d.bucket || d.key || "").replace("–", "-") === b.key.replace("–", "-")
      );
      return { ...b, count: Number(match?.count || 0) };
    });
  }
  if (dist && typeof dist === "object") {
    return BUCKETS.map(b => {
      // try exact key match with either dash style
      const val = dist[b.key] ?? dist[b.key.replace("–", "-")] ?? 0;
      return { ...b, count: Number(val) };
    });
  }
  return null;
}

function bucketsFromAnomalies(anomalies) {
  const counts = { "1–3": 0, "4–7": 0, "8–12": 0, "13–20": 0, "21+": 0 };
  (anomalies || []).forEach(a => {
    const wc = wordCount(a.sample_template || a.sample_message || "");
    const bucket = BUCKETS.find(b => wc >= b.min && wc <= b.max);
    if (bucket) counts[bucket.key] += 1;
  });
  return BUCKETS.map(b => ({ ...b, count: counts[b.key] }));
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
      <div style={{ color: "var(--text-secondary)", fontWeight: 600, marginBottom: 4 }}>
        {label} tokens
      </div>
      <div style={{ color: "var(--text-muted)" }}>
        Templates: <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{val}</span>
      </div>
    </div>
  );
}

export default function TemplateLengthDist({ templateLengthDist, anomalies }) {
  let data = null;
  let isFallback = false;

  if (templateLengthDist) {
    data = bucketsFromDist(templateLengthDist);
  }
  if (!data || data.every(d => d.count === 0)) {
    data = bucketsFromAnomalies(anomalies);
    isFallback = true;
  }

  const hasAny = data.some(d => d.count > 0);

  if (!hasAny) {
    return (
      <div style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border-default)",
        borderRadius: 8,
        padding: "16px 20px",
        height: 160,
        display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-muted)", fontSize: 12,
      }}>
        No template length data
      </div>
    );
  }

  const xMax = Math.max(...data.map(d => d.count), 1);

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
        Template Length Distribution
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", opacity: 0.7, marginBottom: 12 }}>
        token count per template{isFallback ? " · estimated from sample templates" : ""}
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 48, bottom: 0, left: 8 }}
          barCategoryGap="25%"
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
            dataKey="key"
            tick={{ fontSize: 11, fill: "var(--text-secondary)", fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            width={44}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--border-subtle)", opacity: 0.4 }} />
          <Bar dataKey="count" maxBarSize={18} radius={[0, 3, 3, 0]}>
            {data.map(d => (
              <Cell key={d.key} fill={d.color} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}