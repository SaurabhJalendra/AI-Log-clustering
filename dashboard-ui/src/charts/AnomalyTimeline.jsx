// ─── AnomalyTimeline.jsx — stacked bar view ───────────────────────────────────
// Used by OverviewPage when chartMode === "bar"
// Exports: buildChartData (shared with AnomalyTimelineLine)

import {
  ComposedChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

// ── constants ─────────────────────────────────────────────────────────────────

export const SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];

export const SEV_COLOR = {
  CRITICAL: "#dc322f",
  HIGH:     "#cb4b16",
  MEDIUM:   "#b58900",
  LOW:      "#6c71c4",
};

export const BUCKET_MINUTES = 30;

// ── helpers ───────────────────────────────────────────────────────────────────

function toBucket(dateStr) {
  const d = new Date(dateStr);
  if (isNaN(d)) return null;
  d.setSeconds(0, 0);
  d.setMinutes(Math.floor(d.getMinutes() / BUCKET_MINUTES) * BUCKET_MINUTES);
  return d.toISOString();
}

function fmtBucket(iso) {
  return new Date(iso).toISOString().slice(11, 16);
}

export function buildChartData(results) {
  const anomalyMap = {};
  (results?.anomalies || []).forEach(a => {
    anomalyMap[String(a.event_id)] = a.anomaly_label || "LOW";
  });

  const clusters = results?.cluster_summary || [];
  if (clusters.length === 0) return [];

  const buckets = {};

  clusters.forEach(cs => {
    if (!cs.first_seen) return;
    const bucket = toBucket(cs.first_seen);
    if (!bucket) return;

    const label      = anomalyMap[String(cs.cluster_id)] || null;
    const isAnomalous = cs.anomaly_signal && cs.anomaly_signal !== "none";

    if (!buckets[bucket]) {
      buckets[bucket] = { bucket, CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0, routine: 0, total: 0, burst: 0 };
    }

    buckets[bucket].total += 1;

    if (!isAnomalous || !label) {
      buckets[bucket].routine += 1;
    } else {
      buckets[bucket][label] = (buckets[bucket][label] || 0) + 1;
    }
  });

  (results?.anomalies || []).forEach(a => {
    if (!a.burst_detected) return;
    const cs = clusters.find(c => String(c.cluster_id) === String(a.event_id));
    if (!cs?.first_seen) return;
    const bucket = toBucket(cs.first_seen);
    if (bucket && buckets[bucket]) buckets[bucket].burst += 1;
  });

  return Object.values(buckets)
    .sort((a, b) => a.bucket.localeCompare(b.bucket))
    .map(b => ({
      ...b,
      label: fmtBucket(b.bucket),
      anomalyTotal: b.CRITICAL + b.HIGH + b.MEDIUM + b.LOW,
    }));
}

// ── tooltip ───────────────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload || {};
  const total = (data.CRITICAL || 0) + (data.HIGH || 0) + (data.MEDIUM || 0) + (data.LOW || 0);
  return (
    <div style={{
      background: "var(--bg-card)", border: "1px solid var(--border-default)",
      borderRadius: 6, padding: "10px 14px", fontSize: 11,
      fontFamily: "monospace", minWidth: 160,
    }}>
      <div style={{ color: "var(--text-muted)", marginBottom: 6 }}>{label} UTC</div>
      {["CRITICAL","HIGH","MEDIUM","LOW"].map(sev => data[sev] > 0 && (
        <div key={sev} style={{ display: "flex", justifyContent: "space-between", gap: 16, color: SEV_COLOR[sev], marginBottom: 2 }}>
          <span>{sev}</span>
          <span style={{ fontWeight: 700 }}>{data[sev]}</span>
        </div>
      ))}
      {total > 0 && (
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, color: "var(--text-muted)", marginTop: 4, borderTop: "1px solid var(--border-subtle)", paddingTop: 4 }}>
          <span>total</span><span style={{ fontWeight: 700 }}>{total}</span>
        </div>
      )}
      {data.routine > 0 && (
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, color: "var(--text-muted)", marginTop: 4, borderTop: "1px solid var(--border-subtle)", paddingTop: 4 }}>
          <span>routine</span><span>{data.routine}</span>
        </div>
      )}
      {data.burst > 0 && (
        <div style={{ color: "#dc322f", marginTop: 4, fontSize: 10 }}>⚡ {data.burst} burst event(s)</div>
      )}
    </div>
  );
}

// ── component ─────────────────────────────────────────────────────────────────

export default function AnomalyTimeline({ results }) {
  const data          = buildChartData(results);
  const totalAnomalies = data.reduce((s, b) => s + b.anomalyTotal, 0);
  const hasBursts     = data.some(b => b.burst > 0);
  const peakBucket    = data.reduce((best, b) => b.anomalyTotal > (best?.anomalyTotal || 0) ? b : best, null);

  if (data.length === 0) {
    return (
      <div style={{
        height: 260, display: "flex", alignItems: "center", justifyContent: "center",
        color: "var(--text-muted)", fontSize: 12,
      }}>
        No timeline data — cluster_summary timestamps unavailable
      </div>
    );
  }

  return (
    <div>
      {/* sub-header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
        <div style={{ fontSize: 11, color: "var(--text-muted)", opacity: 0.7 }}>
          {totalAnomalies} anomalous cluster(s) across {data.length} windows
          {hasBursts && <span style={{ color: "#dc322f", marginLeft: 8 }}>⚡ bursts detected</span>}
        </div>
        {/* legend */}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {["CRITICAL","HIGH","MEDIUM","LOW"].map(sev => (
            <div key={sev} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: SEV_COLOR[sev] }} />
              <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "monospace" }}>{sev}</span>
            </div>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 9, fill: "var(--text-muted)", fontFamily: "monospace" }}
            interval="preserveStartEnd"
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tick={{ fontSize: 9, fill: "var(--text-muted)" }}
            tickLine={false}
            axisLine={false}
            allowDecimals={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--border-subtle)", opacity: 0.3 }} />
          {peakBucket && peakBucket.anomalyTotal > 0 && (
            <ReferenceLine
              x={peakBucket.label}
              stroke="#dc322f" strokeDasharray="4 3" strokeWidth={1} opacity={0.5}
              label={{ value: "peak", position: "insideTopRight", fill: "#dc322f", fontSize: 9, fontFamily: "monospace", opacity: 0.8 }}
            />
          )}
          {/* stacked LOW → MEDIUM → HIGH → CRITICAL (bottom to top) */}
          {SEVERITY_ORDER.map((sev, i) => (
            <Bar
              key={sev}
              dataKey={sev}
              stackId="a"
              fill={SEV_COLOR[sev]}
              fillOpacity={sev === "LOW" || sev === "MEDIUM" ? 0.75 : sev === "HIGH" ? 0.85 : 0.92}
              maxBarSize={28}
              radius={i === SEVERITY_ORDER.length - 1 ? [3, 3, 0, 0] : [0, 0, 0, 0]}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}