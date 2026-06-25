// ─── AnomalyTimelineLine.jsx — line view (Critical + High only) ───────────────
// Used by OverviewPage when chartMode === "line"
// Reuses buildChartData from AnomalyTimeline

import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Area,
} from "recharts";
import { buildChartData, SEV_COLOR } from "./AnomalyTimeline";

// ── tooltip ───────────────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload || {};
  return (
    <div style={{
      background: "var(--bg-card)", border: "1px solid var(--border-default)",
      borderRadius: 6, padding: "10px 14px", fontSize: 11,
      fontFamily: "monospace", minWidth: 150,
    }}>
      <div style={{ color: "var(--text-muted)", marginBottom: 6 }}>{label} UTC</div>
      {["CRITICAL", "HIGH"].map(sev => (
        <div key={sev} style={{ display: "flex", justifyContent: "space-between", gap: 16, color: SEV_COLOR[sev], marginBottom: 2 }}>
          <span>{sev}</span>
          <span style={{ fontWeight: 700 }}>{data[sev] ?? 0}</span>
        </div>
      ))}
      {data.burst > 0 && (
        <div style={{ color: "#dc322f", marginTop: 6, fontSize: 10 }}>⚡ {data.burst} burst event(s)</div>
      )}
    </div>
  );
}

// ── component ─────────────────────────────────────────────────────────────────

export default function AnomalyTimelineLine({ results }) {
  const data       = buildChartData(results);
  const hasBursts  = data.some(b => b.burst > 0);
  const totalCrit  = data.reduce((s, b) => s + b.CRITICAL, 0);
  const totalHigh  = data.reduce((s, b) => s + b.HIGH,     0);
  const peakBucket = data.reduce((best, b) => {
    const v = b.CRITICAL + b.HIGH;
    return v > ((best?.CRITICAL || 0) + (best?.HIGH || 0)) ? b : best;
  }, null);

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
          {totalCrit} critical · {totalHigh} high across {data.length} windows
          {hasBursts && <span style={{ color: "#dc322f", marginLeft: 8 }}>⚡ bursts detected</span>}
        </div>
        {/* legend */}
        <div style={{ display: "flex", gap: 12 }}>
          {["CRITICAL", "HIGH"].map(sev => (
            <div key={sev} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 18, height: 2, background: SEV_COLOR[sev], borderRadius: 2 }} />
              <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "monospace" }}>{sev}</span>
            </div>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <defs>
            <linearGradient id="critGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#dc322f" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#dc322f" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="highGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#cb4b16" stopOpacity={0.12} />
              <stop offset="95%" stopColor="#cb4b16" stopOpacity={0} />
            </linearGradient>
          </defs>

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
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: "var(--border-default)", strokeWidth: 1 }} />

          {peakBucket && (peakBucket.CRITICAL + peakBucket.HIGH) > 0 && (
            <ReferenceLine
              x={peakBucket.label}
              stroke="#dc322f" strokeDasharray="4 3" strokeWidth={1} opacity={0.5}
              label={{ value: "peak", position: "insideTopRight", fill: "#dc322f", fontSize: 9, fontFamily: "monospace", opacity: 0.8 }}
            />
          )}

          {/* soft fill areas */}
          <Area dataKey="CRITICAL" stroke="none" fill="url(#critGrad)" />
          <Area dataKey="HIGH"     stroke="none" fill="url(#highGrad)" />

          {/* lines */}
          <Line
            dataKey="CRITICAL"
            stroke={SEV_COLOR.CRITICAL}
            strokeWidth={2}
            dot={{ r: 3, fill: SEV_COLOR.CRITICAL, strokeWidth: 0 }}
            activeDot={{ r: 5, fill: SEV_COLOR.CRITICAL, strokeWidth: 0 }}
            type="monotone"
          />
          <Line
            dataKey="HIGH"
            stroke={SEV_COLOR.HIGH}
            strokeWidth={2}
            strokeDasharray="5 3"
            dot={{ r: 3, fill: SEV_COLOR.HIGH, strokeWidth: 0 }}
            activeDot={{ r: 5, fill: SEV_COLOR.HIGH, strokeWidth: 0 }}
            type="monotone"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}