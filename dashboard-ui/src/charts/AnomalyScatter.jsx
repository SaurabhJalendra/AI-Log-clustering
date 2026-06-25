// ─── AnomalyScatter.jsx ───────────────────────────────────────────────────────
// Scatter plot for the Anomalies page.
// X axis: rarity_score (how unusual the pattern is)
// Y axis: anomaly_score (overall severity score)
// Bubble size: count (how many log lines)
// Color: anomaly_label (CRITICAL / HIGH / MEDIUM / LOW)
//
// Usage — add near the top of AnomaliesPage in dashboard.jsx:
//   import AnomalyScatter from './AnomalyScatter';
//   <AnomalyScatter anomalies={anomalyDf} />
// ─────────────────────────────────────────────────────────────────────────────

import { useState, useEffect, useRef } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const SEV_COLOR = {
  CRITICAL: "#dc322f",
  HIGH:     "#cb4b16",
  MEDIUM:   "#b58900",
  LOW:      "#6c71c4",
};

const SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];

// Each tier gets 500ms of its own window within the 2s total.
// tierProgress[t] goes 0→1 as time passes through that window.
const TIER_COUNT  = 4;
const TOTAL_MS    = 2000;
const TIER_WIN    = TOTAL_MS / TIER_COUNT; // 500ms per tier

function useTierAnimation(dataKey) {
  // tierProgress[t] = 0..1, how many points in tier t to show (fractional)
  const [tierProgress, setTierProgress] = useState([0, 0, 0, 0]);
  const rafRef = useRef(null);
  const startRef = useRef(null);

  useEffect(() => {
    if (!dataKey) return;
    cancelAnimationFrame(rafRef.current);
    setTierProgress([0, 0, 0, 0]);
    startRef.current = null;

    const frame = (now) => {
      if (!startRef.current) startRef.current = now;
      const elapsed = now - startRef.current;
      const next = SEVERITY_ORDER.map((_, t) => {
        const tierElapsed = elapsed - t * TIER_WIN;
        if (tierElapsed <= 0) return 0;
        return Math.min(tierElapsed / TIER_WIN, 1);
      });
      setTierProgress(next);
      if (elapsed < TOTAL_MS + TIER_WIN) {
        rafRef.current = requestAnimationFrame(frame);
      } else {
        setTierProgress([1, 1, 1, 1]);
      }
    };
    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  }, [dataKey]);

  return tierProgress;
}

// ── custom tooltip ────────────────────────────────────────────────────────────

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div style={{
      background: "var(--bg-card)",
      border: `1px solid ${SEV_COLOR[d.anomaly_label] || "var(--border-default)"}44`,
      borderLeft: `3px solid ${SEV_COLOR[d.anomaly_label] || "var(--border-default)"}`,
      borderRadius: 6,
      padding: "10px 14px",
      fontSize: 11,
      fontFamily: "monospace",
      maxWidth: 240,
      pointerEvents: "none",
    }}>
      <div style={{ color: SEV_COLOR[d.anomaly_label], fontWeight: 700, marginBottom: 6 }}>
        {d.anomaly_label} — {d.event_id}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3px 16px", color: "var(--text-muted)" }}>
        <span>Anomaly score</span>
        <span style={{ color: "var(--text-secondary)" }}>{(d.anomaly_score * 100).toFixed(0)}</span>
        <span>Rarity score</span>
        <span style={{ color: "var(--text-secondary)" }}>{(d.rarity_score * 100).toFixed(0)}</span>
        <span>Event count</span>
        <span style={{ color: "var(--text-secondary)" }}>{d.count}</span>
        <span>Error %</span>
        <span style={{ color: d.error_pct > 50 ? "#dc322f" : "var(--text-secondary)" }}>
          {Math.round(d.error_pct || 0)}%
        </span>
        <span>Trend</span>
        <span style={{ color: d.trend_direction === "rising" ? "#dc322f" : d.trend_direction === "falling" ? "#3a7d44" : "var(--text-muted)" }}>
          {d.trend_direction === "rising" ? "↑" : d.trend_direction === "falling" ? "↓" : "→"} {d.trend_direction || "—"}
        </span>
        <span>Service</span>
        <span style={{ color: "var(--text-secondary)" }}>{d.top_source || "—"}</span>
      </div>
      {d.burst_detected && (
        <div style={{ marginTop: 6, color: "#dc322f", fontSize: 10 }}>⚡ burst detected</div>
      )}
    </div>
  );
}

// ── quadrant background shading via SVG ──────────────────────────────────────

function QuadrantShading() {
  // Rendered as a Recharts customized layer inside the chart
  // x=0..50 = rare, x=50..100 = common; y=0..50 = mild, y=50..100 = severe
  return null; // handled via CSS overlay below
}

// ── main component ────────────────────────────────────────────────────────────

export default function AnomalyScatter({ anomalies }) {
  const [activeLabel, setActiveLabel] = useState(null);

  // Trigger re-animation whenever the dataset changes (keyed by length)
  const tierProgress = useTierAnimation(anomalies?.length ?? 0);

  if (!anomalies || anomalies.length === 0) {
    return (
      <div style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border-default)",
        borderRadius: 8,
        padding: "16px 20px",
        height: 300,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--text-muted)",
        fontSize: 12,
        marginBottom: 20,
      }}>
        No anomaly data available
      </div>
    );
  }

  // Build animated dataset
  const animatedPoints = SEVERITY_ORDER.flatMap((label, t) => {
    const bucket = anomalies
      .filter(a => a.anomaly_label === label)
      .sort((a, b) => (a.rarity_score || 0) - (b.rarity_score || 0));
    const count = Math.floor(tierProgress[t] * bucket.length);
    return bucket.slice(0, count);
  });

  const displayed = activeLabel
    ? animatedPoints.filter(a => a.anomaly_label === activeLabel)
    : animatedPoints;

  const scatterData = displayed.map(a => ({
    x: Math.round((a.rarity_score || 0) * 100),
    y: Math.round((a.anomaly_score || 0) * 100),
    z: Math.sqrt(Math.max(a.count || 1, 1)),
    ...a,
  }));

  const countsByLabel = {};
  SEVERITY_ORDER.forEach(l => {
    countsByLabel[l] = anomalies.filter(a => a.anomaly_label === l).length;
  });

  // Chart margins — must match exactly so labels align with quadrant boundaries
  const chartMargin = { top: 16, right: 16, bottom: 24, left: 40 };

  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--border-default)",
      borderRadius: 8,
      padding: "16px 20px",
      marginBottom: 20,
    }}>
      {/* header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 12 }}>
        <div>
          <div style={{
            fontSize: 12, color: "var(--text-muted)",
            letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 2,
          }}>
            Anomaly Score vs Rarity
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", opacity: 0.7 }}>
            bubble size = event count · click legend to filter
          </div>
        </div>

        {/* interactive legend */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {SEVERITY_ORDER.filter(l => countsByLabel[l] > 0).map(label => (
            <button
              key={label}
              onClick={() => setActiveLabel(activeLabel === label ? null : label)}
              style={{
                display: "flex", alignItems: "center", gap: 5,
                background: activeLabel === label ? (SEV_COLOR[label] + "18") : "transparent",
                border: `1px solid ${activeLabel === label ? SEV_COLOR[label] : "var(--border-default)"}`,
                borderRadius: 4, padding: "3px 8px", cursor: "pointer",
                opacity: activeLabel && activeLabel !== label ? 0.35 : 1,
                transition: "all 0.15s",
              }}
            >
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: SEV_COLOR[label] }} />
              <span style={{ fontSize: 10, fontFamily: "monospace", color: SEV_COLOR[label] }}>
                {label} ({countsByLabel[label]})
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Quadrant labels OUTSIDE chart — row above chart, row below chart */}
      <div style={{ display: "flex", marginLeft: chartMargin.left, marginRight: chartMargin.right, marginBottom: 4 }}>
        <div style={{ flex: 1, textAlign: "left" }}>
          <span style={{ fontSize: 10, fontFamily: "monospace", fontWeight: 700, color: "#dc322f", letterSpacing: "0.04em" }}>HIGH RISK</span>
          <span style={{ fontSize: 9, color: "var(--text-muted)", marginLeft: 4 }}>rare + severe</span>
        </div>
        <div style={{ flex: 1, textAlign: "right" }}>
          <span style={{ fontSize: 9, color: "var(--text-muted)", marginRight: 4 }}>common + severe</span>
          <span style={{ fontSize: 10, fontFamily: "monospace", fontWeight: 700, color: "#cb4b16", letterSpacing: "0.04em" }}>NOISY</span>
        </div>
      </div>

      {/* chart with quadrant shading overlay */}
      <div style={{ position: "relative" }}>
        {/* Quadrant shading — 4 absolutely positioned divs behind the chart */}
        {/* These are approximate — the chart plot area starts after margin */}
        <div style={{ position: "absolute", top: chartMargin.top, left: chartMargin.left, right: chartMargin.right, bottom: chartMargin.bottom + 24, pointerEvents: "none", overflow: "hidden", borderRadius: 2 }}>
          {/* top-left: HIGH RISK — red tint */}
          <div style={{ position: "absolute", top: 0, left: 0, width: "50%", height: "50%", background: "rgba(220,50,47,0.04)" }} />
          {/* top-right: NOISY — orange tint */}
          <div style={{ position: "absolute", top: 0, right: 0, width: "50%", height: "50%", background: "rgba(203,75,22,0.04)" }} />
          {/* bottom-left: WATCH — yellow tint */}
          <div style={{ position: "absolute", bottom: 0, left: 0, width: "50%", height: "50%", background: "rgba(181,137,0,0.04)" }} />
          {/* bottom-right: ROUTINE — green tint */}
          <div style={{ position: "absolute", bottom: 0, right: 0, width: "50%", height: "50%", background: "rgba(58,125,68,0.04)" }} />
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart margin={chartMargin}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
            <CartesianGrid
              strokeDasharray="6 4"
              stroke="var(--border-default)"
              strokeWidth={0.8}
              horizontalPoints={[50]}
              verticalPoints={[50]}
            />
            <XAxis
              type="number"
              dataKey="x"
              domain={[0, 100]}
              tick={{ fontSize: 9, fill: "var(--text-muted)", fontFamily: "monospace" }}
              tickLine={false}
              label={{ value: "Rarity score →", position: "insideBottomRight", offset: -4, fill: "var(--text-muted)", fontSize: 9, fontFamily: "monospace" }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0, 100]}
              tick={{ fontSize: 9, fill: "var(--text-muted)", fontFamily: "monospace" }}
              tickLine={false}
              axisLine={false}
              label={{ value: "Anomaly score →", angle: -90, position: "insideLeft", offset: 16, fill: "var(--text-muted)", fontSize: 9, fontFamily: "monospace" }}
            />
            <ZAxis type="number" dataKey="z" range={[30, 600]} />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: "3 3", stroke: "var(--border-default)" }} />
            <Scatter data={scatterData}>
              {scatterData.map((d, i) => (
                <Cell
                  key={d.event_id || i}
                  fill={SEV_COLOR[d.anomaly_label] || "#888"}
                  fillOpacity={0.75}
                  stroke={SEV_COLOR[d.anomaly_label] || "#888"}
                  strokeWidth={d.burst_detected ? 2 : 0.5}
                  strokeOpacity={d.burst_detected ? 1 : 0.4}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Quadrant labels OUTSIDE chart — bottom row */}
      <div style={{ display: "flex", marginLeft: chartMargin.left, marginRight: chartMargin.right, marginTop: -20 }}>
        <div style={{ flex: 1, textAlign: "left" }}>
          <span style={{ fontSize: 10, fontFamily: "monospace", fontWeight: 700, color: "#b58900", letterSpacing: "0.04em" }}>WATCH</span>
          <span style={{ fontSize: 9, color: "var(--text-muted)", marginLeft: 4 }}>rare + mild</span>
        </div>
        <div style={{ flex: 1, textAlign: "right" }}>
          <span style={{ fontSize: 9, color: "var(--text-muted)", marginRight: 4 }}>common + mild</span>
          <span style={{ fontSize: 10, fontFamily: "monospace", fontWeight: 700, color: "#3a7d44", letterSpacing: "0.04em" }}>ROUTINE</span>
        </div>
      </div>

      {/* burst legend note */}
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 8, fontFamily: "monospace", opacity: 0.7 }}>
        thick border = burst detected
      </div>
    </div>
  );
}