// ─── ServiceBar.jsx ───────────────────────────────────────────────────────────
// Horizontal stacked bar chart for the Anomalies page.
// Shows anomaly count per service, stacked by severity label.
// Sorted by total anomaly count descending.
//
// Usage in AnomaliesPage (dashboard.jsx), below AnomalyScatter:
//   import ServiceBar from './charts/ServiceBar';
//   <ServiceBar anomalies={anomalyDf} />
// ─────────────────────────────────────────────────────────────────────────────

import { useEffect, useRef, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, LabelList,
} from "recharts";

const SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];

const SEV_COLOR = {
  CRITICAL: "#dc322f",
  HIGH:     "#cb4b16",
  MEDIUM:   "#b58900",
  LOW:      "#6c71c4",
};

// Each severity tier animates in sequence: CRITICAL first, then HIGH, MEDIUM, LOW.
// Each tier grows from 0 to its target over 750ms, staggered 500ms apart.
const SB_TIER_DELAY = 500;
const SB_TIER_DUR   = 750;
const SB_TOTAL_MS   = SB_TIER_DELAY * (SEVERITY_ORDER.length - 1) + SB_TIER_DUR;

function easeOut(t) { return 1 - Math.pow(1 - t, 2); }

function useBarAnimation(dataKey, targets) {
  const [animData, setAnimData] = useState(null);
  const rafRef   = useRef(null);
  const startRef = useRef(null);

  useEffect(() => {
    if (!targets?.length) return;
    cancelAnimationFrame(rafRef.current);
    startRef.current = null;

    // Start from zeros
    setAnimData(targets.map(row => ({
      ...row, CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0,
    })));

    const frame = (now) => {
      if (!startRef.current) startRef.current = now;
      const elapsed = now - startRef.current;

      setAnimData(targets.map(row => {
        const animated = { ...row };
        SEVERITY_ORDER.forEach((sev, t) => {
          const te = elapsed - t * SB_TIER_DELAY;
          if (te <= 0) { animated[sev] = 0; return; }
          const p = easeOut(Math.min(te / SB_TIER_DUR, 1));
          animated[sev] = +(row[sev] * p).toFixed(3);
        });
        return animated;
      }));

      if (elapsed < SB_TOTAL_MS) {
        rafRef.current = requestAnimationFrame(frame);
      } else {
        setAnimData(targets); // snap to exact final values
      }
    };
    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  }, [dataKey]); // eslint-disable-line react-hooks/exhaustive-deps

  return animData;
}

// ── build chart data ──────────────────────────────────────────────────────────

function buildData(anomalies) {
  const serviceMap = {};

  anomalies.forEach(a => {
    const svc = a.top_source || "unknown";
    if (!serviceMap[svc]) {
      serviceMap[svc] = { service: svc, CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0, total: 0 };
    }
    const label = a.anomaly_label;
    if (SEVERITY_ORDER.includes(label)) {
      serviceMap[svc][label] += 1;
      serviceMap[svc].total  += 1;
    }
  });

  return Object.values(serviceMap)
    .sort((a, b) => b.total - a.total);
}

// ── custom tooltip ────────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload || {};
  return (
    <div style={{
      background: "var(--bg-card)",
      border: "1px solid var(--border-default)",
      borderRadius: 6,
      padding: "10px 14px",
      fontSize: 11,
      fontFamily: "monospace",
      minWidth: 160,
    }}>
      <div style={{ color: "var(--text-secondary)", fontWeight: 600, marginBottom: 6 }}>
        {label}
      </div>
      {SEVERITY_ORDER.map(sev => data[sev] > 0 && (
        <div key={sev} style={{
          display: "flex", justifyContent: "space-between",
          gap: 16, color: SEV_COLOR[sev], marginBottom: 2,
        }}>
          <span>{sev}</span>
          <span style={{ fontWeight: 700 }}>{data[sev]}</span>
        </div>
      ))}
      <div style={{
        display: "flex", justifyContent: "space-between",
        gap: 16, color: "var(--text-muted)",
        borderTop: "1px solid var(--border-subtle)",
        marginTop: 4, paddingTop: 4,
      }}>
        <span>total</span>
        <span>{data.total}</span>
      </div>
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export default function ServiceBar({ anomalies }) {
  // Hooks must be called unconditionally — before any early returns
  const targets  = (anomalies && anomalies.length > 0) ? buildData(anomalies) : [];
  const animData = useBarAnimation(anomalies?.length ?? 0, targets);

  if (!anomalies || anomalies.length === 0) {
    return (
      <div style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border-default)",
        borderRadius: 8,
        padding: "16px 20px",
        height: 200,
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

  const data        = animData || targets;
  const chartHeight = Math.max(data.length * 44 + 40, 160);
  // Pin x-axis max to the real (final) max total so the axis never rescales during animation
  const xMax = Math.max(...targets.map(r => r.total), 1);
  // Dynamic left margin — prevent label truncation for long service names
  const maxLabelLen = Math.max(...(targets.map(r => (r.service || "").length)), 4);
  const yAxisWidth  = Math.max(90, Math.min(maxLabelLen * 7, 180));

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
            Anomalies per Service
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", opacity: 0.7 }}>
            {data.length} services · sorted by total anomaly count
          </div>
        </div>

        {/* legend */}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {SEVERITY_ORDER.map(sev => (
            <div key={sev} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: SEV_COLOR[sev] }} />
              <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "monospace" }}>{sev}</span>
            </div>
          ))}
        </div>
      </div>

      {/* chart */}
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 48, bottom: 0, left: 8 }}
          barCategoryGap="30%"
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
            dataKey="service"
            tick={{ fontSize: 11, fill: "var(--text-secondary)", fontFamily: "monospace", textAnchor: "end" }}
            tickLine={false}
            axisLine={false}
            width={yAxisWidth}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--border-subtle)", opacity: 0.4 }} />

          {SEVERITY_ORDER.map((sev, i) => (
            <Bar
              key={sev}
              dataKey={sev}
              stackId="svc"
              fill={SEV_COLOR[sev]}
              maxBarSize={20}
              radius={
                i === SEVERITY_ORDER.length - 1
                  ? [0, 3, 3, 0]
                  : [0, 0, 0, 0]
              }
            >
              {/* show total count label only on the last (rightmost) segment */}
              {i === SEVERITY_ORDER.length - 1 && (
                <LabelList
                  dataKey="total"
                  position="right"
                  style={{
                    fontSize: 10,
                    fontFamily: "monospace",
                    fill: "var(--text-muted)",
                  }}
                />
              )}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}