import { useState } from "react";
import { ResponsiveContainer, Treemap } from "recharts";
import AnomalyTimeline     from "../charts/AnomalyTimeline";
import AnomalyTimelineLine from "../charts/AnomalyTimelineLine";
import { SIGNAL_COLOR, SEV_DOT, FONT_SIZE, FONT } from "../Theme";
import { Badge, ClickableCard } from "../Components";

// ─── OVERVIEW PAGE ────────────────────────────────────────────────────────────

export default function OverviewPage({ results, onNavigate }) {
  const [chartMode, setChartMode] = useState("bar"); // "bar" | "line"

  const anomalyDf      = results?.anomalies       || [];
  const incidents      = results?.incidents        || [];
  const runInfo        = results?.run_info         || {};
  const clusterSummary = results?.cluster_summary  || [];

  const anomalyCount  = results?.anomaly_count  || 0;
  const incidentCount = results?.incident_count || 0;
  const totalClusters = runInfo.total_clusters  || 1;
  const healthScore   = Math.max(
    Math.round(100 - (anomalyCount / totalClusters) * 100 * 0.5 - (runInfo.incident_severity_counts?.CRITICAL || 0) * 8),
    0
  );

  // ── KPI strip ──────────────────────────────────────────────────────────────
  const kpis = [
    { label: "Health Score",     value: `${healthScore}%`,
      sub: "based on anomaly density",
      color: healthScore > 70 ? "#3a7d44" : healthScore > 40 ? "#b58900" : "#dc322f" },
    { label: "Active Anomalies", value: anomalyCount,
      sub: `${incidentCount} incidents`,  color: "#dc322f" },
    { label: "Log Volume",
      value: `${((runInfo.parsed_ok || 0) / 1000).toFixed(1)}K`,
      sub: `${(runInfo.total_lines || 0).toLocaleString()} total lines`, color: "var(--accent-blue)" },
    { label: "Parse Rate",
      value: runInfo.total_lines
        ? `${Math.round((runInfo.parsed_ok / runInfo.total_lines) * 100)}%` : "—",
      sub: `${(runInfo.noise_lines || 0).toLocaleString()} noise lines filtered`, color: "var(--accent-green)" },
  ];

  // ── Domain treemap ─────────────────────────────────────────────────────────
  const anomalySignal = {};
  anomalyDf.forEach(a => {
    anomalySignal[String(a.event_id)] = (
      a.anomaly_label === "CRITICAL" || a.anomaly_label === "HIGH" ? "high"
      : a.anomaly_label === "MEDIUM" ? "medium" : "none"
    );
  });

  const domainStats = {};
  const ensureDomain = (d) => {
    if (!domainStats[d]) domainStats[d] = { name: d, size: 0, anomalous: 0, total: 0, worstSig: "none" };
  };
  const sigRank    = { high: 3, medium: 2, low: 1, none: 0 };
  const sourceRows = clusterSummary.length > 0 ? clusterSummary : anomalyDf;

  sourceRows.forEach(row => {
    const d = row.domain || "other";
    ensureDomain(d);
    domainStats[d].size  += Number(row.total_log_count || row.count || 1);
    domainStats[d].total += 1;

    let sig;
    if (clusterSummary.length > 0) {
      sig = (row.is_routine === true || row.is_routine === "True" || row.is_routine === 1)
        ? "none"
        : anomalySignal[String(row.cluster_id ?? "")] || "none";
    } else {
      sig = row.anomaly_label === "CRITICAL" || row.anomaly_label === "HIGH" ? "high"
          : row.anomaly_label === "MEDIUM" ? "medium" : "none";
    }

    if (sig !== "none") {
      domainStats[d].anomalous += 1;
      if ((sigRank[sig] || 0) > (sigRank[domainStats[d].worstSig] || 0))
        domainStats[d].worstSig = sig;
    }
  });

  const rawTreemapData = Object.values(domainStats).map(ds => {
    const ratio  = ds.total > 0 ? ds.anomalous / ds.total : 0;
    const pct    = Math.round(ratio * 100);
    const signal = clusterSummary.length > 0
      ? (ratio >= 0.5 ? "high" : ratio >= 0.2 ? "medium" : ratio >= 0.05 ? "low" : "none")
      : ds.worstSig;
    return { name: ds.name, size: ds.size, signal, ratio: pct, _total: ds.total, _anomalous: ds.anomalous };
  });

  // ── Enforce minimum tile size so no orphan slivers are left ──────────────
  // 1. Compute a floor: each domain must represent at least (1 / n) * 0.4 of total
  //    so the smallest tile is never less than ~40% of an equal share.
  // 2. Domains too tiny to render cleanly get merged into the existing "other"
  //    bucket (or a synthetic one) so Recharts has fewer, larger tiles to pack.
  const totalSize     = rawTreemapData.reduce((s, d) => s + d.size, 0) || 1;
  const domainCount   = rawTreemapData.length;
  const MIN_SHARE     = 0.03; // each domain must be ≥ 3 % of total area
  const minSize       = totalSize * MIN_SHARE;

  const mainTiles  = rawTreemapData.filter(d => d.size >= minSize);
  const tinyTiles  = rawTreemapData.filter(d => d.size <  minSize);

  let treemapData;
  if (tinyTiles.length === 0) {
    // Nothing to merge — just apply the floor so Recharts never gets size=0
    treemapData = mainTiles.map(d => ({ ...d, size: Math.max(d.size, minSize) }));
  } else {
    // Merge tiny domains into the existing "other" tile or create one
    const mergedSize      = tinyTiles.reduce((s, d) => s + d.size, 0);
    const mergedAnomalous = tinyTiles.reduce((s, d) => s + d._anomalous, 0);
    const mergedTotal     = tinyTiles.reduce((s, d) => s + d._total, 0);
    const mergedRatio     = mergedTotal > 0 ? mergedAnomalous / mergedTotal : 0;
    const mergedPct       = Math.round(mergedRatio * 100);
    const mergedSignal    = clusterSummary.length > 0
      ? (mergedRatio >= 0.5 ? "high" : mergedRatio >= 0.2 ? "medium" : mergedRatio >= 0.05 ? "low" : "none")
      : (tinyTiles.some(d => d.signal === "high") ? "high"
         : tinyTiles.some(d => d.signal === "medium") ? "medium"
         : tinyTiles.some(d => d.signal === "low") ? "low" : "none");

    const existingOther = mainTiles.find(d => d.name === "other");
    if (existingOther) {
      existingOther.size        = Math.max(existingOther.size + mergedSize, minSize);
      existingOther._anomalous += mergedAnomalous;
      existingOther._total     += mergedTotal;
      const nr = existingOther._total > 0 ? existingOther._anomalous / existingOther._total : 0;
      existingOther.ratio = Math.round(nr * 100);
      treemapData = mainTiles.map(d => ({ ...d, size: Math.max(d.size, minSize) }));
    } else {
      treemapData = [
        ...mainTiles.map(d => ({ ...d, size: Math.max(d.size, minSize) })),
        { name: "other", size: Math.max(mergedSize, minSize), signal: mergedSignal, ratio: mergedPct },
      ];
    }
  }

  const CustomTreemapContent = ({ x, y, width, height, name, signal, ratio }) => {
    if (width < 30 || height < 20) return null;
    const bg = signal === "high"   ? "var(--treemap-bg-high)"
             : signal === "medium" ? "var(--treemap-bg-medium)"
             : signal === "low"    ? "var(--treemap-bg-low)"
             :                       "var(--treemap-bg-none)";
    const borderColor = SIGNAL_COLOR[signal] || "#333";
    const showSub     = width > 80 && height > 44 && ratio !== undefined;
    return (
      <g>
        <rect x={x+1} y={y+1} width={width-2} height={height-2} fill={bg} stroke={borderColor} strokeWidth={1} rx={3} />
        {width > 60 && (
          <text x={x+width/2} y={showSub ? y+height/2-8 : y+height/2}
            textAnchor="middle" dominantBaseline="middle"
            fill={borderColor} fontSize={Math.min(FONT_SIZE.md, width/8)} fontFamily="monospace">
            {name}
          </text>
        )}
        {showSub && (
          <text x={x+width/2} y={y+height/2+10}
            textAnchor="middle" dominantBaseline="middle"
            fill={borderColor} fontSize={Math.min(FONT_SIZE.sm, width/10)} fontFamily="monospace" opacity={0.65}>
            {ratio}% anomalous
          </text>
        )}
      </g>
    );
  };

  // ── toggle button style ────────────────────────────────────────────────────
  const toggleBtn = (mode) => ({
    background: chartMode === mode ? "var(--surface-info)" : "none",
    border: `1px solid ${chartMode === mode ? "var(--accent-blue)" : "var(--border-default)"}`,
    color: chartMode === mode ? "var(--accent-blue)" : "var(--text-muted)",
    borderRadius: 4, padding: "3px 12px",
    fontSize: FONT_SIZE.base, cursor: "pointer",
    fontFamily: "monospace", transition: "all 0.15s",
  });

  return (
    <div>

      {/* ── KPI Strip — read-only display, intentionally non-clickable ────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
        {kpis.map(({ label, value, sub, color }) => (
          <div key={label} style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", cursor: "default" }}>
            <div style={{ fontSize: FONT_SIZE.sm, color: "var(--text-muted)", marginBottom: 6, letterSpacing: "0.06em", textTransform: "uppercase" }}>{label}</div>
            <div style={{ fontSize: FONT_SIZE.hero, fontWeight: 700, color, fontFamily: FONT.mono }}>{value}</div>
            <div style={{ fontSize: FONT_SIZE.sm, color: "var(--text-muted)", marginTop: 4 }}>{sub}</div>
          </div>
        ))}
      </div>

      {/* ── Timeline + Incidents ─────────────────────────────────────────────── */}
      {/* Row is pinned to 320px tall so the incidents panel never stretches the grid row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 16, marginBottom: 16, height: 320 }}>

        {/* Chart card with bar / line toggle */}
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4, flexShrink: 0 }}>
            <div>
              <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", letterSpacing: "0.05em", textTransform: "uppercase" }}>
                Anomaly Timeline — 30-min buckets
              </div>
              {chartMode === "line" && (
                <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", opacity: 0.65, marginTop: 2 }}>
                  critical &amp; high only
                </div>
              )}
            </div>
            <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
              <button style={toggleBtn("bar")}  onClick={() => setChartMode("bar")}>▦ bars</button>
              <button style={toggleBtn("line")} onClick={() => setChartMode("line")}>╱ lines</button>
            </div>
          </div>

          {/* Chart fills remaining space in the fixed-height card */}
          <div style={{ flex: 1, minHeight: 0 }}>
            {chartMode === "bar"
              ? <AnomalyTimeline     results={results} />
              : <AnomalyTimelineLine results={results} />
            }
          </div>
        </div>

        {/* Open Incidents — fixed height matches row, list scrolls internally */}
        <ClickableCard
          onClick={() => onNavigate && onNavigate("incidents")}
          style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", display: "flex", flexDirection: "column", overflow: "hidden" }}
        >
          {/* Header — never scrolls away */}
          <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", marginBottom: 12, letterSpacing: "0.05em", textTransform: "uppercase", flexShrink: 0 }}>
            Open Incidents
          </div>
          {/* Scrollable list */}
          <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
            {incidents.length === 0 && (
              <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.md, marginTop: 20 }}>No incidents detected</div>
            )}
            {incidents.map(inc => (
              <div key={inc.incident_id} style={{ borderLeft: `3px solid ${SEV_DOT[inc.incident_severity]}`, paddingLeft: 10, marginBottom: 12 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                  <Badge label={inc.incident_severity} type={inc.incident_severity} />
                  <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", fontFamily: "monospace" }}>{inc.incident_id}</span>
                </div>
                <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-primary)", fontWeight: 600 }}>
                  {String(inc.primary_domain || "—").toUpperCase()} — {inc.root_cause_service || "unknown"}
                </div>
                <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginTop: 2, fontFamily: "monospace" }}>
                  {inc.n_clusters || 0} cluster(s)
                  {inc.duration_minutes ? ` · ${inc.duration_minutes} min` : ""}
                  {inc.recurrence_flag  ? " · ↻ recurring" : ""}
                </div>
              </div>
            ))}
          </div>
        </ClickableCard>
      </div>

      {/* ── Domain Map + Top Anomalies ───────────────────────────────────────── */}
      {/* Row pinned to 300px so Top Anomalies can't grow taller than the treemap */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, height: 300 }}>

        {/* Domain treemap */}
        <ClickableCard
          onClick={() => onNavigate && onNavigate("clusters")}
          style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", display: "flex", flexDirection: "column", overflow: "hidden" }}
        >
          <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", marginBottom: 12, letterSpacing: "0.05em", textTransform: "uppercase", flexShrink: 0 }}>
            Domain Map — {treemapData.length} domains{clusterSummary.length > 0 ? " (all clusters)" : " (anomalies only)"}
          </div>
          {/* Treemap fills available height */}
          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <Treemap data={treemapData} dataKey="size" content={<CustomTreemapContent />} />
            </ResponsiveContainer>
          </div>
          <div style={{ display: "flex", gap: 12, marginTop: 10, flexWrap: "wrap", flexShrink: 0 }}>
            {["none","low","medium","high"].map(s => (
              <div key={s} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: SIGNAL_COLOR[s] }} />
                <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)" }}>{s}</span>
              </div>
            ))}
          </div>
        </ClickableCard>

        {/* Top anomalies — header fixed, list scrolls */}
        <ClickableCard
          onClick={() => onNavigate && onNavigate("anomalies")}
          style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", display: "flex", flexDirection: "column", overflow: "hidden" }}
        >
          <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", marginBottom: 12, letterSpacing: "0.05em", textTransform: "uppercase", flexShrink: 0 }}>Top Anomalies</div>
          {/* Scrollable list */}
          <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
            {anomalyDf.length === 0 && (
              <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.md }}>No anomalies detected</div>
            )}
            {anomalyDf.slice(0, 5).map(row => (
              <div key={row.event_id} style={{ display: "flex", alignItems: "flex-start", gap: 10, marginBottom: 10, paddingBottom: 10, borderBottom: "1px solid var(--border-subtle)" }}>
                <Badge label={row.anomaly_label} type={row.anomaly_label} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-primary)", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {row.sample_message || row.sample_template || row.event_id}
                  </div>
                  <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginTop: 2, fontFamily: "monospace" }}>
                    {row.top_source || "—"}
                    {row.trend_direction === "rising"  && <span style={{ color: "#dc322f", marginLeft: 8 }}>↑ rising</span>}
                    {row.trend_direction === "falling" && <span style={{ color: "#3a7d44", marginLeft: 8 }}>↓ falling</span>}
                  </div>
                </div>
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div style={{ fontSize: FONT_SIZE.md, fontWeight: 700, fontFamily: "monospace", color: SIGNAL_COLOR[row.anomaly_label] }}>
                    {(row.count || 0).toLocaleString()}
                  </div>
                  <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)" }}>events</div>
                </div>
              </div>
            ))}
          </div>
        </ClickableCard>
      </div>

    </div>
  );
}