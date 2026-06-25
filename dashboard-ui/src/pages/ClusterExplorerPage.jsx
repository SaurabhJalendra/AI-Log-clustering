import { useState, useEffect, useRef } from "react";
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
  ZAxis, CartesianGrid, Tooltip, Cell,
} from "recharts";
import { SIGNAL_COLOR, FONT_SIZE, FONT } from "../Theme";
import { Badge, ScoreBar } from "../Components";

// ─── CLUSTER ANIMATION CONSTANTS ─────────────────────────────────────────────
const CL_TIERS    = ["none", "medium", "high"];
const CL_TOTAL_MS = 2000;
const CL_TIER_WIN = CL_TOTAL_MS / CL_TIERS.length; // ~667ms per tier

// ─── CLUSTER EXPLORER PAGE ────────────────────────────────────────────────────

export default function ClusterExplorerPage({ results }) {
  const anomalyDf      = results?.anomalies       || [];
  const clusterSummary = results?.cluster_summary || [];
  const [selected, setSelected]         = useState(null);
  const [excluded, setExcluded]         = useState([]);
  const [domainFilter, setDomainFilter] = useState("ALL");
  const [visibleIds, setVisibleIds]     = useState(null);
  const rafRef   = useRef(null);
  const startRef = useRef(null);

  const anomalyByCluster = {};
  anomalyDf.forEach(a => { anomalyByCluster[String(a.event_id)] = a; });

  let clusters;
  if (clusterSummary.length > 0) {
    clusters = clusterSummary.map(cs => {
      const cid   = String(cs.cluster_id);
      const anom  = anomalyByCluster[cid] || {};
      const label = anom.anomaly_label || "INFO";
      const signal = label === "CRITICAL" || label === "HIGH" ? "high"
                   : label === "MEDIUM" ? "medium" : "none";
      return {
        id: cid, label: cs.cluster_label || cid,
        domain: cs.domain || anom.domain || "other",
        count: Number(cs.total_log_count || cs.count || anom.count || 1),
        signal, anomaly_label: label,
        anomaly_score: anom.anomaly_score || 0,
        error_pct: anom.error_pct || 0,
        dominant_severity: cs.dominant_severity || anom.dominant_severity || "INFO",
        sample_message: cs.sample_template || anom.sample_message || anom.sample_template || "",
        top_source: anom.top_source || "—",
        services: Array.isArray(cs.services) ? cs.services : [],
      };
    });
  } else {
    clusters = anomalyDf.map(a => {
      const label  = a.anomaly_label || "INFO";
      const signal = label === "CRITICAL" || label === "HIGH" ? "high"
                   : label === "MEDIUM" ? "medium" : "none";
      return {
        id: String(a.event_id), label: String(a.event_id),
        domain: a.domain || "other", count: a.count || 1, signal,
        anomaly_label: label, anomaly_score: a.anomaly_score || 0,
        error_pct: a.error_pct || 0,
        dominant_severity: a.dominant_severity || "INFO",
        sample_message: a.sample_message || "", top_source: a.top_source || "—",
        services: [],
      };
    });
  }

  // ── Reveal animation ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!clusters.length) return;
    cancelAnimationFrame(rafRef.current);
    startRef.current = null;

    const tierBuckets = CL_TIERS.map(tier =>
      clusters.filter(c => c.signal === tier).sort((a, b) => a.count - b.count)
    );

    const frame = (now) => {
      if (!startRef.current) startRef.current = now;
      const elapsed = now - startRef.current;
      const ids = new Set();
      CL_TIERS.forEach((_, t) => {
        const te = elapsed - t * CL_TIER_WIN;
        if (te <= 0) return;
        const progress = Math.min(te / CL_TIER_WIN, 1);
        const count = Math.floor(progress * tierBuckets[t].length);
        tierBuckets[t].slice(0, count).forEach(c => ids.add(c.id));
      });
      setVisibleIds(new Set(ids));
      if (elapsed < CL_TOTAL_MS + CL_TIER_WIN) {
        rafRef.current = requestAnimationFrame(frame);
      } else {
        setVisibleIds(null);
      }
    };
    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clusters.length]);

  const allDomains     = ["ALL", ...Array.from(new Set(clusters.map(c => c.domain).filter(Boolean)))];
  const visibleClusters = clusters.filter(c =>
    (domainFilter === "ALL" || c.domain === domainFilter) &&
    !excluded.includes(c.id) &&
    (visibleIds === null || visibleIds.has(c.id))
  );

  const sel      = clusters.find(c => c.id === selected);
  const maxCount = Math.max(...clusters.map(c => c.count), 1);
  const scatterData = visibleClusters.map(c => ({
    x: Math.log1p(c.count) / Math.log1p(maxCount) * 90 + 5,
    y: (c.anomaly_score || 0) * 85 + 5,
    z: Math.sqrt(c.count || 1),
    name: c.label, id: c.id, signal: c.signal,
  }));

  return (
    <div>
      {/* ── Scorecard row ── */}
      <div className="scorecard-row" style={{ gridTemplateColumns: "repeat(5, 1fr)", marginBottom: 16 }}>
        {[
          { label: "Clusters",           value: clusters.length },
          { label: "Silhouette Score",   value: results?.run_info?.silhouette_score != null ? Number(results.run_info.silhouette_score).toFixed(3) : "—" },
          { label: "Isolated Templates", value: results?.run_info?.n_isolated ?? "—" },
          { label: "Cluster Threshold",  value: results?.run_info?.cluster_threshold != null ? Number(results.run_info.cluster_threshold).toFixed(2) : "—" },
          { label: "Suspicious Splits",  value: results?.suspicious_splits?.length ?? 0 },
        ].map(({ label, value }) => (
          <div key={label} style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "12px 14px" }}>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
            <div style={{ fontSize: FONT_SIZE.h1, fontWeight: 700, color: "var(--text-primary)", fontFamily: FONT.mono }}>{value}</div>
          </div>
        ))}
      </div>

      {/* ── Domain filter pills ── */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap", alignItems: "center" }}>
        {allDomains.map(d => (
          <button key={d} onClick={() => setDomainFilter(d)} style={{
            background: domainFilter === d ? "var(--surface-info)" : "var(--bg-card)",
            border: `1px solid ${domainFilter === d ? "var(--accent-blue)" : "var(--border-default)"}`,
            color: domainFilter === d ? "var(--accent-blue)" : "var(--text-muted)",
            borderRadius: 4, padding: "3px 12px", fontSize: FONT_SIZE.base, cursor: "pointer", fontFamily: "monospace",
          }}>{d}</button>
        ))}
        <span style={{ marginLeft: "auto", fontSize: FONT_SIZE.base, color: "var(--text-muted)", alignSelf: "center" }}>
          {visibleClusters.length} / {clusters.length} clusters
          {[
            { sig: "high",   color: "#dc322f" },
            { sig: "medium", color: "#b58900" },
            { sig: "none",   color: "#3a7d44" },
          ].map(({ sig, color }) => {
            const n = clusters.filter(c => c.signal === sig).length;
            return n > 0 ? <span key={sig} style={{ marginLeft: 8, color }}>● {n} {sig}</span> : null;
          })}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 16 }}>
        {/* ── Scatter chart ── */}
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px" }}>
          <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", marginBottom: 4, letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Cluster Map — click to inspect
          </div>
          <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 12, opacity: 0.6 }}>
            x-axis: log volume · y-axis: anomaly score · bubble size: event count
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 8, right: 8, bottom: 24, left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis type="number" dataKey="x" domain={[0, 100]} tick={{ fontSize: FONT_SIZE.xs, fill: "var(--text-muted)" }} label={{ value: "Volume (log scale)", position: "insideBottom", offset: -12, fill: "var(--text-muted)", fontSize: FONT_SIZE.xs }} />
              <YAxis type="number" dataKey="y" domain={[0, 100]} tick={{ fontSize: FONT_SIZE.xs, fill: "var(--text-muted)" }} label={{ value: "Anomaly Score", angle: -90, position: "insideLeft", offset: 12, fill: "var(--text-muted)", fontSize: FONT_SIZE.xs }} />
              <ZAxis type="number" dataKey="z" range={[40, 800]} />
              <Tooltip cursor={false} content={({ payload }) => {
                if (!payload?.[0]) return null;
                const d = payload[0].payload;
                const c = clusters.find(cl => cl.id === d.id);
                return (
                  <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", padding: "8px 12px", borderRadius: 6, fontSize: FONT_SIZE.base, maxWidth: 240 }}>
                    <div style={{ color: SIGNAL_COLOR[d.signal], marginBottom: 4, fontFamily: "monospace" }}>{d.id}</div>
                    <div style={{ color: "var(--text-muted)" }}>Domain: {c?.domain || "—"}</div>
                    <div style={{ color: "var(--text-muted)" }}>Count: {(c?.count || 0).toLocaleString()}</div>
                    <div style={{ color: "var(--text-muted)" }}>Score: {((c?.anomaly_score || 0) * 100).toFixed(0)}</div>
                  </div>
                );
              }} />
              <Scatter data={scatterData} onClick={d => setSelected(d.id === selected ? null : d.id)}>
                {scatterData.map(d => (
                  <Cell key={d.id} fill={SIGNAL_COLOR[d.signal]} opacity={d.id === selected ? 1 : 0.65}
                    stroke={d.id === selected ? "var(--text-heading)" : "none"} strokeWidth={2} style={{ cursor: "pointer" }} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* ── Detail panel ── */}
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", overflowY: "auto", maxHeight: 520 }}>
          {sel ? (
            <>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <Badge label={sel.anomaly_label} type={sel.anomaly_label} />
                {sel.signal !== "none" && (
                  <span style={{ fontSize: FONT_SIZE.xs, fontFamily: "monospace", color: "var(--accent-red)" }}>
                    {sel.is_routine === true || sel.is_routine === "True" ? "ROUTINE" : "ANOMALOUS"}
                  </span>
                )}
              </div>
              <div style={{ fontFamily: "monospace", fontSize: FONT_SIZE.base, color: "var(--text-muted)", marginBottom: 14, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {sel.id}
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 14 }}>
                {[
                  { label: "Count",        v: (sel.count || 0).toLocaleString() },
                  { label: "Error %",      v: `${Math.round(sel.error_pct || 0)}%` },
                  { label: "Domain",       v: sel.domain },
                  { label: "Dominant Sev", v: sel.dominant_severity },
                ].map(({ label, v }) => (
                  <div key={label} style={{ background: "var(--bg-card-deep)", borderRadius: 6, padding: "8px 10px" }}>
                    <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: FONT_SIZE.lg, color: "var(--text-primary)", fontFamily: "monospace" }}>{v}</div>
                  </div>
                ))}
              </div>

              {(sel.first_seen || sel.last_seen) && (
                <div style={{ background: "var(--bg-card-deep)", borderRadius: 6, padding: "8px 10px", marginBottom: 14 }}>
                  <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 6 }}>TIME RANGE</div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: FONT_SIZE.base, fontFamily: "monospace" }}>
                    <div>
                      <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 2 }}>FIRST SEEN</div>
                      <div style={{ color: "var(--text-secondary)" }}>{sel.first_seen ? String(sel.first_seen).slice(11, 19) + " UTC" : "—"}</div>
                    </div>
                    <div style={{ color: "var(--text-muted)", alignSelf: "center" }}>→</div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 2 }}>LAST SEEN</div>
                      <div style={{ color: "var(--text-secondary)" }}>{sel.last_seen ? String(sel.last_seen).slice(11, 19) + " UTC" : "—"}</div>
                    </div>
                  </div>
                </div>
              )}

              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 4 }}>ANOMALY SCORE</div>
                <ScoreBar score={sel.anomaly_score || 0} />
              </div>

              {sel.severity_distribution && (() => {
                const dist   = typeof sel.severity_distribution === "string"
                  ? JSON.parse(sel.severity_distribution.replace(/'/g, '"'))
                  : sel.severity_distribution;
                const levels = ["DEBUG","INFO","WARN","ERROR","FATAL"];
                const colors = { DEBUG: "#6c71c4", INFO: "#268bd2", WARN: "#b58900", ERROR: "#dc322f", FATAL: "#dc322f" };
                const total  = levels.reduce((s, l) => s + (dist[l] || 0), 0);
                if (total === 0) return null;
                return (
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 6 }}>SEVERITY DISTRIBUTION</div>
                    <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", gap: 1 }}>
                      {levels.map(l => {
                        const pct = total > 0 ? (dist[l] || 0) / total * 100 : 0;
                        return pct > 0 ? <div key={l} style={{ width: `${pct}%`, background: colors[l], transition: "width 0.3s" }} title={`${l}: ${dist[l]}`} /> : null;
                      })}
                    </div>
                    <div style={{ display: "flex", gap: 8, marginTop: 6, flexWrap: "wrap" }}>
                      {levels.filter(l => (dist[l] || 0) > 0).map(l => (
                        <div key={l} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                          <div style={{ width: 6, height: 6, borderRadius: 1, background: colors[l] }} />
                          <span style={{ fontSize: FONT_SIZE.xs, fontFamily: "monospace", color: "var(--text-muted)" }}>{l} {dist[l]}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {sel.anomaly_reason && sel.anomaly_reason !== "nan" && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 4 }}>WHY ANOMALOUS</div>
                  <div style={{ fontSize: FONT_SIZE.base, color: "var(--accent-yellow)", background: "var(--surface-warn)", border: "1px solid var(--surface-warn-border)", borderRadius: 4, padding: "6px 8px", fontFamily: "monospace", lineHeight: 1.5 }}>
                    {sel.anomaly_reason}
                  </div>
                </div>
              )}

              {sel.services?.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 4 }}>SERVICES</div>
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                    {sel.services.map(s => (
                      <span key={s} style={{ background: "var(--surface-info)", border: "1px solid var(--surface-info-border)", borderRadius: 4, color: "var(--accent-blue)", padding: "1px 8px", fontSize: FONT_SIZE.xs, fontFamily: "monospace" }}>{s}</span>
                    ))}
                  </div>
                </div>
              )}

              <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 4 }}>LOG TEMPLATE</div>
              <div style={{ fontSize: FONT_SIZE.base, fontFamily: "monospace", color: "var(--text-secondary)", background: "var(--bg-card-deep)", padding: "6px 8px", borderRadius: 4, lineHeight: 1.5, wordBreak: "break-word" }}>
                {(() => {
                  const csMatch = clusterSummary.find(cs => String(cs.cluster_id) === sel.id);
                  return csMatch?.sample_template || sel.sample_message || "—";
                })()}
              </div>
            </>
          ) : (
            <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.md, textAlign: "center", paddingTop: 80 }}>← Click a cluster node to inspect</div>
          )}
        </div>
      </div>

      {/* ── Suspicious Splits ── */}
      {results?.suspicious_splits?.length > 0 && (
        <div style={{ marginTop: 20, background: "var(--bg-card)", border: "1px solid var(--surface-warn-border)", borderRadius: 8, padding: "16px 20px" }}>
          <div style={{ fontSize: FONT_SIZE.md, color: "var(--accent-yellow)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 4 }}>
            ⚠ Suspicious Splits — {results.suspicious_splits.length}
          </div>
          <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-muted)", marginBottom: 12, opacity: 0.8 }}>
            Cluster pairs that are unusually close — may be over-split templates
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: FONT_SIZE.base, fontFamily: "monospace" }}>
              <thead>
                <tr style={{ background: "var(--bg-card-deep)", borderBottom: "1px solid var(--border-default)" }}>
                  {["Cluster A", "Cluster B", "Distance", "Sample Template A", "Sample Template B"].map(h => (
                    <th key={h} style={{ padding: "7px 12px", color: "var(--text-muted)", fontWeight: 400, textAlign: "left", fontSize: FONT_SIZE.xs, letterSpacing: "0.05em", whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.suspicious_splits.map((split, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "7px 12px", color: "var(--accent-blue)", whiteSpace: "nowrap" }}>{split.cluster_a ?? "—"}</td>
                    <td style={{ padding: "7px 12px", color: "var(--accent-blue)", whiteSpace: "nowrap" }}>{split.cluster_b ?? "—"}</td>
                    <td style={{ padding: "7px 12px", color: "var(--accent-yellow)", whiteSpace: "nowrap" }}>{split.distance != null ? Number(split.distance).toFixed(4) : "—"}</td>
                    <td style={{ padding: "7px 12px", color: "var(--text-secondary)", maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={split.sample_template_a}>{split.sample_template_a || "—"}</td>
                    <td style={{ padding: "7px 12px", color: "var(--text-secondary)", maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={split.sample_template_b}>{split.sample_template_b || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}