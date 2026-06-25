import { useState } from "react";
import DomainDistribution from "../charts/DomainDistribution";
import TemplateLengthDist from "../charts/TemplateLengthDist";
import { FONT_SIZE, FONT } from "../Theme";
import { Badge } from "../Components";

// ─── TEMPLATE INTELLIGENCE PAGE ───────────────────────────────────────────────

export default function TemplateIntelPage({ results }) {
  const ri             = results?.run_info      || {};
  const anomalies      = results?.anomalies     || [];
  const clusterSummary = results?.cluster_summary || [];
  const [search, setSearch] = useState("");
  const [showConfDist, setShowConfDist] = useState(false);

  // ── Domain confidence histogram buckets (10 × 0.1 buckets, 0.0 → 1.0) ───────
  const confBuckets = Array(10).fill(0);
  clusterSummary.forEach(c => {
    const v = parseFloat(c.domain_confidence);
    if (!isNaN(v)) {
      const idx = Math.min(Math.floor(v * 10), 9);
      confBuckets[idx]++;
    }
  });
  const confBucketMax = Math.max(...confBuckets, 1);

  const totalTemplates  = ri.unique_templates || clusterSummary.length || 0;
  const activeDomains   = new Set(clusterSummary.map(c => c.domain).filter(Boolean)).size;
  const mergedTemplates = ri.merged_template_count
    ?? anomalies.filter(a => a.is_merged === true).length;
  const trueAnomalies   = anomalies.filter(a => a.singleton_class === "true_anomaly");
  const unseenVariants  = anomalies.filter(a => a.singleton_class === "unseen_variant");
  const mergedList      = anomalies.filter(a => a.is_merged === true);

  // ── Template Quality metrics ─────────────────────────────────────────────────
  // calibrated_drain_similarity: pipeline-level value emitted by the backend.
  // If absent, derive a proxy: compression ratio = 1 − (unique_templates / total_lines).
  // A high ratio (close to 1) means DRAIN collapsed logs heavily → strong similarity.
  const drainSimilarity = (() => {
    if (ri.calibrated_drain_similarity != null)
      return Number(ri.calibrated_drain_similarity).toFixed(3);
    const totalLines = clusterSummary.reduce((s, c) => s + Number(c.total_log_count || 0), 0);
    if (totalLines > 0 && clusterSummary.length > 0)
      return (1 - clusterSummary.length / totalLines).toFixed(3);
    return "—";
  })();

  // cluster_threshold: pipeline-level value. If absent, use mean domain_confidence
  // across clusters as a proxy (reflects how cleanly clusters were separated).
  const clusterThreshold = (() => {
    if (ri.cluster_threshold != null)
      return Number(ri.cluster_threshold).toFixed(3);
    const confs = clusterSummary.map(c => Number(c.domain_confidence)).filter(v => !isNaN(v) && v > 0);
    if (confs.length > 0)
      return (confs.reduce((s, v) => s + v, 0) / confs.length).toFixed(3);
    return "—";
  })();

  const csLookup = {};
  clusterSummary.forEach(cs => { csLookup[String(cs.cluster_id)] = cs; });

  const filteredAnomalies = anomalies.filter(row => {
    if (!search) return true;
    const q  = search.toLowerCase();
    const cs = csLookup[String(row.event_id)] || {};
    return (
      String(row.event_id   || "").toLowerCase().includes(q) ||
      String(row.domain || cs.domain || "").toLowerCase().includes(q) ||
      String(row.singleton_class || "").toLowerCase().includes(q) ||
      String(row.sample_template || cs.sample_template || row.sample_message || "").toLowerCase().includes(q)
    );
  });

  return (
    <div>
      {/* ── Scorecard row ── */}
      <div className="scorecard-row" style={{ gridTemplateColumns: "repeat(5, 1fr)", marginBottom: 20 }}>
        {[
          { label: "Total Templates",           value: totalTemplates.toLocaleString() },
          { label: "Active Domains",            value: activeDomains },
          { label: "Merged Templates",          value: mergedTemplates },
          { label: "True Anomaly Singletons",   value: trueAnomalies.length },
          { label: "Unseen Variant Singletons", value: unseenVariants.length },
        ].map(({ label, value }) => (
          <div key={label} style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "12px 14px" }}>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
            <div style={{ fontSize: FONT_SIZE.h1, fontWeight: 700, color: "var(--text-primary)", fontFamily: FONT.mono }}>{value}</div>
          </div>
        ))}
      </div>

      {/* ── Domain Distribution + Template Quality ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <DomainDistribution clusterSummary={clusterSummary} anomalies={anomalies} />

        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px" }}>
          <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 12 }}>
            Template Quality
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
            {[
              { label: "Drain Similarity",  value: drainSimilarity },
              { label: "Cluster Threshold", value: clusterThreshold },
            ].map(({ label, value }) => (
              <div key={label} style={{ background: "var(--bg-card-deep)", border: "1px solid var(--border-subtle)", borderRadius: 6, padding: "10px 12px", textAlign: "center" }}>
                <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: "var(--accent-blue)", fontFamily: FONT.mono }}>{value}</div>
              </div>
            ))}
          </div>

          {/* ── Confidence distribution — collapsible ── */}
          <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: 10, marginBottom: 16 }}>
            <button
              onClick={() => setShowConfDist(v => !v)}
              style={{
                display: "flex", alignItems: "center", gap: 6, width: "100%",
                background: "none", border: "none", cursor: "pointer", padding: 0,
              }}
            >
              <span style={{
                fontSize: FONT_SIZE.xs, display: "inline-block",
                transform: showConfDist ? "rotate(90deg)" : "none",
                transition: "transform .15s",
                color: "var(--text-muted)",
              }}>▶</span>
              <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Confidence distribution
              </span>
              <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginLeft: "auto", fontFamily: FONT.mono }}>
                {clusterSummary.length} clusters
              </span>
            </button>

            {showConfDist && (
              <div style={{ marginTop: 12 }}>
                {/* Bar chart — pure CSS, no library needed */}
                <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 80 }}>
                  {confBuckets.map((count, i) => {
                    const heightPct = confBucketMax > 0 ? (count / confBucketMax) * 100 : 0;
                    const bucketLabel = (i / 10).toFixed(1);
                    const isNearMean  = i === Math.floor(parseFloat(clusterThreshold) * 10);
                    return (
                      <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", height: "100%", justifyContent: "flex-end" }}
                        title={`${bucketLabel}–${((i + 1) / 10).toFixed(1)}: ${count} cluster${count !== 1 ? "s" : ""}`}>
                        <div style={{
                          width: "100%",
                          height: `${heightPct}%`,
                          minHeight: count > 0 ? 2 : 0,
                          background: isNearMean ? "var(--accent-red, #dc322f)" : "var(--accent-blue, #268bd2)",
                          borderRadius: "2px 2px 0 0",
                          opacity: 0.85,
                          transition: "height .2s",
                        }} />
                      </div>
                    );
                  })}
                </div>

                {/* X-axis labels */}
                <div style={{ display: "flex", gap: 3, marginTop: 4 }}>
                  {confBuckets.map((_, i) => (
                    <div key={i} style={{ flex: 1, textAlign: "center", fontSize: FONT_SIZE.xs, color: "var(--text-muted)", fontFamily: FONT.mono }}>
                      {(i / 10).toFixed(1)}
                    </div>
                  ))}
                </div>

                {/* Legend */}
                <div style={{ display: "flex", gap: 14, marginTop: 8, flexWrap: "wrap" }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: FONT_SIZE.xs, color: "var(--text-muted)" }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: "var(--accent-blue, #268bd2)", display: "inline-block" }} />
                    clusters
                  </span>
                  <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: FONT_SIZE.xs, color: "var(--text-muted)" }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: "var(--accent-red, #dc322f)", display: "inline-block" }} />
                    mean bucket ({clusterThreshold})
                  </span>
                  <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginLeft: "auto", fontStyle: "italic" }}>
                    hover bar for count
                  </span>
                </div>
              </div>
            )}
          </div>

          <TemplateLengthDist templateLengthDist={ri.template_length_distribution} anomalies={anomalies} />
        </div>
      </div>

      {/* ── Template Explorer table ── */}
      <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px", marginBottom: 16 }}>
        <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 10 }}>
          Template Explorer
        </div>
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Filter by template text, domain, cluster, or singleton class..."
          style={{
            width: "100%", boxSizing: "border-box",
            background: "var(--bg-input)", border: "1px solid var(--border-default)",
            borderRadius: 6, color: "var(--text-primary)", padding: "7px 12px",
            fontSize: FONT_SIZE.md, fontFamily: "monospace", marginBottom: 12,
          }}
        />
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: FONT_SIZE.base, fontFamily: "monospace" }}>
            <thead>
              <tr style={{ background: "var(--bg-card-deep)", borderBottom: "1px solid var(--border-default)" }}>
                {["Template ID", "Domain", "Cluster", "Severity", "Log Count", "Singleton Class", "Event Template"].map(h => (
                  <th key={h} style={{ padding: "7px 12px", color: "var(--text-muted)", fontWeight: 400, textAlign: "left", fontSize: FONT_SIZE.xs, letterSpacing: "0.05em", whiteSpace: "nowrap" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredAnomalies.map((row, i) => {
                const cs            = csLookup[String(row.event_id)] || {};
                const domain        = row.domain || cs.domain || "—";
                const eventTemplate = row.sample_template || cs.sample_template || row.sample_message || "—";
                return (
                  <tr key={row.event_id || i} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "6px 12px", color: "var(--accent-blue)", whiteSpace: "nowrap" }}>{row.event_id}</td>
                    <td style={{ padding: "6px 12px", color: "var(--text-secondary)", whiteSpace: "nowrap" }}>{domain}</td>
                    <td style={{ padding: "6px 12px", color: "var(--text-muted)", whiteSpace: "nowrap" }}>{row.cluster_id ?? row.event_id}</td>
                    <td style={{ padding: "6px 12px", whiteSpace: "nowrap" }}>
                      <Badge label={row.dominant_severity || row.anomaly_label || "—"} type={row.dominant_severity || row.anomaly_label} />
                    </td>
                    <td style={{ padding: "6px 12px", color: "var(--text-secondary)", fontFamily: "monospace" }}>{(row.count || 0).toLocaleString()}</td>
                    <td style={{ padding: "6px 12px", whiteSpace: "nowrap" }}>
                      {row.singleton_class
                        ? <span style={{
                            fontSize: FONT_SIZE.xs, fontFamily: "monospace", padding: "2px 7px", borderRadius: 4,
                            background: row.singleton_class === "true_anomaly" ? "var(--surface-error)" : row.singleton_class === "unseen_variant" ? "var(--surface-warn)" : "var(--bg-card-deep)",
                            color: row.singleton_class === "true_anomaly" ? "var(--accent-red)" : row.singleton_class === "unseen_variant" ? "var(--accent-yellow)" : "var(--text-muted)",
                            border: `1px solid ${row.singleton_class === "true_anomaly" ? "var(--surface-error-border)" : row.singleton_class === "unseen_variant" ? "var(--surface-warn-border)" : "var(--border-subtle)"}`,
                          }}>{row.singleton_class}</span>
                        : <span style={{ color: "var(--text-muted)" }}>—</span>
                      }
                    </td>
                    <td style={{ padding: "6px 12px", color: "var(--text-secondary)", maxWidth: 340, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={eventTemplate}>
                      {eventTemplate}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {filteredAnomalies.length === 0 && (
            <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.md, textAlign: "center", padding: "28px 0" }}>
              No templates match your filter
            </div>
          )}
        </div>
        <div style={{ marginTop: 8, fontSize: FONT_SIZE.base, color: "var(--text-muted)" }}>
          {filteredAnomalies.length} / {anomalies.length} templates
        </div>
      </div>

      {/* ── Singleton Spotlight ── */}
      {(trueAnomalies.length > 0 || unseenVariants.length > 0) && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
          <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px" }}>
            <div style={{ fontSize: FONT_SIZE.md, color: "var(--accent-red)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 10 }}>
              ● True Anomaly Singletons — {trueAnomalies.length}
            </div>
            {trueAnomalies.length === 0
              ? <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.md }}>None detected</div>
              : trueAnomalies.map(a => {
                const cs = csLookup[String(a.event_id)] || {};
                const templateText = a.sample_template || cs.sample_template || a.sample_message || cs.sample_message || "—";
                const humanLabel   = cs.cluster_label  || a.cluster_label   || "";
                return (
                  <div key={a.event_id} className="singleton-card anomaly">
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                      <Badge label={a.anomaly_label} type={a.anomaly_label} />
                      <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", fontFamily: "monospace" }}>{a.event_id}</span>
                      {humanLabel && (
                        <span style={{ fontSize: FONT_SIZE.xs, color: "var(--accent-blue, #268bd2)", fontFamily: "monospace",
                          background: "var(--surface-info)", border: "1px solid var(--surface-info-border)",
                          borderRadius: 3, padding: "1px 6px" }}>
                          {humanLabel}
                        </span>
                      )}
                      <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginLeft: "auto" }}>{a.domain || cs.domain}</span>
                    </div>
                    <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-secondary)", fontFamily: "monospace", lineHeight: 1.5, wordBreak: "break-word" }}>
                      {templateText}
                    </div>
                  </div>
                );
              })
            }
          </div>

          <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px" }}>
            <div style={{ fontSize: FONT_SIZE.md, color: "var(--accent-yellow)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 10 }}>
              ● Unseen Variant Singletons — {unseenVariants.length}
            </div>
            {unseenVariants.length === 0
              ? <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.md }}>None detected</div>
              : unseenVariants.map(a => {
                const cs = csLookup[String(a.event_id)] || {};
                const templateText = a.sample_template || cs.sample_template || a.sample_message || cs.sample_message || "—";
                const humanLabel   = cs.cluster_label  || a.cluster_label   || "";
                return (
                  <div key={a.event_id} className="singleton-card unseen">
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                      <Badge label={a.anomaly_label} type={a.anomaly_label} />
                      <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", fontFamily: "monospace" }}>{a.event_id}</span>
                      {humanLabel && (
                        <span style={{ fontSize: FONT_SIZE.xs, color: "#b58900", fontFamily: "monospace",
                          background: "var(--surface-warn)", border: "1px solid var(--surface-warn-border)",
                          borderRadius: 3, padding: "1px 6px" }}>
                          {humanLabel}
                        </span>
                      )}
                      <span style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginLeft: "auto" }}>{a.domain || cs.domain}</span>
                    </div>
                    <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-secondary)", fontFamily: "monospace", lineHeight: 1.5, wordBreak: "break-word" }}>
                      {templateText}
                    </div>
                  </div>
                );
              })
            }
          </div>
        </div>
      )}

      {/* ── Merged Templates ── */}
      {mergedList.length > 0 && (
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px" }}>
          <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-muted)", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 12 }}>
            Merged Templates — {mergedList.length}
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: FONT_SIZE.base, fontFamily: "monospace" }}>
              <thead>
                <tr style={{ background: "var(--bg-card-deep)", borderBottom: "1px solid var(--border-default)" }}>
                  {["Template ID", "Merged Into", "Domain", "Severity"].map(h => (
                    <th key={h} style={{ padding: "7px 12px", color: "var(--text-muted)", fontWeight: 400, textAlign: "left", fontSize: FONT_SIZE.xs, letterSpacing: "0.05em" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {mergedList.map((row, i) => (
                  <tr key={row.event_id || i} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                    <td style={{ padding: "6px 12px", color: "var(--accent-blue)" }}>{row.event_id}</td>
                    <td style={{ padding: "6px 12px", color: "var(--text-secondary)" }}>{row.merged_into || "—"}</td>
                    <td style={{ padding: "6px 12px", color: "var(--text-muted)" }}>{row.domain || "—"}</td>
                    <td style={{ padding: "6px 12px" }}>
                      <Badge label={row.dominant_severity || row.anomaly_label || "—"} type={row.dominant_severity || row.anomaly_label} />
                    </td>
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