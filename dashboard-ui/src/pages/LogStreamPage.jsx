import { useState, useMemo } from "react";

// ─── Constants ────────────────────────────────────────────────────────────────

const SIGNAL_COLOR = {
  CRITICAL: "#dc322f", HIGH: "#cb4b16", MEDIUM: "#b58900",
  LOW: "#6c71c4", INFO: "#268bd2", ROUTINE: "#3a7d44", none: "#3a7d44",
};

const SIGNAL_BG = {
  CRITICAL: "var(--signal-bg-critical)",
  HIGH:     "var(--signal-bg-high-sev)",
  MEDIUM:   "var(--signal-bg-medium-sev)",
  LOW:      "var(--signal-bg-low-sev)",
  INFO:     "var(--signal-bg-info)",
};

// Domain label colours — intentionally distinct from severity palette.
// Red (#dc322f), Orange (#cb4b16), Yellow (#b58900) are RESERVED for
// CRITICAL / HIGH / MEDIUM severity signals and must never appear here.
const DOMAIN_COLOR = {
  hardware:        "#2aa198", // teal
  connectivity:    "#268bd2", // blue
  telemetry:       "#6c71c4", // purple
  api:             "#29a8a8", // cyan-teal
  scheduler:       "#4fa3d1", // sky blue
  messaging:       "#8f6fd6", // violet
  infrastructure:  "#5babcb", // steel blue
  other:           "#839496", // muted grey-blue
};

// anomaly_signal from stage3 → display severity label
// "high" → CRITICAL, "low" → LOW, "none" + not routine → MEDIUM, routine → ROUTINE
function signalToSev(anomaly_signal, is_routine) {
  if (anomaly_signal === "high")   return "CRITICAL";
  if (anomaly_signal === "medium") return "HIGH";
  if (anomaly_signal === "low")    return "LOW";
  if (is_routine === true || is_routine === "True" || is_routine === 1) return "ROUTINE";
  return "MEDIUM"; // anomaly_signal=none but not routine
}

// anomaly_signal → rough score for ScoreBar
function signalToScore(anomaly_signal) {
  if (anomaly_signal === "high")   return 0.9;
  if (anomaly_signal === "medium") return 0.65;
  if (anomaly_signal === "low")    return 0.35;
  return 0;
}

// Parse Python-style list string "['A', 'B']" → ['A', 'B']
function parseList(str) {
  if (!str || str === "[]") return [];
  try {
    return JSON.parse(str.replace(/'/g, '"'));
  } catch {
    return str.replace(/[\[\]']/g, "").split(",").map(s => s.trim()).filter(Boolean);
  }
}

const SEV_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, ROUTINE: 4 };

const GROUP_OPTIONS = [
  { value: "cluster",  label: "By Cluster"  },
  { value: "domain",   label: "By Domain"   },
  { value: "service",  label: "By Service"  },
  { value: "severity", label: "By Severity" },
  { value: "none",     label: "No Grouping" },
];
const SORT_OPTIONS = [
  { value: "count_desc",  label: "Log Count ↓"     },
  { value: "count_asc",   label: "Log Count ↑"     },
  { value: "score_desc",  label: "Anomaly Score ↓" },
  { value: "score_asc",   label: "Anomaly Score ↑" },
  { value: "alpha",       label: "Name A → Z"      },
];
const SEV_PILLS = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW", "ROUTINE"];

// ─── Sub-components ───────────────────────────────────────────────────────────

function SevBadge({ sev }) {
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: "2px 6px", borderRadius: 3,
      letterSpacing: ".05em", fontFamily: "monospace",
      color:      SIGNAL_COLOR[sev] || "var(--text-muted)",
      background: SIGNAL_BG[sev]    || "var(--bg-card-deep)",
      border:     `1px solid ${(SIGNAL_COLOR[sev] || "var(--border-default)") + "44"}`,
    }}>
      {sev}
    </span>
  );
}

function ScoreBar({ score }) {
  const pct   = Math.round((score || 0) * 100);
  const color = score > 0.75 ? "#dc322f" : score > 0.5 ? "#cb4b16" : score > 0.3 ? "#b58900" : "#6c71c4";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ flex: 1, background: "var(--border-default)", borderRadius: 2, height: 4 }}>
        <div style={{ width: `${pct}%`, background: color, height: 4, borderRadius: 2, transition: "width .3s" }} />
      </div>
      <span style={{ fontSize: 10, fontFamily: "monospace", color, minWidth: 22, textAlign: "right" }}>{pct}</span>
    </div>
  );
}

function TrendCell({ trend }) {
  const cfg = trend === "rising"  ? { icon: "↑", label: "rising",  color: "#dc322f" }
            : trend === "falling" ? { icon: "↓", label: "falling", color: "#3a7d44" }
            :                       { icon: "→", label: "stable",  color: "var(--text-muted)" };
  return (
    <span style={{ fontSize: 11, color: cfg.color, whiteSpace: "nowrap" }}>
      {cfg.icon} {cfg.label}
    </span>
  );
}

// ─── Detail panel (expanded row) ─────────────────────────────────────────────

function DetailPanel({ cluster }) {
  const sevDist = cluster.severity_distribution || {};
  const sevLevels = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"];
  const sevColors = { DEBUG: "#6c71c4", INFO: "#268bd2", WARN: "#b58900", ERROR: "#dc322f", FATAL: "#dc322f" };
  const totalSev  = sevLevels.reduce((s, l) => s + (sevDist[l] || 0), 0);

  const fields = [
    { label: "Domain",
      value: cluster.domain,
      color: DOMAIN_COLOR[cluster.domain] || "var(--text-secondary)" },
    { label: "Log Count",
      value: `${(cluster.count || 0).toLocaleString()} lines` },
    { label: "Dominant Severity",
      value: cluster.dominant_severity || "—",
      color: SIGNAL_COLOR[cluster.dominant_severity] || "var(--text-secondary)" },
    { label: "Anomaly Signal",
      value: cluster.anomaly_signal || "none",
      color: cluster.anomaly_signal === "high" ? "#dc322f"
           : cluster.anomaly_signal === "low"  ? "#b58900"
           : "var(--text-muted)" },
    { label: "Burst Collapsed",
      value: cluster.burst_collapsed_count > 0 ? `⚡ ${cluster.burst_collapsed_count}` : "none",
      color: cluster.burst_collapsed_count > 0 ? "#dc322f" : "var(--text-muted)" },
    { label: "Singleton Class",
      value: cluster.singleton_class || "—",
      color: cluster.singleton_class === "true_anomaly"   ? "#dc322f"
           : cluster.singleton_class === "unseen_variant" ? "#b58900"
           : "var(--text-muted)" },
  ];

  return (
    <div style={{
      background: "var(--bg-card-deep)",
      borderBottom: "1px solid var(--border-subtle)",
      padding: "12px 14px 12px 42px",
    }}>
      {/* Field grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 12 }}>
        {fields.map(f => (
          <div key={f.label}>
            <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 3 }}>{f.label}</div>
            <div style={{ fontSize: 11, fontFamily: "monospace", color: f.color || "var(--text-secondary)" }}>{f.value}</div>
          </div>
        ))}
      </div>

      {/* Severity distribution bar */}
      {totalSev > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 5 }}>Severity Distribution</div>
          <div style={{ display: "flex", height: 6, borderRadius: 3, overflow: "hidden", gap: 1 }}>
            {sevLevels.map(l => {
              const pct = totalSev > 0 ? (sevDist[l] || 0) / totalSev * 100 : 0;
              return pct > 0
                ? <div key={l} style={{ width: `${pct}%`, background: sevColors[l] }} title={`${l}: ${sevDist[l]}`} />
                : null;
            })}
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 4, flexWrap: "wrap" }}>
            {sevLevels.filter(l => (sevDist[l] || 0) > 0).map(l => (
              <span key={l} style={{ fontSize: 9, fontFamily: "monospace", color: sevColors[l] }}>
                {l} {sevDist[l]}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Time range */}
      {(cluster.first_seen || cluster.last_seen) && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 4 }}>Time Range</div>
          <div style={{ fontSize: 10, fontFamily: "monospace", color: "var(--text-secondary)" }}>
            {cluster.first_seen ? String(cluster.first_seen).slice(0, 19).replace("T", " ") : "—"}
            <span style={{ color: "var(--text-muted)", margin: "0 8px" }}>→</span>
            {cluster.last_seen  ? String(cluster.last_seen).slice(0, 19).replace("T", " ")  : "—"}
            <span style={{ color: "var(--text-muted)", marginLeft: 4 }}>UTC</span>
          </div>
        </div>
      )}

      {/* Anomaly reason */}
      {cluster.anomaly_reason && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 4 }}>Why Anomalous</div>
          <div style={{ fontSize: 10, fontFamily: "monospace", color: "#b58900", background: "var(--surface-warn)", border: "1px solid var(--surface-warn-border)", borderRadius: 4, padding: "5px 8px" }}>
            {cluster.anomaly_reason}
          </div>
        </div>
      )}

      {/* Services */}
      {cluster.services?.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 4 }}>Services</div>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {cluster.services.map(s => (
              <span key={s} style={{ fontSize: 9, fontFamily: "monospace", background: "var(--surface-info)", border: "1px solid var(--surface-info-border)", borderRadius: 3, color: "var(--accent-blue)", padding: "1px 6px" }}>
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Full sample template */}
      <div>
        <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 5 }}>Sample Template</div>
        <div style={{
          color: "var(--text-primary)", fontSize: 11, lineHeight: 1.6,
          background: "var(--bg-app)", border: "1px solid var(--border-default)",
          borderRadius: 4, padding: "8px 10px", fontFamily: "monospace", wordBreak: "break-word",
        }}>
          {cluster.msg || "—"}
        </div>
      </div>
    </div>
  );
}

// ─── Table row ────────────────────────────────────────────────────────────────

const ROW_COLS = "28px 80px 140px 1fr 90px 80px 160px";

function TableHeader() {
  const cols = [
    { label: "" }, { label: "Severity" }, { label: "Service" },
    { label: "Sample Template" }, { label: "Score" },
    { label: "Signal" }, { label: "Cluster ID" },
  ];
  return (
    <div style={{
      display: "grid", gridTemplateColumns: ROW_COLS,
      background: "var(--bg-card-deep)", border: "1px solid var(--border-subtle)",
      borderRadius: "6px 6px 0 0",
    }}>
      {cols.map((c, i) => (
        <div key={i} style={{
          padding: "7px 10px", fontSize: 9, color: "var(--text-muted)",
          textTransform: "uppercase", letterSpacing: ".06em",
          borderRight: i < cols.length - 1 ? "1px solid var(--border-subtle)" : "none",
        }}>
          {c.label}
        </div>
      ))}
    </div>
  );
}

function LogRow({ cluster, isExpanded, onToggle }) {
  const flags = [
    cluster.burst_collapsed_count > 0 && "⚡",
    cluster.centroid_drifted && "⟳",
  ].filter(Boolean).join(" ");

  return (
    <>
      <div
        onClick={onToggle}
        style={{
          display: "grid", gridTemplateColumns: ROW_COLS,
          alignItems: "center", minHeight: 34,
          background: isExpanded ? "rgba(38,139,210,.04)" : "var(--bg-card)",
          borderBottom: "1px solid var(--border-subtle)",
          cursor: "pointer", transition: "background .1s",
        }}
        onMouseEnter={e => { if (!isExpanded) e.currentTarget.style.background = "rgba(255,255,255,.02)"; }}
        onMouseLeave={e => { if (!isExpanded) e.currentTarget.style.background = "var(--bg-card)"; }}
      >
        {/* expand chevron */}
        <div style={{ padding: "0 0 0 14px", display: "flex", alignItems: "center" }}>
          <span style={{
            fontSize: 9, color: isExpanded ? "var(--accent-blue, #268bd2)" : "var(--text-muted)",
            transition: "transform .15s", display: "inline-block",
            transform: isExpanded ? "rotate(90deg)" : "none",
          }}>▶</span>
        </div>

        {/* severity */}
        <div style={{ padding: "6px 10px" }}><SevBadge sev={cluster.sev} /></div>

        {/* service */}
        <div style={{ padding: "6px 10px", fontSize: 11, fontWeight: 600, color: "var(--text-primary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {cluster.service}
          {flags && <span style={{ fontSize: 9, marginLeft: 5, color: "#b58900" }}>{flags}</span>}
        </div>

        {/* sample template */}
        <div style={{ padding: "6px 10px", fontSize: 11, color: "var(--text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: "monospace" }}
          title={cluster.msg}>
          {cluster.msg}
        </div>

        {/* score bar */}
        <div style={{ padding: "6px 10px" }}><ScoreBar score={cluster.score} /></div>

        {/* anomaly signal */}
        <div style={{ padding: "6px 10px" }}>
          <span style={{
            fontSize: 10, fontFamily: "monospace",
            color: cluster.anomaly_signal === "high"   ? "#dc322f"
                 : cluster.anomaly_signal === "medium" ? "#cb4b16"
                 : cluster.anomaly_signal === "low"    ? "#b58900"
                 : "var(--text-muted)",
          }}>
            {cluster.anomaly_signal === "none" ? (cluster.is_routine ? "— routine" : "— watch") : `● ${cluster.anomaly_signal}`}
          </span>
        </div>

        {/* cluster ID */}
        <div style={{ padding: "6px 10px", fontSize: 10, color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: "monospace" }}
          title={cluster.id}>
          {cluster.id}
        </div>
      </div>
      {isExpanded && <DetailPanel cluster={cluster} />}
    </>
  );
}

// ─── Group header ─────────────────────────────────────────────────────────────

function GroupHeader({ groupKey, items, isOpen, onToggle }) {
  const worstSev   = items.reduce((best, c) => (SEV_ORDER[c.sev] ?? 4) < (SEV_ORDER[best] ?? 4) ? c.sev : best, "ROUTINE");
  const totalLines = items.reduce((s, c) => s + (c.count || 0), 0);
  const dom        = items[0]?.domain;
  const domColor   = DOMAIN_COLOR[dom] || "#839496";

  return (
    <div
      onClick={onToggle}
      style={{
        display: "flex", alignItems: "center", gap: 10,
        padding: "8px 14px",
        background: "var(--bg-card-deep)", border: "1px solid var(--border-subtle)",
        borderRadius: 6, marginBottom: 2, marginTop: 8,
        cursor: "pointer", position: "sticky", top: 0, zIndex: 2,
      }}
      onMouseEnter={e => e.currentTarget.style.background = "var(--bg-card)"}
      onMouseLeave={e => e.currentTarget.style.background = "var(--bg-card-deep)"}
    >
      <span style={{ fontSize: 9, color: "var(--text-muted)", transition: "transform .2s", display: "inline-block", transform: isOpen ? "rotate(90deg)" : "none", flexShrink: 0 }}>▶</span>
      <span style={{
        fontSize: 9, padding: "2px 7px", borderRadius: 3, flexShrink: 0,
        background: domColor + "22", border: `1px solid ${domColor}44`, color: domColor,
        textTransform: "uppercase", letterSpacing: ".04em",
      }}>{dom || "—"}</span>
      <span style={{ fontSize: 11, color: "var(--text-secondary)", fontWeight: 600, flexShrink: 0 }}>{groupKey}</span>
      <span style={{ fontSize: 11, color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1, minWidth: 0 }}>
        {items[0]?.msg || ""}
      </span>
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
        <span style={{
          fontSize: 10, padding: "1px 8px", borderRadius: 10,
          background: "var(--bg-card)", border: "1px solid var(--border-default)",
          color: "var(--text-secondary)",
        }}>
          {totalLines.toLocaleString()} lines · {items.length} cluster{items.length > 1 ? "s" : ""}
        </span>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: SIGNAL_COLOR[worstSev] || "#839496", flexShrink: 0, display: "inline-block" }} />
      </div>
    </div>
  );
}

// ─── Main LogStreamPage ───────────────────────────────────────────────────────

export default function LogStreamPage({ results }) {

  // ── Merge cluster_summary (stage3) with anomalies (stage4) ─────────────────
  // cluster_summary is the source of truth for all clusters.
  // anomalies enrich with: anomaly_label, anomaly_score, trend_direction,
  //   burst_detected, recurrence_flag, top_source, error_pct.
  // Join key: cluster_summary.cluster_id === anomalies.event_id

  const allClusters = useMemo(() => {
    const clusters  = results?.cluster_summary || [];
    const anomalyMap = {};
    (results?.anomalies || []).forEach(a => {
      anomalyMap[String(a.event_id)] = a;
    });

    return clusters.map(c => {
      const a        = anomalyMap[String(c.cluster_id)] || {};
      const services = parseList(c.services);

      // Severity: prefer stage4 anomaly_label, fall back to stage3 signal derivation
      const sev = a.anomaly_label
        ? a.anomaly_label
        : signalToSev(c.anomaly_signal, c.is_routine);

      // Score: prefer stage4 anomaly_score, fall back to signal→score mapping
      const score = a.anomaly_score != null
        ? a.anomaly_score
        : signalToScore(c.anomaly_signal);

      // Parse severity_distribution (stored as Python dict string)
      let sevDist = {};
      try {
        const raw = c.severity_distribution || "{}";
        sevDist = JSON.parse(raw.replace(/'/g, '"'));
      } catch { sevDist = {}; }

      return {
        // identity
        id:                 c.cluster_id,
        label:              c.cluster_label || c.cluster_id,

        // display
        msg:                c.sample_template || a.sample_message || "—",
        service:            a.top_source || services[0] || "—",
        services,
        domain:             c.domain || a.domain || "other",

        // severity & signal
        sev,
        anomaly_signal:     c.anomaly_signal || "none",
        anomaly_reason:     c.anomaly_reason || a.anomaly_reason || "",
        dominant_severity:  c.dominant_severity || "INFO",
        severity_distribution: sevDist,
        is_routine:         c.is_routine === "True" || c.is_routine === true,

        // metrics
        score,
        count:              Number(c.total_log_count || a.count || 0),
        error_pct:          a.error_pct ?? 0,
        trend:              a.trend_direction || "stable",

        // flags
        burst_collapsed_count: Number(c.burst_collapsed_count || 0),
        burst_detected:     a.burst_detected || false,
        recurrence_flag:    a.recurrence_flag || false,
        centroid_drifted:   c.centroid_drifted === "True" || c.centroid_drifted === true,
        singleton_class:    c.singleton_class || a.singleton_class || "",

        // time
        first_seen:         c.first_seen || "",
        last_seen:          c.last_seen  || "",
      };
    });
  }, [results]);

  // ── UI state ───────────────────────────────────────────────────────────────
  const [search,    setSearch]    = useState("");
  const [groupBy,   setGroupBy]   = useState("cluster");
  const [sortBy,    setSortBy]    = useState("count_desc");
  const [viewMode,  setViewMode]  = useState("grouped");
  const [sevFilter, setSevFilter] = useState(new Set(["ALL"]));
  const [expanded,  setExpanded]  = useState(new Set());
  const [collapsed, setCollapsed] = useState(new Set());

  const toggleSev = (label) => {
    setSevFilter(prev => {
      const next = new Set(prev);
      if (label === "ALL") return new Set(["ALL"]);
      next.delete("ALL");
      if (next.has(label)) { next.delete(label); if (next.size === 0) next.add("ALL"); }
      else next.add(label);
      return next;
    });
  };
  const toggleRow   = (id)  => setExpanded(prev  => { const n = new Set(prev);  n.has(id)  ? n.delete(id)  : n.add(id);  return n; });
  const toggleGroup = (key) => setCollapsed(prev => { const n = new Set(prev);  n.has(key) ? n.delete(key) : n.add(key); return n; });

  // ── Filtered + sorted ──────────────────────────────────────────────────────
  const filtered = useMemo(() => {
    const q     = search.toLowerCase();
    const isAll = sevFilter.has("ALL");

    return allClusters
      .filter(c => {
        if (!isAll && !sevFilter.has(c.sev)) return false;
        if (!q) return true;
        return (
          (c.msg     || "").toLowerCase().includes(q) ||
          (c.service || "").toLowerCase().includes(q) ||
          (c.id      || "").toLowerCase().includes(q) ||
          (c.domain  || "").toLowerCase().includes(q) ||
          (c.label   || "").toLowerCase().includes(q)
        );
      })
      .sort((a, b) => {
        switch (sortBy) {
          case "count_desc":  return b.count - a.count;
          case "count_asc":   return a.count - b.count;
          case "score_desc":  return b.score - a.score;
          case "score_asc":   return a.score - b.score;
          case "alpha":       return (a.service || "").localeCompare(b.service || "");
          default:            return (SEV_ORDER[a.sev] ?? 4) - (SEV_ORDER[b.sev] ?? 4);
        }
      });
  }, [allClusters, search, sevFilter, sortBy]);

  // ── Counts for pills ───────────────────────────────────────────────────────
  const sevCounts = useMemo(() => {
    const counts = {};
    allClusters.forEach(c => { counts[c.sev] = (counts[c.sev] || 0) + 1; });
    return counts;
  }, [allClusters]);

  const totalLines = filtered.reduce((s, c) => s + (c.count || 0), 0);

  // ── Grouped data ───────────────────────────────────────────────────────────
  const grouped = useMemo(() => {
    if (groupBy === "none") return null;
    const map = {}, order = [];
    filtered.forEach(c => {
      const key = groupBy === "cluster"  ? c.id
                : groupBy === "domain"   ? c.domain
                : groupBy === "service"  ? c.service
                : c.sev;
      if (!map[key]) { map[key] = []; order.push(key); }
      map[key].push(c);
    });
    return { map, order };
  }, [filtered, groupBy]);

  const selectStyle = {
    appearance: "none", background: "var(--bg-card)",
    border: "1px solid var(--border-default)", borderRadius: 6,
    color: "var(--text-primary)", padding: "6px 28px 6px 10px",
    fontSize: 11, fontFamily: "monospace", cursor: "pointer", outline: "none",
  };

  const showHeader = viewMode === "flat" || groupBy === "cluster" || groupBy === "none";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, height: "100%" }}>

      {/* ── Toolbar ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", flexShrink: 0 }}>

        {/* Search */}
        <div style={{ position: "relative", flex: 1, minWidth: 200 }}>
          <span style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", fontSize: 12, pointerEvents: "none" }}>⌕</span>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search template, service, cluster ID, domain…"
            style={{
              width: "100%", background: "var(--bg-input)",
              border: "1px solid var(--border-default)", borderRadius: 6,
              color: "var(--text-primary)", padding: "7px 12px 7px 30px",
              fontSize: 11, fontFamily: "monospace", outline: "none",
            }}
          />
        </div>

        <div style={{ width: 1, height: 24, background: "var(--border-default)", flexShrink: 0 }} />

        {/* Group by */}
        <span style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em" }}>Group</span>
        <div style={{ position: "relative" }}>
          <select value={groupBy} onChange={e => setGroupBy(e.target.value)} style={selectStyle}>
            {GROUP_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          <span style={{ position: "absolute", right: 9, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", fontSize: 10, pointerEvents: "none" }}>▾</span>
        </div>

        {/* Sort by */}
        <span style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em" }}>Sort</span>
        <div style={{ position: "relative" }}>
          <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={selectStyle}>
            {SORT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          <span style={{ position: "absolute", right: 9, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted)", fontSize: 10, pointerEvents: "none" }}>▾</span>
        </div>

        <div style={{ width: 1, height: 24, background: "var(--border-default)", flexShrink: 0 }} />

        {/* View toggle */}
        <div style={{ display: "flex", border: "1px solid var(--border-default)", borderRadius: 6, overflow: "hidden" }}>
          {[{ id: "grouped", label: "≡ Grouped" }, { id: "flat", label: "⊟ Flat" }].map(v => (
            <button key={v.id} onClick={() => setViewMode(v.id)} style={{
              background: viewMode === v.id ? "rgba(38,139,210,.12)" : "var(--bg-card)",
              border: "none",
              borderRight: v.id === "grouped" ? "1px solid var(--border-default)" : "none",
              color: viewMode === v.id ? "var(--accent-blue, #268bd2)" : "var(--text-muted)",
              padding: "6px 12px", fontSize: 11, cursor: "pointer", fontFamily: "monospace",
            }}>
              {v.label}
            </button>
          ))}
        </div>

        <div style={{ width: 1, height: 24, background: "var(--border-default)", flexShrink: 0 }} />

        {/* Severity pills */}
        {SEV_PILLS.map(label => {
          const isAll    = label === "ALL";
          const isActive = isAll ? sevFilter.has("ALL") : sevFilter.has(label);
          const color    = SIGNAL_COLOR[label] || "var(--text-secondary)";
          const count    = sevCounts[label];
          return (
            <button key={label} onClick={() => toggleSev(label)} style={{
              display: "flex", alignItems: "center", gap: 5,
              background: isActive ? (isAll ? "var(--bg-card-deep)" : color + "15") : "var(--bg-card)",
              border: `1px solid ${isActive ? (isAll ? "var(--border-strong)" : color) : "var(--border-default)"}`,
              borderRadius: 20,
              color: isActive ? (isAll ? "var(--text-heading)" : color) : "var(--text-muted)",
              padding: "4px 12px", fontSize: 11, cursor: "pointer", fontFamily: "monospace",
              fontWeight: isActive ? 600 : 400, transition: "all .15s", whiteSpace: "nowrap",
            }}>
              {!isAll && <span style={{ width: 6, height: 6, borderRadius: "50%", background: isActive ? color : "var(--text-muted)", flexShrink: 0, display: "inline-block" }} />}
              {label}
              {!isAll && count > 0 && <span style={{ fontSize: 10, opacity: .7 }}>({count})</span>}
            </button>
          );
        })}

        {/* Result count */}
        <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: "auto", whiteSpace: "nowrap" }}>
          {filtered.length} clusters · {totalLines.toLocaleString()} lines
        </span>
      </div>

      {/* ── Table header ── */}
      {showHeader && <TableHeader />}

      {/* ── Log list ── */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {filtered.length === 0 && (
          <div style={{ textAlign: "center", padding: "48px 0", color: "var(--text-muted)", fontSize: 12 }}>
            No clusters match your filters
          </div>
        )}

        {/* Flat / no-grouping */}
        {(viewMode === "flat" || groupBy === "none") && filtered.length > 0 && (
          <div style={{ border: "1px solid var(--border-subtle)", borderTop: "none", borderRadius: "0 0 6px 6px", overflow: "hidden" }}>
            {filtered.map(c => (
              <LogRow key={c.id} cluster={c} isExpanded={expanded.has(c.id)} onToggle={() => toggleRow(c.id)} />
            ))}
          </div>
        )}

        {/* Grouped by cluster — one row per cluster, no section headers */}
        {viewMode === "grouped" && groupBy === "cluster" && grouped && grouped.order.map(key => {
          const c = grouped.map[key][0];
          return (
            <div key={key} style={{ marginBottom: 2 }}>
              <LogRow cluster={c} isExpanded={expanded.has(c.id)} onToggle={() => toggleRow(c.id)} />
            </div>
          );
        })}

        {/* Grouped by domain / service / severity — collapsible sections */}
        {viewMode === "grouped" && groupBy !== "cluster" && groupBy !== "none" && grouped && grouped.order.map(key => {
          const items  = grouped.map[key];
          const isOpen = !collapsed.has(key);
          return (
            <div key={key}>
              <GroupHeader groupKey={key} items={items} isOpen={isOpen} onToggle={() => toggleGroup(key)} />
              {isOpen && (
                <div style={{ border: "1px solid var(--border-subtle)", borderRadius: "0 0 6px 6px", overflow: "hidden", marginBottom: 2 }}>
                  {items.map(c => (
                    <LogRow key={c.id} cluster={c} isExpanded={expanded.has(c.id)} onToggle={() => toggleRow(c.id)} />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}