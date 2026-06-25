import { useState, useEffect, useMemo, useRef } from "react";
import { createPortal } from "react-dom";
import { SIGNAL_COLOR, FONT_SIZE, FONT } from "../Theme";
import { Badge } from "../Components";

// ─── INCIDENTS PAGE ───────────────────────────────────────────────────────────

// Tooltip rendered into document.body via a portal so it can never be clipped
// by an ancestor's overflow:hidden or viewport edge.
function ClusterChip({ id, clusterLookup }) {
  const [rect, setRect]     = useState(null);
  const chipRef             = useRef(null);
  const cs                  = clusterLookup[id] || {};
  const label               = cs.cluster_label || id;
  const template            = cs.sample_template || "";

  const handleMouseEnter = () => {
    if (chipRef.current) setRect(chipRef.current.getBoundingClientRect());
  };
  const handleMouseLeave = () => setRect(null);

  // Tooltip position: prefer above the chip, flip below if too close to top.
  // Clamp left so it never overflows right edge.
  const TOOLTIP_W = 340;
  const tooltipStyle = rect ? (() => {
    const spaceAbove = rect.top;
    const above      = spaceAbove > 160;
    const rawLeft    = rect.left;
    const clampedLeft = Math.min(rawLeft, window.innerWidth - TOOLTIP_W - 12);
    return {
      position:  "fixed",
      zIndex:    9999,
      width:     TOOLTIP_W,
      left:      clampedLeft,
      ...(above
        ? { bottom: window.innerHeight - rect.top + 6 }
        : { top:    rect.bottom + 6 }),
      background:    "var(--bg-card)",
      border:        "1px solid var(--border-default)",
      borderRadius:  6,
      padding:       "8px 10px",
      boxShadow:     "0 4px 20px rgba(0,0,0,.5)",
      pointerEvents: "none",
    };
  })() : null;

  return (
    <span style={{ position: "relative", display: "inline-block" }}
      ref={chipRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <span style={{
        display: "inline-block",
        fontFamily: "monospace", fontSize: FONT_SIZE.base,
        background: "var(--bg-card-deep)",
        border: "1px solid var(--border-default)",
        borderRadius: 4, padding: "1px 7px", cursor: "help",
        color: "var(--accent-blue, #268bd2)",
        textDecoration: "underline dotted",
      }}>
        {label}
      </span>

      {rect && createPortal(
        <div style={tooltipStyle}>
          <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 4 }}>
            Cluster ID
          </div>
          <div style={{ fontSize: FONT_SIZE.base, fontFamily: "monospace", color: "var(--text-secondary)", marginBottom: 6, wordBreak: "break-all" }}>
            {id}
          </div>
          {template && <>
            <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 4 }}>
              Log Template
            </div>
            <div style={{ fontSize: FONT_SIZE.base, fontFamily: "monospace", color: "var(--text-primary)", lineHeight: 1.5, wordBreak: "break-word" }}>
              {template}
            </div>
          </>}
        </div>,
        document.body
      )}
    </span>
  );
}

// Scans a narrative string for cluster ID tokens (SCxxxxxxxx pattern) and
// replaces each with a <ClusterChip>. Returns an array of React nodes.
function renderNarrativeWithChips(text, clusterLookup) {
  if (!text) return null;
  const parts = text.split(/(SC[0-9A-F]{12})/g);
  return parts.map((part, i) =>
    /^SC[0-9A-F]{12}$/.test(part)
      ? <ClusterChip key={i} id={part} clusterLookup={clusterLookup} />
      : <span key={i}>{part}</span>
  );
}

// The backend appends similar-incident context after a " | " separator.
// Split it out so both parts get their own readable section.
// The "similar past incident" line is clamped to 1 line by default and
// expands to full text when clicked, then collapses again on a second click.
function RecommendedAction({ text, clusterLookup }) {
  const [contextExpanded, setContextExpanded] = useState(false);
  if (!text) return null;
  const pipeIdx = text.indexOf(" | ");
  const primary = pipeIdx !== -1 ? text.slice(0, pipeIdx).trim()  : text.trim();
  const context = pipeIdx !== -1 ? text.slice(pipeIdx + 3).trim() : "";

  return (
    <div style={{ background: "var(--surface-info)", border: "1px solid var(--accent-green)33", borderRadius: 6, padding: "12px 14px", marginBottom: 14 }}>
      <div style={{ fontSize: FONT_SIZE.xs, color: "var(--accent-green)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>
        Recommended Action
      </div>
      <div style={{ fontSize: FONT_SIZE.lg, color: "var(--text-primary)", lineHeight: 1.7, wordBreak: "break-word" }}>
        {renderNarrativeWithChips(primary, clusterLookup)}
      </div>
      {context && (
        <div style={{
          marginTop: 10, paddingTop: 10,
          borderTop: "1px solid var(--border-subtle)",
        }}>
          {/* Label + expand toggle in one clickable row */}
          <div
            onClick={() => setContextExpanded(x => !x)}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              cursor: "pointer", userSelect: "none", marginBottom: contextExpanded ? 6 : 0,
            }}
          >
            <span style={{
              color: "var(--text-muted)", textTransform: "uppercase",
              letterSpacing: ".05em", fontSize: FONT_SIZE.xs,
            }}>
              Similar past incident ›
            </span>
            <span style={{ fontSize: FONT_SIZE.xs, color: "var(--accent-blue)", fontFamily: "monospace", marginLeft: "auto" }}>
              {contextExpanded ? "▲ collapse" : "▼ expand"}
            </span>
          </div>

          {/* Truncated preview — always visible, single line */}
          {!contextExpanded && (
            <div
              onClick={() => setContextExpanded(true)}
              style={{
                fontSize: FONT_SIZE.base, color: "var(--text-muted)",
                fontFamily: FONT.mono, lineHeight: 1.6,
                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                cursor: "pointer",
              }}
              title="Click to expand"
            >
              {/* Strip chip tokens for the preview so it reads as plain text */}
              {context.replace(/SC[0-9A-F]{12}/g, id => id)}
            </div>
          )}

          {/* Full expanded view */}
          {contextExpanded && (
            <div style={{
              fontSize: FONT_SIZE.base, color: "var(--text-muted)",
              fontFamily: FONT.mono, lineHeight: 1.6, wordBreak: "break-word",
            }}>
              {renderNarrativeWithChips(context, clusterLookup)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function IncidentsPage({ results }) {
  const incidents = results?.incidents || [];
  const [selected, setSelected] = useState(incidents[0]?.incident_id || null);

  // Build cluster_id → { cluster_label, sample_template } lookup
  const clusterLookup = useMemo(() => {
    const map = {};
    (results?.cluster_summary || []).forEach(cs => {
      map[String(cs.cluster_id)] = {
        cluster_label:   cs.cluster_label   || "",
        sample_template: cs.sample_template || "",
      };
    });
    return map;
  }, [results]);

  useEffect(() => {
    if (incidents.length > 0 && !selected) setSelected(incidents[0].incident_id);
  }, [incidents]);

  const sel = incidents.find(i => i.incident_id === selected);

  if (incidents.length === 0) {
    return (
      <div style={{ color: "var(--text-muted)", fontSize: FONT_SIZE.lg, textAlign: "center", marginTop: 60 }}>
        No incidents detected in this log file
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 16 }}>
      {/* ── Incident list ── */}
      <div>
        {incidents.map(inc => (
          <div key={inc.incident_id} onClick={() => setSelected(inc.incident_id)} style={{
            background: selected === inc.incident_id ? "var(--bg-card)" : "var(--bg-card-deep)",
            border: `1px solid ${selected === inc.incident_id ? SIGNAL_COLOR[inc.incident_severity] + "66" : "var(--border-default)"}`,
            borderLeft: `3px solid ${SIGNAL_COLOR[inc.incident_severity]}`,
            borderRadius: 8, padding: "12px 14px", marginBottom: 8, cursor: "pointer",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <Badge label={inc.incident_severity} type={inc.incident_severity} />
              <span style={{ fontSize: FONT_SIZE.base, fontFamily: "monospace", color: "var(--text-muted)" }}>{inc.incident_id}</span>
            </div>
            <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-secondary)" }}>
              {String(inc.primary_domain || "—").toUpperCase()} · {inc.root_cause_service || "unknown"}
            </div>
            <div style={{ fontSize: FONT_SIZE.base, color: "var(--text-muted)", marginTop: 3 }}>
              {inc.n_clusters} cluster(s)
              {inc.recurrence_flag && <span style={{ color: "#b58900", marginLeft: 8 }}>↻ recurring</span>}
            </div>
          </div>
        ))}
      </div>

      {/* ── Incident detail ── */}
      {sel && (
        <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "20px 24px", overflowY: "auto", maxHeight: "80vh" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
            <Badge label={sel.incident_severity} type={sel.incident_severity} />
            <span style={{ fontFamily: "monospace", fontSize: FONT_SIZE.xl, color: "var(--text-heading)" }}>{sel.incident_id}</span>
            <span style={{ fontSize: FONT_SIZE.base, color: "var(--text-muted)", marginLeft: "auto" }}>
              {sel.incident_end ? "RESOLVED" : "● ACTIVE"}
            </span>
          </div>

          {sel.what_happened && (
            <div style={{ background: "var(--surface-error)", border: "1px solid var(--surface-error-border)", borderRadius: 6, padding: "12px 14px", marginBottom: 14 }}>
              <div style={{ fontSize: FONT_SIZE.xs, color: "var(--accent-red)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>What Happened</div>
              <div style={{ fontSize: FONT_SIZE.lg, color: "var(--text-primary)", lineHeight: 1.5 }}>{renderNarrativeWithChips(sel.what_happened, clusterLookup)}</div>
            </div>
          )}

          <RecommendedAction text={sel.recommended_action} clusterLookup={clusterLookup} />

          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 16 }}>
            {[
              { label: "Domain",   v: sel.primary_domain || "—" },
              { label: "Root Cause", v: sel.root_cause_service || "—" },
              { label: "Duration", v: sel.duration_minutes ? `${sel.duration_minutes} min` : "—" },
              { label: "Start",    v: sel.incident_start ? String(sel.incident_start).slice(11,16) + " UTC" : "—" },
              { label: "End",      v: sel.incident_end   ? String(sel.incident_end).slice(11,16) + " UTC" : "Ongoing" },
              { label: "Clusters", v: sel.n_clusters || "—" },
            ].map(({ label, v }) => (
              <div key={label} style={{ background: "var(--bg-card-deep)", borderRadius: 6, padding: "8px 10px" }}>
                <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 2, textTransform: "uppercase" }}>{label}</div>
                <div style={{ fontSize: FONT_SIZE.md, color: "var(--text-secondary)", fontFamily: "monospace" }}>{v}</div>
              </div>
            ))}
          </div>

          {(() => {
            const raw  = sel.services_affected;
            const svcs = Array.isArray(raw) ? raw
              : typeof raw === "string" && raw.trim()
                ? raw.split(/[|,]/).map(s => s.trim()).filter(Boolean)
                : [];
            return svcs.length > 0 ? (
              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em" }}>Services Affected</div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {svcs.map(svc => (
                    <span key={svc} style={{ background: "var(--surface-info)", border: "1px solid var(--surface-info-border)", borderRadius: 4, color: "var(--accent-blue)", padding: "2px 10px", fontSize: FONT_SIZE.base, fontFamily: "monospace" }}>
                      {svc}
                    </span>
                  ))}
                </div>
              </div>
            ) : null;
          })()}

          {sel.cascade_chain && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: FONT_SIZE.xs, color: "var(--text-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em" }}>Cascade Chain</div>
              <div style={{ fontSize: FONT_SIZE.md, fontFamily: "monospace", color: "#b58900", background: "var(--bg-card-deep)", padding: "8px 12px", borderRadius: 6, lineHeight: 1.8 }}>
                {renderNarrativeWithChips(sel.cascade_chain, clusterLookup)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}