// ─── SHARED UI COMPONENTS ─────────────────────────────────────────────────────
// Small reusable components imported by page files and dashboard.jsx

import { SIGNAL_BG, SIGNAL_COLOR, FONT_SIZE, FONT } from "./Theme";

// ── Badge ──────────────────────────────────────────────────────────────────────
export function Badge({ label, type }) {
  return (
    <span style={{
      background: SIGNAL_BG[type] || "var(--bg-card-deep)",
      color: SIGNAL_COLOR[type] || "var(--text-muted)",
      border: `1px solid ${SIGNAL_COLOR[type] || "var(--border-default)"}33`,
      padding: "2px 8px", borderRadius: 4,
      fontSize: FONT_SIZE.base,
      fontFamily: "monospace", fontWeight: 600, letterSpacing: "0.05em",
    }}>
      {label}
    </span>
  );
}

// ── ScoreBar ───────────────────────────────────────────────────────────────────
export function ScoreBar({ score }) {
  const pct   = Math.round((score || 0) * 100);
  const color = score > 0.75 ? "#dc322f"
              : score > 0.5  ? "#cb4b16"
              : score > 0.3  ? "#b58900"
              :                "#6c71c4";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, background: "var(--border-default)", borderRadius: 3, height: 6 }}>
        <div style={{ width: `${pct}%`, background: color, height: 6, borderRadius: 3, transition: "width 0.4s" }} />
      </div>
      <span style={{ fontSize: FONT_SIZE.base, fontFamily: "monospace", color, minWidth: 30 }}>{pct}</span>
    </div>
  );
}

// ── ClickableCard ──────────────────────────────────────────────────────────────
export function ClickableCard({ onClick, children, style, className }) {
  return (
    <div
      onClick={onClick}
      className={`clickable-card${className ? " " + className : ""}`}
      style={style}
    >
      {children}
    </div>
  );
}

// ── KpiCard ────────────────────────────────────────────────────────────────────
// Generic stat tile used across multiple pages
export function KpiCard({ label, value, color, sub }) {
  return (
    <div style={{ background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 8, padding: "16px 20px" }}>
      <div style={{ fontSize: FONT_SIZE.sm, color: "var(--text-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
      <div style={{ fontSize: FONT_SIZE.kpi, fontWeight: 700, color, fontFamily: FONT.mono }}>{value}</div>
      {sub && <div style={{ fontSize: FONT_SIZE.sm, color: "var(--text-muted)", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}