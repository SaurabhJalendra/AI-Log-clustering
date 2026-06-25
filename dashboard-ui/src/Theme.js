// ─── GLOBAL THEME & SHARED CONSTANTS ─────────────────────────────────────────
// All font sizes, colours, and opacity values used across the dashboard.
// Import from here — never hardcode these values in page files.

// ─── TYPOGRAPHY ───────────────────────────────────────────────────────────────
export const FONT = {
  mono: "'JetBrains Mono', 'Fira Code', monospace",
};

export const FONT_SIZE = {
  xs:    9,
  sm:   10,
  base: 11,
  md:   12,
  lg:   13,
  xl:   14,
  h3:   15,
  h2:   18,
  h1:   20,
  kpi:  26,
  hero: 28,
};

// ─── OPACITY ──────────────────────────────────────────────────────────────────
export const OPACITY = {
  dim:      0.4,
  muted:    0.5,
  soft:     0.6,
  label:    0.65,
  standard: 0.8,
  full:     1,
};

// ─── SIGNAL COLOURS ───────────────────────────────────────────────────────────
// Foreground / stroke colours keyed by severity or signal tier
export const SIGNAL_COLOR = {
  none:     "#3a7d44",
  low:      "#b58900",
  medium:   "#cb4b16",
  high:     "#dc322f",
  CRITICAL: "#dc322f",
  HIGH:     "#cb4b16",
  MEDIUM:   "#b58900",
  LOW:      "#6c71c4",
  INFO:     "#268bd2",
  ROUTINE:  "#3a7d44",
};

// Background colours — injected as CSS variables per theme in index.css
export const SIGNAL_BG = {
  none:     "var(--signal-bg-none)",
  low:      "var(--signal-bg-low)",
  medium:   "var(--signal-bg-medium)",
  high:     "var(--signal-bg-high)",
  CRITICAL: "var(--signal-bg-critical)",
  HIGH:     "var(--signal-bg-high-sev)",
  MEDIUM:   "var(--signal-bg-medium-sev)",
  LOW:      "var(--signal-bg-low-sev)",
  INFO:     "var(--signal-bg-info)",
};

export const SEV_DOT = {
  CRITICAL: "#dc322f",
  HIGH:     "#cb4b16",
  MEDIUM:   "#b58900",
  LOW:      "#6c71c4",
  INFO:     "#268bd2",
  none:     "#3a7d44",
};

// ─── READABILITY STYLE (light + dark mode text overrides) ─────────────────────
export const READABILITY_STYLE = `
  /* ── Light mode: dark text on light background ── */
  :root,
  body:not(.dark) {
    --text-primary:    #0f1117 !important;
    --text-secondary:  #1e2130 !important;
    --text-muted:      #3a3f55 !important;
    --text-heading:    #0a0d14 !important;
    --sidebar-text:         #1e2130 !important;
    --sidebar-text-dim:     #3a3f55 !important;
    --sidebar-active-text:  #0a0d14 !important;
    --chart-grid:      rgba(0,0,0,0.08) !important;
  }
  /* ── Dark mode: bright text on dark background ── */
  body.dark {
    --text-primary:    #e8eaf0 !important;
    --text-secondary:  #c8ccd8 !important;
    --text-muted:      #9aa0b8 !important;
    --text-heading:    #ffffff !important;
    --sidebar-text:         #c8ccd8 !important;
    --sidebar-text-dim:     #9aa0b8 !important;
    --sidebar-active-text:  #ffffff !important;
    --chart-grid:      rgba(255,255,255,0.06) !important;
  }
`;