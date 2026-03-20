# Universal Log Clustering Architecture

## Problem Statement

**Input:** Raw, heterogeneous production logs (any format, any source, any structure)
**Output:**
1. Group similar log patterns into Event Types
2. Detect anomalies (rare/unusual events)
3. Classify severity of each event
4. Provide actionable summary for ops teams

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 1: INGEST                              │
│                                                                  │
│  ANY log file (any format, any source, any structure)            │
│       ↓                                                          │
│  Line-level intake (preserve every line, skip only blanks)       │
│       ↓                                                          │
│  Format auto-detection per line:                                 │
│    ┌──────────────────────────────────────────────┐              │
│    │  Probe chain (each line tested independently) │              │
│    │                                               │              │
│    │  1. Try JSON parse          → structured      │              │
│    │  2. Try syslog regex        → structured      │              │
│    │  3. Try Apache/NGINX CLF    → structured      │              │
│    │  4. Try pipe-delimited      → structured      │              │
│    │  5. Try bracket format      → structured      │              │
│    │  6. Try key=value extraction → semi-structured │              │
│    │  7. Fallback: raw text      → unstructured    │              │
│    └──────────────────────────────────────────────┘              │
│       ↓                                                          │
│  Unified schema (best-effort extraction):                        │
│    timestamp | source | severity | message | format_type         │
│                                                                  │
│  Principle: NEVER drop a line. If it can't be parsed,            │
│  store raw text in "message" and mark format as "unknown"        │
│                                                                  │
│  Target: 100% retention, best-effort structure                   │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 2: NORMALIZE                            │
│                                                                  │
│  Operates ONLY on the "message" field (not raw log)              │
│       ↓                                                          │
│  Token-type replacement (order matters):                         │
│                                                                  │
│    1. Quoted strings     → <STRING>                              │
│    2. Email addresses    → <EMAIL>                               │
│    3. URLs               → <URL>                                 │
│    4. IP addresses       → <IP>                                  │
│    5. File paths         → <PATH>                                │
│    6. UUIDs / GUIDs      → <UUID>                                │
│    7. Hex values (0x...) → <HEX>                                 │
│    8. Key=value IDs      → key=<ID>                              │
│    9. Durations (5ms)    → <DURATION>                            │
│   10. Decimals (3.14)    → <NUM>                                 │
│   11. Integers           → <NUM>                                 │
│                                                                  │
│  Principle: Replace VARIABLE data, preserve STRUCTURAL meaning   │
│  "Payment failed order=ORD-123 amount=50.00"                     │
│  → "payment failed order=<ID> amount=<NUM>"                      │
│                                                                  │
│  Same logical event → same normalized form regardless of         │
│  which format it was logged in                                   │
│                                                                  │
│  Target: Maximum template collapse, minimum information loss     │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 3: EMBED & CLUSTER                     │
│                                                                  │
│  3a. Pre-embedding cleanup                                       │
│      Strip ALL placeholder tokens from normalized message        │
│      so the model only sees semantic words                       │
│      "payment failed order=<ID> amount=<NUM>"                    │
│      → "payment failed order= amount="                           │
│                                                                  │
│  3b. Embed unique templates only (not every log line)            │
│      Model: sentence-transformers (lightweight, fast)            │
│                                                                  │
│  3c. Cluster with validated threshold                            │
│      - Compute similarity matrix                                 │
│      - Auto-tune threshold using knee/elbow detection            │
│        on the similarity distribution                            │
│      - Merge via Union-Find or DBSCAN                            │
│      - Validate with silhouette score / DBCV                     │
│      - Log: "N events found, silhouette = X"                     │
│                                                                  │
│  3d. Assign Event IDs (E0001, E0002, ...)                        │
│                                                                  │
│  Principle: Cluster on MEANING, not on format or variable data   │
│  Target: Same logical event = same ID, regardless of log format  │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 4: ANALYZE                             │
│                                                                  │
│  Per Event ID, compute:                                          │
│                                                                  │
│  4a. Frequency                                                   │
│      - Total count                                               │
│      - Rarity score (is this common or unusual?)                 │
│                                                                  │
│  4b. Severity                                                    │
│      - From PARSED level field (Stage 1), not keyword guessing   │
│      - Aggregate: what % are ERROR vs WARN vs INFO?              │
│                                                                  │
│  4c. Temporal patterns (if timestamps available)                 │
│      - Distribution over time                                    │
│      - Burst detection (sudden spike = incident?)                │
│      - First/last occurrence                                     │
│                                                                  │
│  4d. Source distribution (if source field available)              │
│      - Which services/hosts produce this event?                  │
│      - Is it isolated to one host or widespread?                 │
│                                                                  │
│  4e. Anomaly scoring                                             │
│      - Composite: rarity + severity + burstiness                 │
│      - Rank all events from most to least anomalous              │
│      - Flag: CRITICAL / HIGH / MEDIUM / LOW                      │
│                                                                  │
│  Principle: Use STRUCTURED fields from Stage 1, not regex hacks  │
│  Target: Actionable insights per event type                      │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 5: VISUALIZE & EXPORT                  │
│                                                                  │
│  5a. Cluster map                                                 │
│      - 2D scatter (PCA or t-SNE) colored by Event ID            │
│      - Anomalies highlighted                                     │
│                                                                  │
│  5b. Frequency chart                                             │
│      - Bar chart: top N event types by count                     │
│      - Rare events table                                         │
│                                                                  │
│  5c. Timeline (if timestamps available)                          │
│      - Event frequency over time                                 │
│      - Anomaly spikes marked                                     │
│                                                                  │
│  5d. Severity breakdown                                          │
│      - Per-source severity heatmap                               │
│      - ERROR/WARN ratio per event type                           │
│                                                                  │
│  5e. Export                                                       │
│      - event_summary.csv (event_id, template, count, severity,  │
│        anomaly_score, anomaly_label)                             │
│      - anomalies.csv (top anomalies with examples)              │
│      - full_parsed.csv (every log with event_id assigned)        │
│                                                                  │
│  Principle: Human-readable, actionable, shareable                │
│  Target: Ops team can act on it without reading code             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Why This Architecture Is General

| Property | How the architecture handles it |
|----------|---------------------------------|
| **Unknown log format** | Probe chain falls through to raw text — nothing dropped |
| **Mixed formats in one file** | Each line detected independently — syslog, JSON, pipe can coexist |
| **No timestamps** | Timestamp field is `None`, temporal analysis gracefully skipped |
| **No severity field** | Severity is `None`, falls back to message-level keyword detection as last resort |
| **No hostname/service** | Source fields are `None`, source analysis skipped |
| **Binary/garbage lines** | Stored as raw text, format="unknown", still clustered |
| **New format never seen** | Key=value extraction catches most structured logs; worst case = raw text |
| **Massive files** | Embed unique templates only (not every line) — scales sub-linearly |
| **Single-format files** | Probe chain hits on first try — no overhead |

---

## Key Design Principles

1. **Never drop data** — every line gets an Event ID
2. **Parse what you can, skip what you can't** — graceful degradation
3. **Normalize message only** — not the entire raw line
4. **Structure enables analysis** — the more you parse, the more you can analyze, but nothing breaks if you can't
5. **Validate everything** — metrics prove the clustering works

---

## Current State vs Ideal

```
STAGE          IDEAL                          CURRENT (solution3.ipynb)          STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. INGEST      Multi-format probe chain       Read raw lines as blob            No parsing
               → structured fields            No format detection
                                              No field extraction

2. NORMALIZE   On message field only           On entire raw log line            Wrong input
               Full token coverage             Misses SKUs, case bugs            Incomplete

3. CLUSTER     Clean → embed → validate        Case bug in cleaning              Bug
               Auto-threshold + metrics        Auto-threshold (good!)            Partial
                                              No quality metrics                 Missing

4. ANALYZE     Structured severity             Keyword guessing                  Fragile
               Temporal analysis               No timestamps parsed              Missing
               Per-service breakdown           No service field                  Missing
               Composite anomaly score         Severity + rarity (good idea)     Partial

5. VISUALIZE   Scatter, bar, timeline,         Scrollable HTML tables only       Missing
               heatmap, exports
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL COMPLETION: ~36%
```

---

## Priority Implementation Order

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | **Multi-format parser (Stage 1)** — detect format, extract fields | Medium | Unlocks Stages 4 and 5 entirely |
| 2 | **Fix case mismatch bugs** — template generation + clean_for_embedding | Small | Fixes embedding quality |
| 3 | **Normalize identifiers** — SKU prefixes, all variable IDs | Small | Collapses 880 templates to ~100 |
| 4 | **Add cluster quality metrics** — silhouette score | Small | Validates clustering |
| 5 | **Add visualizations** — scatter, bar, timeline, heatmap | Medium | Makes results interpretable |
| 6 | **Time-based anomaly analysis** — burst detection | Medium | Catches temporal anomalies |
| 7 | **Per-service severity breakdown** | Small (once parser exists) | Actionable for ops |
| 8 | **Export CSV/JSON reports** | Small | Deliverable output |
