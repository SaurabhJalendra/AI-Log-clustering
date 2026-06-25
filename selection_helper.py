#!/usr/bin/env python3
"""
selection_helper.py
──────────────────────────────────────
Stratified sampler for ground truth labelling. Scales to large log files.

Basic usage (small files, ~250 rows):
    python selection_helper.py --pipeline_output outputs/sample_logs_output.csv

Large file usage (200k+ logs):
    python selection_helper.py --pipeline_output outputs/bigfile_output.csv \
                                --n 600 \
                                --max_error 150 --max_warn 100 \
                                --min_per_domain 15 \
                                --out validation/bigfile_rows_to_label.csv

Stratified selection tiers (run in order; each tier skips already-selected rows):
  1. ERROR rows            — up to --max_error  (default 80)
  2. WARN rows             — up to --max_warn   (default 60)
  3. Domain floor          — top up every domain to --min_per_domain rows each
                             (skipped when min_per_domain == 0, the default)
  4. One row per unique template_id not yet selected
  5. Random INFO/DEBUG/other rows to reach --n total

Output CSV has all pipeline columns plus empty expected_* columns ready
for manual labelling in Excel. After labelling, append to
validation/master_ground_truth.csv with source_file column set.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

# ── GROUND TRUTH COLUMNS (empty — labeller fills these in) ────────────
_GT_COLUMNS = [
    "expected_domain",
    "expected_severity",
    "expected_template_id",
    "expected_singleton_class",
    # anomaly_signal: label as one of:
    #   true_anomaly | unseen_variant | routine | noise_filtered
    "expected_anomaly_signal",
    # domain_confidence_band: coarse bucket — do NOT look at the
    # domain_confidence float first; judge from the message directly.
    #   low    = domain assignment is a guess / message is ambiguous
    #   medium = probably right but message is somewhat generic
    #   high   = domain is obvious from the message
    "expected_domain_confidence_band",
    "notes",
]


def _load_pipeline_output(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if "line_no" in df.columns:
        df["line_no"] = pd.to_numeric(df["line_no"], errors="coerce")
    if "severity" in df.columns:
        df["severity"] = df["severity"].str.upper().str.strip()
    return df


def _stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    max_error: int,
    max_warn: int,
    min_per_domain: int,
) -> pd.DataFrame:
    """
    Build a stratified sample of up to n rows.

    Tiers (each tier excludes rows already selected):
      1. ERROR rows sorted by line_no              — up to max_error
      2. WARN  rows sorted by line_no              — up to max_warn
      3. Domain floor: top up every domain to      — min_per_domain rows each
         (skipped when min_per_domain == 0)
      4. One row per unique template_id not yet selected
      5. Random INFO/DEBUG/other rows to reach n
    """
    rng = random.Random(seed)
    selected_line_nos: Set = set()
    frames: List[pd.DataFrame] = []

    def _add(rows: pd.DataFrame, cap: int) -> None:
        """Add up to cap rows (not already selected) and track their line_nos."""
        rows = rows[~rows["line_no"].isin(selected_line_nos)]
        rows = rows.sort_values("line_no").head(cap)
        selected_line_nos.update(rows["line_no"].tolist())
        if not rows.empty:
            frames.append(rows)

    def _remaining() -> int:
        return n - len(selected_line_nos)

    # ── Tier 1: ERROR ────────────────────────────────────────────────
    _add(df[df["severity"] == "ERROR"], max_error)

    # ── Tier 2: WARN ─────────────────────────────────────────────────
    _add(df[df["severity"] == "WARN"], max_warn)

    # ── Tier 3: Domain floor ─────────────────────────────────────────
    # For each domain, count how many rows are already selected, then
    # pull additional rows (any severity) until the domain reaches
    # min_per_domain, or we hit the overall cap n.
    if min_per_domain > 0 and "domain" in df.columns and _remaining() > 0:
        # Count already-selected rows per domain
        if frames:
            selected_so_far = pd.concat(frames, ignore_index=True)
            domain_counts: Dict[str, int] = (
                selected_so_far["domain"].value_counts().to_dict()
            )
        else:
            domain_counts = {}

        for domain in df["domain"].dropna().unique():
            if _remaining() <= 0:
                break
            already_have = domain_counts.get(str(domain), 0)
            need = min_per_domain - already_have
            if need <= 0:
                continue
            pool = df[
                (df["domain"] == domain)
                & ~df["line_no"].isin(selected_line_nos)
            ]
            if pool.empty:
                continue
            take = min(need, _remaining(), len(pool))
            picked = pool.sample(n=take, random_state=seed) if len(pool) > take else pool
            _add(picked, take)

    # ── Tier 4: One row per unique template_id ────────────────────────
    if "template_id" in df.columns and _remaining() > 0:
        seen_tids: Set = set()
        tmpl_rows: list = []
        for _, row in df.sort_values("line_no").iterrows():
            tid = row.get("template_id", "")
            if pd.isna(tid) or str(tid).strip() in ("", "nan"):
                continue
            if tid in seen_tids or row["line_no"] in selected_line_nos:
                continue
            seen_tids.add(tid)
            tmpl_rows.append(row)
        if tmpl_rows:
            _add(pd.DataFrame(tmpl_rows), len(tmpl_rows))

    # ── Tier 5: Random INFO/DEBUG/other to fill up to n ──────────────
    if _remaining() > 0:
        pool = df[
            ~df["line_no"].isin(selected_line_nos)
            & ~df["severity"].isin(["ERROR", "WARN"])
        ]
        if not pool.empty:
            take = min(_remaining(), len(pool))
            picked = pool.sample(n=take, random_state=seed) if len(pool) > take else pool
            _add(picked, take)

    if not frames:
        return pd.DataFrame()

    result = (
        pd.concat(frames, ignore_index=True)
        .sort_values("line_no")
        .drop_duplicates(subset=["line_no"])
        .reset_index(drop=True)
    )
    return result.head(n)


def _print_coverage_summary(df: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Print a breakdown of how well the sample covers the full file."""
    total   = len(df)
    sampled = len(sample)
    print(f"\n  Coverage summary  ({sampled:,} / {total:,} rows = {sampled/total*100:.2f}%)")

    # Severity
    print("\n  Severity:")
    for sev in ["ERROR", "WARN", "INFO", "DEBUG"]:
        full_n = int((df["severity"] == sev).sum()) if "severity" in df.columns else 0
        samp_n = int((sample["severity"] == sev).sum()) if "severity" in sample.columns else 0
        pct    = samp_n / full_n * 100 if full_n else 0
        bar    = "█" * min(int(pct / 5), 20)
        print(f"    {sev:<6}  {samp_n:>4} / {full_n:>6}  ({pct:5.1f}%)  {bar}")

    # Domain
    if "domain" in df.columns:
        print("\n  Domain:")
        for dom in sorted(df["domain"].dropna().unique()):
            full_n = int((df["domain"] == dom).sum())
            samp_n = int((sample["domain"] == dom).sum()) if "domain" in sample.columns else 0
            pct    = samp_n / full_n * 100 if full_n else 0
            bar    = "█" * min(int(pct / 5), 20)
            flag   = "  ⚠ low coverage" if samp_n < 5 else ""
            print(f"    {dom:<20}  {samp_n:>4} / {full_n:>6}  ({pct:5.1f}%)  {bar}{flag}")

    # Template IDs
    if "template_id" in df.columns:
        full_tids = df["template_id"].dropna().nunique()
        samp_tids = sample["template_id"].dropna().nunique() if "template_id" in sample.columns else 0
        print(f"\n  Template IDs covered: {samp_tids} / {full_tids}")
        if samp_tids < full_tids:
            missing = (
                set(df["template_id"].dropna().unique())
                - set(sample["template_id"].dropna().unique())
            )
            print(f"  ⚠  {len(missing)} template_id(s) not represented.")
            print(f"     Consider raising --n to cover them.")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate stratified rows_to_label.csv for ground truth annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tier cap guidance by file size:
  ~5k  rows  → --n 250  --max_error  80 --max_warn  60              (defaults)
  ~20k rows  → --n 400  --max_error 120 --max_warn  80  --min_per_domain 10
  ~50k rows  → --n 500  --max_error 150 --max_warn 100  --min_per_domain 12
  200k+ rows → --n 600  --max_error 200 --max_warn 120  --min_per_domain 15
        """,
    )
    parser.add_argument("--pipeline_output", required=True,
                        help="Path to pipeline output CSV")
    parser.add_argument("--n", type=int, default=250,
                        help="Total rows to sample (default: 250)")
    parser.add_argument("--max_error", type=int, default=80,
                        help="Tier 1 cap: max ERROR rows to include (default: 80)")
    parser.add_argument("--max_warn", type=int, default=60,
                        help="Tier 2 cap: max WARN rows to include (default: 60)")
    parser.add_argument("--min_per_domain", type=int, default=0,
                        help="Tier 3: guarantee at least this many rows per domain "
                             "(default: 0 = off). Recommended: 10-15 for large files.")
    parser.add_argument("--out", default="validation/rows_to_label.csv",
                        help="Output path (default: validation/rows_to_label.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--source_file", default=None,
                        help="Value for source_file column (defaults to pipeline_output stem)")
    args = parser.parse_args(argv)

    output_path = Path(args.pipeline_output)
    out_path    = Path(args.out)
    source_file = args.source_file or output_path.stem

    if not output_path.exists():
        print(f"❌  Pipeline output not found: {output_path}")
        sys.exit(1)

    print(f"\nLoading pipeline output: {output_path}")
    df = _load_pipeline_output(output_path)
    total_rows = len(df)
    print(f"  {total_rows:,} total rows")
    if "severity" in df.columns:
        print(f"  Severity breakdown: {df['severity'].value_counts().to_dict()}")
    if "domain" in df.columns:
        print(f"  Domain breakdown:   {df['domain'].value_counts().to_dict()}")

    # Warn if defaults look undersized for this file
    if total_rows > 50_000 and args.n == 250:
        print(
            f"\n  ⚠  File has {total_rows:,} rows but --n is still 250 (the default).\n"
            f"     For files this large consider:\n"
            f"       --n 600 --max_error 200 --max_warn 120 --min_per_domain 15"
        )

    print(
        f"\n  Sampling config:  n={args.n}  max_error={args.max_error}  "
        f"max_warn={args.max_warn}  min_per_domain={args.min_per_domain}"
    )

    sample = _stratified_sample(
        df,
        n=args.n,
        seed=args.seed,
        max_error=args.max_error,
        max_warn=args.max_warn,
        min_per_domain=args.min_per_domain,
    )

    _print_coverage_summary(df, sample)

    # Add empty ground truth columns for the labeller
    for col in _GT_COLUMNS:
        sample[col] = ""

    # Add source_file column for master_ground_truth.csv merge
    sample.insert(0, "source_file", source_file)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)

    print(f"✅  Saved {len(sample)} rows → {out_path}")
    print()
    print("Next steps:")
    print("  1. Open rows_to_label.csv in Excel.")
    print("  2. Fill ALL expected_* columns for each row:")
    print("       expected_domain              — e.g. network, auth, payment, storage …")
    print("       expected_severity            — ERROR | WARN | INFO | DEBUG")
    print("       expected_template_id         — copy from template_id if correct,")
    print("                                      write WRONG if mismatched,")
    print("                                      write MISSING if blank")
    print("       expected_singleton_class     — true_anomaly | unseen_variant |")
    print("                                      impossible_attempt_count | known_normal |")
    print("                                      noise_filtered")
    print("                                      (leave blank ONLY for INFO/DEBUG rows")
    print("                                      you are certain are routine)")
    print("       expected_anomaly_signal      — true_anomaly | unseen_variant |")
    print("                                      routine | noise_filtered")
    print("       expected_domain_confidence_band — low | medium | high")
    print("                                         judge from the message, NOT the")
    print("                                         domain_confidence float column")
    print("     Use the 'notes' column for any ambiguous cases.")
    print()
    print("  3. ⚠  IMPORTANT: Do NOT look at the pipeline output columns first.")
    print("     Cover domain/severity/singleton_class while you label, then")
    print("     reveal them to check. Copying from the pipeline defeats the point.")
    print()
    print("  4. Save the file.")
    print("  5. Append to validation/master_ground_truth.csv")
    print("     (never replace — only append, with source_file column set).")
    print(f"  6. Run: python validate_pipeline.py --pipeline_output {args.pipeline_output}")
    print()


if __name__ == "__main__":
    main()