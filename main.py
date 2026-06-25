# main.py
#
# ══════════════════════════════════════════════════════════════════════
# MAIN — project root entry point
# ══════════════════════════════════════════════════════════════════════
#
# Run the full pipeline from the project root:
#   python main.py
#   python main.py campaign-template-generator-6.log
#   python main.py path/to/any.log
#
# Every run creates a unique timestamped folder under outputs/runs/
# so previous results are never overwritten.
#
# IMPORTANT: Always run from the PROJECT ROOT, never from inside
# the backend/ folder. Relative imports will break otherwise.
#
# PLACEMENT: project root (same level as config.py)
# ══════════════════════════════════════════════════════════════════════

import sys
from backend.pipeline import run_pipeline

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    results  = run_pipeline(log_path)

    ri = results["run_info"]
    print(f"\n{'=' * 60}")
    print(f"  Run ID    : {ri['run_id']}")
    print(f"  Log file  : {ri['log_filename']}")
    print(f"  Incidents : {ri['total_incidents']}")
    print(f"  Anomalies : {ri['total_scored_clusters']}")
    print(f"  Severity  : {ri['incident_severity_counts']}")
    print(f"\n  Results saved to:")
    print(f"  {results['run_folder']}")
    print(f"{'=' * 60}")