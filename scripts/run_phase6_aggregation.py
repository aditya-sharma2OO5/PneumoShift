"""Unified runner entry point for Phase 6 Results Aggregation and Statistical Meta-Analysis."""

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_results import main, run_aggregation_pipeline

if __name__ == "__main__":
    main()

