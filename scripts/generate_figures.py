"""Command-line interface to render and export all Phase 7 publication figures.

Usage:
    py -3.12 scripts/generate_figures.py --smoke_test
    py -3.12 scripts/generate_figures.py --output_dir results/figures --formats pdf,svg,png
"""

import argparse
from pathlib import Path
import sys
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pneumoshift.figures.plotters import generate_all_figures
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.generate_figures")


def main() -> None:
    parser = argparse.ArgumentParser(description="PneumoShift Phase 7 Publication Figures Generator.")
    parser.add_argument("--output_dir", default="results/figures", help="Directory where figures will be saved.")
    parser.add_argument("--tables_dir", default="results/tables/master", help="Directory of Phase 6 master tables.")
    parser.add_argument("--formats", default="pdf,svg,png", help="Comma-separated list of formats (pdf,svg,png).")
    parser.add_argument("--smoke_test", action="store_true", help="Execute rapid zero-data synthetic figure rendering.")
    args = parser.parse_args()

    format_list = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    out_dir = Path(args.output_dir)

    print("\n" + "=" * 80)
    print("  PNEUMOSHIFT PHASE 7: PUBLICATION FIGURES GENERATION ENGINE")
    print(f"  Target Output Directory : {out_dir}")
    print(f"  Export Formats          : {format_list}")
    print(f"  Execution Mode          : {'Synthetic Smoke Test' if args.smoke_test else 'Production Run'}")
    print("=" * 80 + "\n")

    results = generate_all_figures(
        output_dir=out_dir,
        tables_dir=args.tables_dir,
        formats=format_list,
        smoke_test=args.smoke_test,
    )

    print("\n" + "=" * 80)
    print("  PUBLICATION FIGURES EXPORT SUMMARY")
    print("=" * 80)
    for fig_stem, saved_dict in results.items():
        formats_str = ", ".join([f".{fmt}" for fmt in sorted(saved_dict.keys())])
        print(f"  [OK] {fig_stem:<36} -> {formats_str}")
    print("=" * 80)
    print(f"  Total Figures Rendered: {len(results)} (Total Files: {len(results) * len(format_list)})\n")


if __name__ == "__main__":
    main()

