"""Command-line interface to generate the standalone interactive HTML gallery dashboard.

Usage:
    py -3.12 scripts/generate_gallery.py
    py -3.12 scripts/generate_gallery.py --output_file reports/gallery.html --smoke_test
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pneumoshift.figures.gallery import generate_gallery
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.generate_gallery")


def main() -> None:
    parser = argparse.ArgumentParser(description="PneumoShift Phase 7 Interactive Gallery Generator.")
    parser.add_argument("--output_file", default="reports/gallery.html", help="Path to destination HTML file.")
    parser.add_argument("--tables_dir", default="results/tables/master", help="Directory of master aggregated tables.")
    parser.add_argument("--figures_dir", default="results/figures", help="Directory of publication figures.")
    parser.add_argument("--smoke_test", action="store_true", help="Execute in rapid zero-data fallback mode.")
    args = parser.parse_args()

    out_p = Path(args.output_file)

    print("\n" + "=" * 80)
    print("  PNEUMOSHIFT PHASE 7: INTERACTIVE HTML GALLERY GENERATOR")
    print(f"  Target File    : {out_p}")
    print(f"  Tables Source  : {args.tables_dir}")
    print(f"  Figures Source : {args.figures_dir}")
    print(f"  Mode           : {'Synthetic Smoke Test' if args.smoke_test else 'Production Run'}")
    print("=" * 80 + "\n")

    saved_path = generate_gallery(
        output_file=out_p,
        tables_dir=args.tables_dir,
        figures_dir=args.figures_dir,
        smoke_test=args.smoke_test,
    )

    size_kb = saved_path.stat().st_size / 1024.0
    print("\n" + "=" * 80)
    print(f"  [PASS] Standalone Interactive Gallery Generated Successfully!")
    print(f"  File Location : {saved_path.resolve()}")
    print(f"  File Size     : {size_kb:.1f} KB")
    print("  Zero external CDN or internet dependencies required.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

