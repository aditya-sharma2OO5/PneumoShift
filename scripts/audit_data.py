"""CLI script to audit dataset distributions, prevalences, demographics, and intensities."""

import argparse
from pathlib import Path
import pandas as pd
from pneumoshift.data.audit import run_data_audit
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.audit_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset splits for PneumoShift.")
    parser.add_argument("--splits_dir", default="data/splits", help="Directory containing split CSVs")
    parser.add_argument("--dataset", choices=["nih", "chexpert", "synthetic"], default="synthetic")
    parser.add_argument("--images_dir", default=None, help="Base path to image files")
    parser.add_argument("--output_dir", default="results/tables/audit", help="Directory to save audit outputs")
    args = parser.parse_args()

    splits_path = Path(args.splits_dir)
    ds = args.dataset

    # Discover split files
    split_names = ["train", "val", "test", "target_cal"]
    splits = {}

    for name in split_names:
        possible_csv = splits_path / f"{ds}_{name}.csv"
        if not possible_csv.is_file():
            # Check subdirectory
            possible_csv = splits_path / ds / f"{ds}_{name}.csv"

        if possible_csv.is_file():
            splits[name] = pd.read_csv(possible_csv)
            logger.info("Loaded split '%s' with %d images.", name, len(splits[name]))

    if not splits:
        logger.error("No split CSVs found in %s matching dataset '%s'.", splits_path, ds)
        return

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_dir = args.images_dir
    if img_dir is None and ds == "synthetic":
        img_dir = "data/synthetic/images"

    logger.info("Running data audit on %s splits...", ds)
    report = run_data_audit(splits, images_dir=img_dir, output_dir=out_path, generate_grid=True)
    logger.info("Audit finished. Summary saved to %s", out_path)


if __name__ == "__main__":
    main()

