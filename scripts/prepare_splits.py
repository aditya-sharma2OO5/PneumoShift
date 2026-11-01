"""Script to generate patient-partitioned splits with zero-leakage validation."""

import argparse
from pathlib import Path
from pneumoshift.data.splits import (
    create_chexpert_splits,
    create_nih_splits,
    create_synthetic_dataset,
    verify_patient_leakage,
)
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.prepare_splits")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate zero-leakage patient splits for PneumoShift.")
    parser.add_argument("--dataset", choices=["nih", "chexpert", "synthetic", "all"], default="synthetic")
    parser.add_argument("--smoke_test", action="store_true", help="Generate synthetic dry-run data")
    parser.add_argument("--nih_metadata", default="data/nih_cxr/Data_Entry_2017_v2020.csv")
    parser.add_argument("--nih_train_val_list", default="data/nih_cxr/train_val_list.txt")
    parser.add_argument("--nih_test_list", default="data/nih_cxr/test_list.txt")
    parser.add_argument("--chexpert_train_csv", default="data/chexpert/CheXpert-v1.0-small/train.csv")
    parser.add_argument("--output_dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    if args.smoke_test or args.dataset == "synthetic":
        synth_dir = Path("data/synthetic")
        logger.info("Generating synthetic smoke test dataset at %s...", synth_dir)
        paths = create_synthetic_dataset(output_dir=synth_dir, num_samples=30, image_size=64, seed=args.seed)
        logger.info("Synthetic splits generated successfully: %s", paths)

    if args.dataset in ("nih", "all") and not args.smoke_test:
        nih_dir = out_base / "nih"
        logger.info("Generating NIH ChestX-ray14 patient splits...")
        paths = create_nih_splits(
            metadata_csv=args.nih_metadata,
            train_val_list=args.nih_train_val_list,
            test_list=args.nih_test_list,
            output_dir=nih_dir,
            seed=args.seed,
        )
        logger.info("NIH splits generated: %s", paths)

    if args.dataset in ("chexpert", "all") and not args.smoke_test:
        chex_dir = out_base / "chexpert"
        logger.info("Generating Stanford CheXpert patient splits...")
        paths = create_chexpert_splits(
            train_csv=args.chexpert_train_csv,
            output_dir=chex_dir,
            seed=args.seed,
        )
        logger.info("CheXpert splits generated: %s", paths)


if __name__ == "__main__":
    main()

