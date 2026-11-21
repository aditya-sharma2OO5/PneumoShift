# PneumoShift: Scientific Machine Learning & Reproducibility Guidelines

This repository enforces strict medical machine learning research standards, single-script reproducibility, and empirical integrity. Every agent operating in this codebase must adhere to the following invariants:

## 1. Empirical Single Source of Truth Invariant
- **Stamped Real Inference Only**: No table or figure may use anything but stamped real-inference predictions (`data_source == "real_inference"`) verified against `checkpoint_manifest.json`. Access to quarantined or unverified data is strictly prohibited and guarded.
- **Zero Synthetic Curves or Parametric Approximations**: Never use mathematical parametric approximations (such as power laws $TPR = FPR^a$, exponential decay PR formulas, or Gaussian bell F1 curves). All curves in publication figures must be computed strictly via empirical functions (`roc_curve()`, `precision_recall_curve()`) on discrete `(y_true, y_score)` arrays loaded from verified real inference.
- **Raw Empirical Curves (Zero Cosmetic Smoothing)**: Never generate, smooth, tune, or alter data to improve visual appearance. ROC and Precision-Recall curves are drawn strictly from raw empirical coordinates. At ~2% clinical prevalence, step-like, discrete, or jagged PR curves are the expected mathematical reality; smoothing or simulating continuous distributions to eliminate steps is scientific fabrication.
- **On-Curve Operating Points**: Any operating point marker (e.g., Youden's J cutoff $(FPR, TPR)$) must be indexed directly from the empirical `roc_curve()` coordinates so it mathematically sits strictly on the empirical curve.
- **No Hardcoded Metric Badges**: Never hardcode evaluation metrics, confidence intervals, p-values, or standard error bars (`yerr=[0.008, 0.007]`) in plotting functions. All figure annotations and error bars must be dynamically calculated from verified empirical arrays or master tables.

## 2. Zero Placeholder & Zero Fabrication Policy
- **Never Put Placeholder Data Under results/**: The `results/` hierarchy (`results/predictions/`, `results/tables/`, `results/figures/`) is strictly reserved for verified empirical artifacts. Never generate or store placeholder, dummy, simulated, or fallback data in `results/`.
- **Isolated Smoke Test Execution**: Rapid smoke tests, dry runs, and CI checks must route their outputs strictly to `outputs/smoke/<run_id>/` and must never pollute `results/`.
- **Refusal to Overwrite Empirical Results**: Prediction-generating scripts must refuse to overwrite existing files on disk unless `--allow_overwrite` is explicitly provided.
- **Immediate Rejection of Quarantined Data**: All data loaders in evaluation, aggregation, and figure generation must raise an immediate error if attempting to access files in `results/quarantine/` or any file lacking the authorized provenance stamp.

## 3. Zero Target Domain Leakage Protocol
- **Strict Validation Selection**: Clinical operating thresholds ($\tau$) and temperature scaling parameters ($T$) must be selected or fitted strictly on the source validation cohort (`predictions_val_*.npz`).
- **Blind Transfer**: Transferred thresholds must be applied to target test cohorts (`predictions_cross_domain_*.npz`) without adjustment. Target-optimal thresholds are evaluated only as hypothetical oracles for discordance gap quantification.

## 4. Multi-Seed Replication & Early Stopping
- All deep learning architectures must be replicated across at least 3 distinct random seeds (default: `42, 1337, 2024`).
- Training must be capped (maximum 10 epochs) with early stopping on the primary clinical finding validation metric (`val_pneumonia_auroc`, patience 3) to prevent majority-class overfitting and compute waste.
- Report all metrics as empirical mean $\pm$ standard deviation across seeds.

## 5. Single-Command Reproducibility & Auditing
- Master aggregation scripts must remain functional and self-contained, capable of ingesting raw `.npz` arrays from `results/predictions/`, updating all master tables in CSV/JSON/MD/LaTeX, rendering all figures in PDF/SVG/PNG, and running automated consistency checks in under 20 seconds.
- Every figure file (18 total across 6 figures $\times$ 3 formats) must exist and be non-empty.

## 6. Transparent Literature Benchmark Positioning
- Never claim unconditioned superiority over literature benchmarks (such as Pan et al., 2019) when patient split protocols differ.
- Always disclose whether evaluations adhere to official patient-partitioned splits (`test_list.txt`, zero patient leakage) versus image-level random splits.
