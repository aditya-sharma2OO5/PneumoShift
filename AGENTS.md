# PneumoShift: Scientific Machine Learning & Reproducibility Guidelines

This repository enforces strict medical machine learning research standards, single-script reproducibility, and empirical integrity. Every agent operating in this codebase must adhere to the following invariants:

## 1. Empirical Single Source of Truth Invariant
- **Zero Synthetic Curves**: Never use mathematical parametric approximations (such as power laws $TPR = FPR^a$, exponential decay PR formulas, or Gaussian bell F1 curves). All curves in publication figures must be computed strictly via empirical functions (`roc_curve()`, `precision_recall_curve()`) on discrete `(y_true, y_score)` arrays.
- **On-Curve Operating Points**: Any operating point marker (e.g., Youden's J cutoff $(FPR, TPR)$) must be indexed directly from the empirical `roc_curve()` coordinates so it mathematically sits strictly on the line.
- **No Hardcoded Metric Badges**: Never hardcode evaluation metrics, confidence intervals, p-values, or standard error bars (`yerr=[0.008, 0.007]`) in plotting functions. All figure annotations and error bars must be dynamically calculated from loaded arrays or generated master tables.

## 2. Continuous Latent Score Modeling (Zero Discontinuity Artifacts)
- When generating baseline, cross-domain, or synthetic evaluation arrays, **never use piecewise hard thresholding** (e.g., manually overriding top $k$ positive samples with high scores while leaving the rest low).
- **Smooth Bi-Normal / Latent Gaussian Model**: Always model classification scores through continuous latent logits:
  $$z_{neg} \sim \mathcal{N}(\mu_{neg}, \sigma^2), \quad z_{pos} \sim \mathcal{N}(\mu_{neg} + \Delta, \sigma^2)$$
  with probabilities $p = \sigma(z) = \frac{1}{1 + e^{-z}}$.
- This guarantees strictly continuous, concave ROC curves, realistic Precision-Recall curves that decay smoothly towards cohort prevalence, and prevents artificial 90-degree step functions or vertical PR spikes.

## 3. Zero Target Domain Leakage Protocol
- **Strict Validation Selection**: Clinical operating thresholds ($\tau$) and temperature scaling parameters ($T$) must be selected or fitted strictly on the source validation cohort (`predictions_val_*.npz`).
- **Blind Transfer**: Transferred thresholds must be applied to target test cohorts (`predictions_cross_domain_*.npz`) without adjustment. Target-optimal thresholds are evaluated only as hypothetical oracles for discordance gap quantification.

## 4. Multi-Seed Replication & Early Stopping
- All deep learning architectures must be replicated across at least 3 distinct random seeds (default: `42, 1337, 2024`).
- Training must be capped (maximum 10 epochs) with early stopping on the primary clinical finding validation metric (`val_pneumonia_auroc`, patience 3) to prevent majority-class overfitting and compute waste.
- Report all metrics as empirical mean $\pm$ standard deviation across seeds.

## 5. Single-Command Reproducibility & Auditing
- Master script `scripts/regenerate_all.py` must remain functional and self-contained, capable of ingesting raw `.npz` arrays from `results/predictions/`, updating all master tables in CSV/JSON/MD/LaTeX, rendering all figures in PDF/SVG/PNG, and running automated consistency checks (`consistency_report.json`) in under 20 seconds.
- Every figure file (18 total across 6 figures $\times$ 3 formats) must exist and be non-empty.

## 6. Transparent Literature Benchmark Positioning
- Never claim unconditioned superiority over literature benchmarks (such as Pan et al., 2019) when patient split protocols differ.
- Always disclose whether evaluations adhere to official patient-partitioned splits (`test_list.txt`, zero patient leakage) versus image-level random splits.

