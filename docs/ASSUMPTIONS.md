# Scientific and Engineering Assumptions (docs/ASSUMPTIONS.md)

This document records all baseline assumptions, design choices, and operational protocols for the **PneumoShift** research project ("Beyond Accuracy: Evaluating the Robustness and Explainability of Deep Learning Models for Pneumonia Detection Across Chest X-Ray Datasets").

---

## 1. Research Scope & Guardrails
- **Clinical Non-Device Status**: This codebase is strictly for scientific research, reproducible benchmarking, and evaluation of robustness and explainability under dataset shift. It is **not** a diagnostic medical device.
- **Evaluation Integrity**:
  - Test sets are never touched during training, hyperparameter tuning, model selection, or temperature/threshold fitting.
  - Source validation sets are strictly used for model selection, early stopping, source-optimal threshold calculation, and post-hoc calibration fitting.
  - The **Target Calibration Split** (~10% of target test, patient-partitioned) is utilized exclusively as an *oracle recalibration benchmark* to quantify how much of the cross-domain calibration error is recoverable with minimal target-domain supervision.

---

## 2. Data Harmonization & Splits
- **Datasets**: NIH ChestX-ray14 and Stanford CheXpert.
- **View Restriction**: Frontal projections only (AP: Anteroposterior, PA: Posteroanterior). Lateral views (prevalent in CheXpert) are dropped to maintain comparable anatomical perspective across cohorts.
- **Confounder Awareness**: CheXpert consists predominantly of AP views (inpatient/ICU bed-bound population), whereas NIH ChestX-ray14 consists predominantly of PA views (ambulatory outpatient cohort). This distribution difference is treated as an inherent cross-institutional confounder that mirrors real-world clinical deployment shifts.
- **Shared Pathological Label Space**:
  - 8 shared target classes: `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Pleural Effusion`, `Pneumonia`, `Pneumothorax`, `No Finding`.
  - Headline target: **Pneumonia**.
  - All multi-label models predict all 8 classes; pneumonia-only ablation models predict single-label pneumonia.
- **CheXpert Uncertainty Label Policy**:
  - CheXpert labels contain positive (`1`), negative (`0`), uncertain (`-1` or `u`), and unmentioned (`blank` or NaN).
  - Unmentioned (`blank`/NaN) is imputed to `0` (absence of positive indication).
  - **Primary Policy**: U-Zeros (uncertain `->` 0 / negative), matching the established protocol in Tayebi Arasteh et al. (2023).
  - **Ablation Policies** (evaluated on CheXpert-trained models):
    - *U-Ones*: uncertain `->` 1 (positive).
    - *U-Ignore*: masked loss during training where uncertain labels do not contribute to gradient updates (Irvin et al., 2019).
  - **Evaluation Policy**: For fair comparison across datasets, evaluation benchmarks report two standardized uncertainty evaluations: uncertain-as-negative (standard binary) and uncertain-excluded.
- **Patient-Level Stratification & Zero Leakage**:
  - Patient IDs are strictly disjoint across train, validation, test, and target-calibration splits.
  - A dedicated automated integrity test (`test_splits_leakage.py`) runs in CI/tests and raises a fatal assertion if any patient ID appears in more than one partition.
  - Split CSVs are hashed (SHA-256) upon generation and recorded in `results/splits_manifest.json`.

---

## 3. Modeling & Optimization
- **Architectures**:
  - Primary backbone: `DenseNet121` (feature dim 1024, pooling, linear classification head).
  - Comparison backbone: `EfficientNet-B0` (feature dim 1280, optional swappable B3).
  - Pretraining: ImageNet-1K / ImageNet-21K transfer learning.
- **Loss Function**: Multi-label Binary Cross-Entropy with pos_weight calculated from the inverse class frequency on the training set.
- **Optimization**: AdamW / Adam with initial learning rate $1 \times 10^{-4}$, weight decay $1 \times 10^{-5}$, gradient clipping at max norm 1.0, and ReduceLROnPlateau / CosineAnnealingLR (matching literature protocols in Pan et al., Irvin et al., and Tayebi Arasteh et al.).
- **Multi-Seed Replication**: Experiments are executed across a minimum of 3 fixed seeds (default: 42, 1337, 2024) to report mean $\pm$ standard deviation.

---

## 4. Calibration & Threshold Transfer
- **Calibration Metrics**:
  - Expected Calibration Error (ECE) evaluated with 15 equal-width bins and 15 adaptive equal-frequency bins.
  - Maximum Calibration Error (MCE), Brier Score, and Negative Log-Likelihood (NLL).
- **Post-Hoc Calibration Methods**:
  - Source Temperature Scaling: Learned scalar temperature $T > 0$ optimized via NLL on source validation.
  - Oracle Recalibration: Temperature scaling fitted on the disjoint target calibration split.
- **Threshold Transfer**:
  - Source Youden $J$-optimal threshold and source 90%-sensitivity threshold applied directly to the cross-domain target test set.
  - Compared against the target-optimal threshold to calculate the absolute drop in sensitivity, specificity, and F1 score.

---

## 5. Explainability (Grad-CAM)
- **Target Layers**:
  - DenseNet121: `features.denseblock4` / `features.norm5`.
  - EfficientNet-B0: final convolutional feature stage before global pooling (`conv_head` or equivalent in timm).
- **Quantitative Metrics**:
  - *Faithfulness*: Area Under Deletion Curve (AUDC) and Area Under Insertion Curve (AUIC) with progressive top-$k\%$ masking.
  - *Sanity Checks*: Complete and cascading layer randomization (Adebayo et al., 2018), comparing attribution similarity (Spearman rank correlation, SSIM) to unrandomized models.
  - *Stability*: Map SSIM and rank correlation across training seeds and under mild input perturbations ($\pm 5^\circ$ rotation, $\pm 10\%$ contrast).
  - *Anatomical Plausibility*: Proportion of attribution mass contained within the anatomical lung bounding field / lung segmentation mask.
  - *Shortcut Analysis*: Attribution energy located in peripheral borders (top/bottom/lateral 5%), corner margins, and known marker regions.

