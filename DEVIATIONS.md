# Protocol Deviations & Pre-Analysis Audit Findings Log

This log records findings and pre-analysis decisions in accordance with Section 12.1 of [`ANALYSIS_PLAN_TRANSFER.md`](ANALYSIS_PLAN_TRANSFER.md).  
The locked pre-specified protocol (`ANALYSIS_PLAN_TRANSFER.md` v2.2.3, LF-normalized SHA-256 `10baae5f06a6e4e33035f2acabaeae48b01ca8386d5ac318e8ddd8b41e465d48`, tag `prov-hardening-v3`) remains unmodified.

---

## Record PRE-E-001: Corrupted CheXpert Scans & Extreme Finite Logits in `efficientnet_b0_2024`

- **Type**: Label-blind QC audit finding and pre-specified sensitivity rule (**not** a post-hoc deviation).
- **Date**: 2026-11-25
- **Label-Blind Status**: Recorded prior to Task E evaluation; zero CheXpert label arrays (`y_true_*`, `raw_pneumonia*`, `mask_*`) or outcome metrics (AUROC, calibration, NLL) were loaded or inspected when establishing this rule.
- **Source Run**: `oldckpt_real_v1` (`predictions_chexpert_frontal_efficientnet_b0_2024.npz`).

### 1. Audit Finding
During metadata/provenance QC ([`outputs/audit_predictions.txt`](outputs/audit_predictions.txt) and [`outputs/diag_extreme_logits.txt`](outputs/diag_extreme_logits.txt)), exactly **two rows** across all six models ($6 \times 191,027 = 1,146,162$ model-image evaluations) exhibited extreme finite logits (`|logit| > 80`), both exclusively in `efficientnet_b0_2024`:

1. **Row `68292`**:
   - **Metadata**: `CheXpert-v1.0-small/train/patient20683/study2/view1_frontal.jpg` | Projection: `PA` | Sex: `Male` | Age: `64.0` | Patient ID: `patient20683`
   - **Image Inspection (Kaggle source JPEG)**: `356x320`, `mode=L`, `min=0`, `max=255`, `mean=150.4`, `std=59.8`, `frac_0=0.000`, `frac_255=0.003`. Visually pure vertical/textured static noise with no visible thoracic anatomy.
   - **Model Output (`efficientnet_b0_2024`)**: Finite, non-zero penultimate features (`feat_absmax = 57.72`). Disease logits `[-23.17, -62.78, -22.67, -98.44, -31.98, -54.97, -40.94]`, `No Finding = +2.35` (`Edema = -98.44` triggers `|logit| > 80`; `Pneumonia = -54.97`).

2. **Row `76123`**:
   - **Metadata**: `CheXpert-v1.0-small/train/patient23121/study4/view2_frontal.jpg` | Projection: `AP` | Sex: `Male` | Age: `83.0` | Patient ID: `patient23121`
   - **Image Inspection (Kaggle source JPEG)**: `389x320`, `mode=L`, `min=0`, `max=255`, `mean=175.9`, `std=78.3`, `frac_0=0.001`, `frac_255=0.168`. Visually pure high-contrast salt-and-pepper static noise (16.8% saturated white pixels), completely non-anatomical.
   - **Model Output (`efficientnet_b0_2024`)**: Finite, non-zero penultimate features (`feat_absmax = 139.88`). Disease logits `[-23.80, -119.12, -40.16, -200.38, -63.03, -123.38, -108.50]`, `No Finding = +35.38` (`Cardiomegaly`, `Edema`, `Pneumonia`, `Pneumothorax` have `|logit| > 80`; `Pneumonia = -123.38`).

All other five checkpoints (`densenet121_42`, `densenet121_1337`, `densenet121_2024`, `efficientnet_b0_42`, `efficientnet_b0_1337`) produced ordinary bounded logits (within roughly `[-5, +2]`) on both rows, all six files contain `0` NaN/Inf logits and `0` all-zero feature rows, and `EXCLUSIONS.json` records `n_excluded = 0` (`191,027` rows present).

### 2. Reason
Both JPEGs decoded without PIL I/O errors and therefore were not caught by the unreadable-file exclusion rule (§2.4). However, because `y_score` underflows to exact `0.0` at logit `-123.38` in `float32`, computing NLL from `y_score` would produce `inf` (or arbitrary probability clipping artifacts), and an extreme negative logit could heavily influence NLL-based calibration metrics if that image carries a positive label.

### 3. Original Analysis (`ANALYSIS_PLAN_TRANSFER.md` v2.2.3)
Section 2.4 specifies exclusion only for images that fail to decode or load (`EXCLUSIONS.json`, with a >0.1% failure patient-exclusion sensitivity rule). The plan is silent on decoded corrupted-content scans and extreme finite logits (`|logit| > 80`).

### 4. Changed / Pre-Specified Analysis Rule (For Task E)
1. **Primary Analysis (Full Cohort Retained, Numerically Stable Logits)**:
   - Retain all $N = 191,027$ frontal images (`0` dropped, **no** logit clipping or winsorization).
   - Compute NLL, Platt scaling, temperature scaling, logistic calibration slope/intercept, and Brier calculations directly from `y_logits` using numerically stable logit formulations (`logaddexp` / `log_sigmoid` / `scipy.special.expit`), **never** from pre-computed `y_score`.
2. **Pre-Specified Sensitivity Analysis (`efficientnet_b0_2024`)**:
   - Report `efficientnet_b0_2024` discrimination and calibration/NLL results side-by-side for:
     - **(a) Full cohort**: $N = 191,027$ images (primary); and
     - **(b) Corrupted-scan sensitivity cohort**: $N = 191,025$ images excluding rows `68292` and `76123`.
3. **Scope Boundary**: No post-hoc visual audit of the remaining `191,025` CheXpert images will be performed; outlier identification remains strictly bounded by the label-blind `|logit| > 80` check prior to Task E.
