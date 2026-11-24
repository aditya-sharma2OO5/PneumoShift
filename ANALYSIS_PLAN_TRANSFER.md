# Pre-Specified Analysis Plan: Cross-Domain Transfer Evaluation (NIH to CheXpert)

**Document Version**: 2.2.3  
**Date**: 2026-11-24  
**Protocol Status**: Pre-specified; committed to the project repository before CheXpert forward inference (an OSF registration, if made, is recorded in the version history)  
**Target Repository**: `https://github.com/aditya-sharma2OO5/PneumoShift.git`  
**Execution Run ID**: `oldckpt_real_v1`

---

## 1. Study Overview & Objective
This pre-specified protocol specifies the evaluation methodology for assessing the out-of-distribution transferability of deep neural networks trained on the NIH ChestX-ray14 dataset when applied to the external CheXpert dataset. All models, model weights, decision thresholds, label mapping schemes, and primary endpoints are locked prior to running CheXpert inference.

All results will be reported fully and transparently regardless of observed direction (i.e., whether indicating performance degradation, no statistically significant change, or performance improvement).

---

## 2. Evaluation Cohorts & Demographics

### 2.1 Primary Target Cohort (CheXpert Frontal)
- **Source**: `train.csv` from CheXpert (`willarevalo/chexpert-v10-small`).
- **View Selection**: Frontal views strictly (Anteroposterior [AP] and Posteroanterior [PA]). Lateral views (`Frontal/Lateral == "Lateral"`) are excluded.
- **Measured Empirical Baseline (from `CHEXPERT_COHORT_MEASURED.json`)**:
  - **Total Frontal Radiographs ($N$)**: **191,027**
  - **Unique Patients**: **64,534**
  - **Images per Patient**: Mean **2.96**
- **Runtime Measurement Specification**:
  Subgroup counts, acquisition projection breakdowns (AP vs. PA), demographic distributions (patient sex and age bands: `<40`, `40-64`, `\ge 65` years), and image resolution dimensions are not assumed a priori. Instead, all such distributions are measured empirically at runtime directly from the dataset manifests and written to `CHEXPERT_COHORT_MEASURED.json`.

### 2.2 Reference In-Domain Cohort (NIH Test)
- **Source**: `nih_test.csv` from NIH ChestX-ray14 (`data/splits/nih/nih_test.csv`).
- **Cohort Dimensions**:
  - **Total Radiographs ($N$)**: **22,744**
  - **Positive Pneumonia Radiographs**: **491** (radiograph-level prevalence: **2.16%**)
  - **Unique Patients**: **2,518**
  - **Positive Pneumonia Patients**: **277**
  - **Images per Patient**: Mean **9.03**
  - **Patient-Level Pneumonia Prevalence**: **11.00%** ($277 / 2,518$)
- **Prevalence Characterization & Comparability**:
  At the image level, disease prevalence is directly comparable between cohorts: **2.16%** in the NIH test cohort (491 / 22,744) versus **2.45%** in CheXpert frontal under primary `u_zeros` (4,675 / 191,027). At the patient level, the NIH test list is enriched relative to NIH train/val (patient prevalence 2.47% in train and 2.62% in val versus 11.00% in test). This structural clustering (mean 9.03 images/patient) necessitates patient-clustered resampling and prevalence-robust evaluation metrics.
- **Execution Run Origin**:
  NIH validation and NIH test outputs used for threshold derivation and for any NIH-vs-CheXpert contrast are fresh forward passes of the same six checkpoints (verified by SHA-256), the same code tag and the same run_id (oldckpt_real_v1) as the CheXpert outputs. They are produced in the NIH-gate stage and may belong to a different Kaggle session than the CheXpert stage; the GPU model, torch version and git commit recorded in the prediction stamps must be identical across stages. Stored predictions serve only as the reference for the NIH gate.
- **Reference Stored Predictions Bundle**:
  `configs/nih_test_reference_logits.npz` (raw SHA-256: `026fb68e71d0cbb944f446a9eb643ef3d7e0c5fb413627c1a6a92a12a246feb8`). Holds pre-computed reference predictions for both NIH validation ($N=13,500$) and NIH test ($N=22,744$) sets, used exclusively as the reference for the NIH sanity gate check.

### 2.3 Benchmark Validation Split (`valid.csv`)
- **Source**: `valid.csv` from CheXpert.
- **Protocol Role**: Descriptive only, reported separately as an external benchmark reference against literature-reported validation subsets. It is **not** used for any threshold selection, calibration fitting, or hyperparameter tuning.

### 2.4 Data-Loading Failures & Exclusion Accounting
- **Pre-Inference Cohort Measurement**: `CHEXPERT_COHORT_MEASURED.json` is generated and written to disk **before inference begins**, measuring the raw tabular frontal cohort demographics, projection distributions (AP vs. PA), patient clustering, and label policy counts.
- **Exclusion Logging**: Any image file that fails to decode or load (e.g., corrupted file, I/O read failure, unreadable image data) is caught, logged with its full filepath and patient ID, and excluded from metric computation. All such unreadable images are written to a separate `EXCLUSIONS.json` manifest (recording relative path, patient ID, failure reason, and total exclusion count) on disk **before any evaluation metric or contrast is computed**.
- **Exclusion Threshold Sensitivity Guard**: If more than 0.1% of CheXpert images fail to load, the primary analysis is repeated with all patients who have any failed image removed, and both results are reported.

---

## 3. Pre-Specified Pneumonia Label Mapping Policies

In the CheXpert labeling pipeline (Irvin et al., 2019), radiological reports are parsed into positive ($1.0$), uncertain ($-1.0$), negative ($0.0$), and unmentioned/blank ($\text{NaN}$). The empirical radiograph distribution measured on the 191,027 frontal images is:
- **Positive ($1.0$)**: 4,675 images
- **Uncertain ($-1.0$)**: 15,981 images
- **Negative ($0.0$)**: 1,875 images
- **Blank / Missing ($\text{NaN}$)**: 168,496 images (88.20% unmentioned)

For every label policy, the evaluation reports both the number of positive images ($N_{\text{pos}}$) and unique positive patients ($N_{\text{pos, patients}}$), as well as both image-level and patient-level prevalences, as measured at runtime and recorded in `CHEXPERT_COHORT_MEASURED.json`.

The four mutually distinct label policies are defined as follows:

### 3.1 Primary Labeling Policy (`u_zeros`)
- **Mapping**: Blank ($\text{NaN}$) $\rightarrow 0.0$; Uncertain ($-1.0$) $\rightarrow 0.0$; Positive ($1.0$) $\rightarrow 1.0$.
- **Clinical Rationale**: Unmentioned findings in narrative reports denote absence of pneumonia in standard radiology workflows; uncertain opacities reflect equivocal findings not confirmed as pneumonia.
- **Cohort Size**: $N = 191,027$ images ($64,534$ patients).
- **Positive Cases**: $N_{\text{pos}} = 4,675$ images; positive patients ($N_{\text{pos, patients}}$) recorded in `CHEXPERT_COHORT_MEASURED.json`.
- **Prevalence**: Image prevalence **2.447%** ($\approx 2.45\%$).

### 3.2 Sensitivity Policy 1 (`u_ones`)
- **Mapping**: Uncertain ($-1.0$) $\rightarrow 1.0$; Blank ($\text{NaN}$) $\rightarrow 0.0$; Positive ($1.0$) $\rightarrow 1.0$.
- **Clinical Rationale**: Tests model performance under maximal sensitivity where all equivocal opacities are treated as presumptive pneumonia.
- **Cohort Size**: $N = 191,027$ images ($64,534$ patients).
- **Positive Cases**: $N_{\text{pos}} = 20,656$ images ($4,675 + 15,981$); positive patients ($N_{\text{pos, patients}}$) recorded in `CHEXPERT_COHORT_MEASURED.json`.
- **Prevalence**: Image prevalence **10.813%** ($\approx 10.81\%$).

### 3.3 Sensitivity Policy 2 (`u_excluded`, Blank Kept Negative)
- **Mapping**: Uncertain ($-1.0$) excluded entirely ($15,981$ images dropped); Blank ($\text{NaN}$) kept as negative ($0.0$).
- **Clinical Rationale**: Eliminates equivocal report noise while preserving the true natural negative denominator.
- **Cohort Size**: $N = 175,046$ images ($191,027 - 15,981$).
- **Positive Cases**: $N_{\text{pos}} = 4,675$ images; positive patients ($N_{\text{pos, patients}}$) recorded in `CHEXPERT_COHORT_MEASURED.json`.
- **Prevalence**: Image prevalence **2.671%** ($\approx 2.67\%$).

### 3.4 Sensitivity Policy 3 (`explicitly_labeled_only`)
- **Mapping**: Excludes both Uncertain ($-1.0$) and Blank ($\text{NaN}$).
- **Clinical Rationale**: Isolates radiographs carrying explicit positive ($1.0$) or negative ($0.0$) mentions in reports ($4,675$ positive, $1,875$ negative).
- **Cohort Size**: $N = 6,550$ images.
- **Positive Cases**: $N_{\text{pos}} = 4,675$ images; positive patients ($N_{\text{pos, patients}}$) recorded in `CHEXPERT_COHORT_MEASURED.json`.
- **Prevalence**: Image prevalence **71.374%** ($\approx 71.37\%$).
- **Methodological Status**: Analyzed strictly as an artificially high-prevalence sensitivity check, distinct from natural clinical prevalence.

---

## 4. Primary Endpoints, Test Statistics & Multiplicity Control

The confirmatory evaluation tests exactly **one primary test statistic per endpoint** under the primary labeling policy (`u_zeros`). All primary tests are two-sided neutral estimands without directional bias.

### 4.1 Primary Endpoints & Neutral Estimands
1. **Primary Endpoint 1: Discrimination Generalization Gap ($\Delta\text{AUROC}$)**
   $$\Delta\text{AUROC} = \text{AUROC}_{\text{NIH, test}} - \text{AUROC}_{\text{CheXpert}}$$
   - **Estimand**: Difference in Area Under the Receiver Operating Characteristic between NIH test ($N=22,744$) and CheXpert frontal ($N=191,027$).
   - **Hypothesis**: $H_0: \Delta\text{AUROC} = 0$ versus $H_1: \Delta\text{AUROC} \neq 0$ (two-sided).
2. **Primary Endpoint 2: Uncalibrated Calibration Slope Change ($\Delta\text{Slope}$)**
   $$\Delta\text{Slope} = \text{Slope}_{\text{NIH, test}} - \text{Slope}_{\text{CheXpert}}$$
   - **Estimand**: Difference in calibration slope estimated via unpenalized logistic regression of true binary labels $y \in \{0, 1\}$ directly on raw model logits $z$ ($\text{logit}(P(Y=1)) = a + b \cdot z$) using `statsmodels.api.GLM(..., family=sm.families.Binomial())` or `sklearn.linear_model.LogisticRegression(penalty=None)` between NIH test and CheXpert (uncalibrated probabilities).
   - **Hypothesis**: $H_0: \Delta\text{Slope} = 0$ versus $H_1: \Delta\text{Slope} \neq 0$ (two-sided).
3. **Primary Endpoint 3: Decision Operating Sensitivity Gap ($\Delta\text{Sensitivity}$ at $\tau_{\text{sens90}}$)**
   $$\Delta\text{Sensitivity} = \text{Sensitivity}_{\text{NIH, test}}(\tau_{\text{sens90}}) - \text{Sensitivity}_{\text{CheXpert}}(\tau_{\text{sens90}})$$
   - **Estimand**: Change in clinical sensitivity evaluated at the fixed source-derived threshold $\tau_{\text{sens90}}$ (the largest threshold achieving $\ge 90\%$ sensitivity on NIH validation).
   - **Hypothesis**: $H_0: \Delta\text{Sensitivity} = 0$ versus $H_1: \Delta\text{Sensitivity} \neq 0$ (two-sided).

**Primary Test Statistics & Multi-Seed Inference**:
For each architecture (DenseNet-121 and EfficientNet-B0), the test statistic for each primary endpoint is the **three-seed mean difference**:
$$\Delta\bar{\theta} = \frac{1}{3} \sum_{s \in \{42, 1337, 2024\}} \Delta\theta_s$$
Two-sided empirical $p$-values are computed from the paired bootstrap replicates of that three-seed mean ($\Delta\bar{\theta}^{*(b)}$ for $b = 1, \dots, 2000$).

### 4.2 Multiplicity Control Family
The confirmatory family comprises exactly **6 primary hypothesis tests**:
$$\text{Family Size} = 3 \text{ endpoints} \times 2 \text{ architectures (DenseNet-121, EfficientNet-B0)} = 6 \text{ tests}$$
Multiplicity across this 6-test family is controlled using the Benjamini-Hochberg procedure at False Discovery Rate $q = 0.05$.

### 4.3 Secondary & Exploratory Status
All other contrasts and metrics are designated strictly as **secondary** or **exploratory**:
- Area Under the Precision-Recall Curve (AUPRC)
- Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
- Specificity contrasts ($\Delta\text{Specificity}$)
- Youden's $J$ thresholding ($\tau_{\text{Youden}}$) and High-Sensitivity 95% thresholding ($\tau_{\text{sens95}}$)
- Expected Calibration Error ($\text{ECE}_{\text{uniform}}$ and $\text{ECE}_{\text{adaptive}}$)
- Brier score and Brier Skill Score ($\text{BSS}$)
- Post-hoc recalibration models (temperature scaling, Platt scaling, target-10% Platt)
- Demographic and acquisition projection subgroup breakdowns

### 4.4 Outputs and Precision Specification
- **Multi-Label Classification Head**: All model architectures output 8 classification logits corresponding to `SHARED_CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax", "No Finding"]`.
- **Pneumonia Column Index**: Pneumonia is positioned at **index 5** of the 8-label head (`PNEU_COL_IDX = 5`). All discrimination, calibration, and operating threshold metrics are computed strictly using logits column 5 ($z_5$).
- **Inference Hardware & Precision**: Inference runs on a CUDA GPU only; the notebook aborts if no CUDA device is available (outside smoke mode). Forward passes are executed under mixed-precision fp16 autocast (`torch.amp.autocast(device.type, dtype=torch.float16)`).
- **DataLoader Batch Size**: The DataLoader batch size is fixed to **64** on GPU.
- **Ground-Truth Target**: CheXpert ground-truth labels for pneumonia are extracted directly from the tabular `Pneumonia` column in `train.csv`.

---

## 5. Statistical Inference & Resampling Framework

### 5.1 Patient-Clustered Bootstrap Resampling
- **Clustering Unit**: Unique patient identifier (`patient_id`). All radiographs belonging to a selected patient are sampled together as an atomic cluster.
  - **CheXpert Cluster ID**: For CheXpert, `patient_id` is parsed directly from the `Path` column in `train.csv` (extracting the `"patientNNNNN"` directory component; implemented in `notebooks/kaggle_infer_old_checkpoints.ipynb`, Cell 8, lines 26–27).
  - **NIH Cluster ID**: For NIH ChestX-ray14, `patient_id` is extracted directly from the `Patient ID` column of the split CSV (`data/splits/nih/nih_val.csv` and `data/splits/nih/nih_test.csv`).
- **Replications**: $B = 2,000$ bootstrap iterations exactly.
- **Random Seed**: Fixed random seed (seed 42) for deterministic reproducibility.

### 5.2 Independent Resampling for Cross-Domain Comparisons
- For cross-cohort comparisons ($\Delta\text{AUROC}$, $\Delta\text{Slope}$, and $\Delta\text{Sensitivity}$), patient clusters are resampled **independently with replacement** from the NIH test cohort and **independently with replacement** from the CheXpert cohort at each iteration $b \in \{1, \dots, B\}$.
- The test statistic $\Delta\theta^{*(b)} = \theta_{\text{NIH, test}}^{*(b)} - \theta_{\text{CheXpert}}^{*(b)}$ is computed on these independently resampled cohorts.

### 5.3 Paired Resampling Across Model Seeds
- For multi-seed models (seeds 42, 1337, 2024), the **exact same bootstrap resamples** (identical sequence of patient cluster indices at each iteration $b$) are applied across all model seeds to eliminate sampling discrepancy across seeds.

### 5.4 Two-Sided Bootstrap p-Value Formula
For a test statistic $\Delta\theta$ with point estimate $\Delta\hat{\theta}$ and bootstrap replications $\Delta\theta^{*(b)}$ ($b=1,\dots,B$), the two-sided bootstrap p-value under $H_0: \Delta\theta = 0$ is defined using pseudocount continuity correction:
$$p = \min\left(1.0, \; 2 \cdot \min\left( \frac{1 + \sum_{b=1}^B \mathbb{I}(\Delta\theta^{*(b)} \le 0)}{B + 1}, \; \frac{1 + \sum_{b=1}^B \mathbb{I}(\Delta\theta^{*(b)} \ge 0)}{B + 1} \right) \right)$$
where $\mathbb{I}(\cdot)$ is the indicator function.

### 5.5 Joint Estimates & Dispersion Reporting
- **Per-Seed Results**: Individual results for each of the three seeds (42, 1337, 2024) are reported explicitly in all result tables with their 95% bootstrap confidence intervals ($2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles).
- **Joint Estimate & Percentile Interval**: A joint pooled point estimate is reported as the arithmetic mean across the three seeds:
  $$\bar{\theta} = \frac{1}{K} \sum_{k=1}^K \hat{\theta}_k \quad (K=3)$$
  The 95% confidence interval of the joint estimate is the $[2.5^{\text{th}}, 97.5^{\text{th}}]$ percentile interval of the three-seed mean across paired resamples:
  $$\bar{\theta}^{*(b)} = \frac{1}{K} \sum_{k=1}^K \theta_k^{*(b)} \quad (b=1,\dots,B)$$
- **Between-Seed Standard Deviation ($ddof = 1$)**: Reported separately to quantify algorithmic/training initialization variance:
  $$s_{\text{seed}} = \sqrt{\frac{1}{K - 1} \sum_{k=1}^K (\hat{\theta}_k - \bar{\theta})^2} \quad (K=3)$$
  This is explicitly distinguished from test-set bootstrap finite sampling uncertainty.

### 5.6 DeLong Sensitivity Analysis
- DeLong's non-parametric asymptotic test for comparing ROC curves is reported strictly as a secondary sensitivity analysis, explicitly accompanied by the caveat that it assumes independent observations and ignores patient-level clustering.

---

## 6. Models Under Evaluation

### 6.1 Checkpoint Lineage & Cryptographic Manifest Verification
The models under evaluation consist of the 6 historical models verified against `configs/checkpoint_manifest.json`. The table below is generated programmatically from `configs/checkpoint_manifest.json`:

### Checkpoint Manifest Provenance
- **Manifest File**: `configs/checkpoint_manifest.json`
- **Manifest SHA-256 (LF-normalized)**: `4ae0b9fc751c7453c1308ee5518ae4100ae018fbf922daa269d46ccf92cfbc1f`

| Model Key | Architecture | Seed | Checkpoint Path | SHA-256 Digest |
|:---|:---|:---:|:---|:---|
| `densenet121_42` | densenet121 | 42 | `results/runs/nih_densenet121/checkpoints/seed_42/checkpoint_best.pt` | `6e77c6a6cc3e10cb5150924dd81ff74c5eb44359466ae2d3f56519c875d23a59` |
| `densenet121_1337` | densenet121 | 1337 | `results/runs/nih_densenet121/checkpoints/seed_1337/checkpoint_best.pt` | `c3f6afccac87254c980a88ea72c71fb13631364483401cbed5bc8b2052b946a0` |
| `densenet121_2024` | densenet121 | 2024 | `results/runs/nih_densenet121/checkpoints/seed_2024/checkpoint_best.pt` | `494dd05fd54cd0effdc3393534d99d6a821b2ff93f63bf0c4841493eb0c851c3` |
| `efficientnet_b0_42` | efficientnet_b0 | 42 | `results/runs/nih_efficientnet_b0/checkpoints/seed_42/checkpoint_best.pt` | `dc2b37375b6909b814830dd57eb384765f5c18ed8fdc34ced349fa48872458bc` |
| `efficientnet_b0_1337` | efficientnet_b0 | 1337 | `results/runs/nih_efficientnet_b0/checkpoints/seed_1337/checkpoint_best.pt` | `7772e387d3d2b4000825da5e70f858c09ce81862340d1f6199b0d0aad7f958b9` |
| `efficientnet_b0_2024` | efficientnet_b0 | 2024 | `results/runs/nih_efficientnet_b0/checkpoints/seed_2024/checkpoint_best.pt` | `8d8b6f80892e9854001ac24bfadea81f25703c5d617ea83c24eab9935788815c` |

---

## 7. Post-Hoc Recalibration Protocols

### 7.1 Source-Fitted Temperature Scaling
- Scalar parameter $T > 0$ fitted via negative log-likelihood minimization on NIH validation logits ($N=13,500$):
  $$\hat{p} = \sigma(z / T)$$
- Applied directly to CheXpert logits without parameter updates.

### 7.2 Source-Fitted Platt Scaling
- Two-parameter affine logistic regression fitted via unpenalized maximum likelihood on NIH validation logits ($N=13,500$):
  $$\hat{p} = \sigma(a \cdot z + b)$$
- Applied directly to CheXpert logits without parameter updates.

### 7.3 Target-10% Platt Recalibration (Institutional Adaptation Benchmark)
- Evaluates the effect of few-shot target institutional calibration:
  1. A patient-level 10% partition of CheXpert (partitioned by unique `patient_id` to prevent cluster leakage) is used to fit Platt scaling parameters $(a, b)$.
  2. Fitted parameters are evaluated on the held-out remaining 90% patient cohort of CheXpert.
  3. Repeated across **20 independent random patient-level partitions** (with fixed, recorded random seeds).
  4. The mean and standard deviation of calibration metrics across the 20 iterations are reported.

---

## 8. Calibration & Diagnostic Performance Metrics

### 8.1 Fixed Bin Count
The bin count for all calibration calculations is fixed to **$M = 15$ bins** as the pre-specified choice (`src/pneumoshift/evaluation/calibration.py`).

### 8.2 Calibration Metrics Suite
1. **Equal-Width Expected Calibration Error ($\text{ECE}_{\text{uniform}}$)**:
   $$\text{ECE}_{\text{uniform}} = \sum_{m=1}^{15} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   Partitioned into 15 uniform-width probability intervals over $[0, 1]$.
2. **Equal-Mass (Adaptive) Expected Calibration Error ($\text{ECE}_{\text{adaptive}}$)**:
   $$\text{ECE}_{\text{adaptive}} = \sum_{m=1}^{15} \frac{1}{15} \left| \text{acc}(Q_m) - \text{conf}(Q_m) \right|$$
   Partitioned into 15 quantile intervals each containing an equal number of test samples ($N / 15$).
3. **Brier Score ($\text{BS}$)**:
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
4. **Brier Skill Score ($\text{BSS}$)**:
   $$\text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{ref}}}$$
   where $\text{BS}_{\text{ref}} = \bar{y}(1 - \bar{y})$ is the Brier score of a naive reference model predicting the target cohort base rate / prevalence $\bar{y}$.
5. **Negative Log-Likelihood ($\text{NLL}$)**:
   $$\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right]$$
6. **Calibration Slope ($b$) and Intercept ($a$)**:
   Estimated via unpenalized logistic regression of true binary labels $y \in \{0, 1\}$ directly on raw model logits $z$:
   $$\text{logit}(P(Y=1)) = a + b \cdot z$$
   Estimated using `statsmodels.api.GLM` with binomial family and logit link (or unpenalized `sklearn.linear_model.LogisticRegression(penalty=None)`). Perfect calibration corresponds to $a = 0$ (intercept) and $b = 1$ (slope).

---

## 9. Clinical Decision Thresholds

### 9.1 Decision Rule
Predictions are mapped to binary triage classifications via the standard decision rule:
$$\hat{y} = \begin{cases} 1 & \text{if } p \ge \tau \\ 0 & \text{if } p < \tau \end{cases}$$
("positive if $p \ge \tau$").

### 9.2 Threshold Derivation
Operating thresholds and all in-domain vs. cross-domain comparative analyses are derived strictly from the fresh NIH validation and NIH test outputs produced during the `oldckpt_real_v1` run. See section 2.2 (Execution Run Origin).
- **$\tau_{\text{Youden}}$**: The threshold maximizing Youden's $J = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1$ on fresh NIH validation outputs. If multiple thresholds achieve the maximum $J$, the **highest threshold** is selected as the tie-break:
  $$\tau_{\text{Youden}} = \max \left\{ \arg\max_\tau (\text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1) \right\}$$
- **$\tau_{\text{sens90}}$**: The largest threshold on fresh NIH validation outputs achieving sensitivity $\ge 90.0\%$:
  $$\tau_{\text{sens90}} = \max \{ \tau : \text{Sensitivity}(\tau) \ge 0.90 \}$$
- **$\tau_{\text{sens95}}$**: The largest threshold on fresh NIH validation outputs achieving sensitivity $\ge 95.0\%$:
  $$\tau_{\text{sens95}} = \max \{ \tau : \text{Sensitivity}(\tau) \ge 0.95 \}$$

### 9.3 Per-Seed Threshold Selection
All thresholds ($\tau_{\text{Youden}}$, $\tau_{\text{sens90}}$, $\tau_{\text{sens95}}$) are derived individually per seed on that specific seed's fresh NIH validation predictions.

### 9.4 Status of Max-F1
The maximum F1 threshold ($\tau_{\text{max-F1}}$) is dropped from primary and secondary decision analysis. Because F1 ($2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$) depends strongly on both baseline prevalence and the chosen operating point, optimal F1 cutoffs do not transfer invariantly across populations with different base rates or sampling schemes. Any F1 reporting is designated strictly as exploratory, accompanied by this caveat.

---

## 10. Pre-Stated Methodological Limitations

1. **Different Labelers & NLP Ground-Truth Pipelines**:
   NIH ChestX-ray14 ground-truth labels were generated using rule-based NLP extraction from clinical radiology reports from the NIH Clinical Center (Wang et al., 2017). CheXpert labels were extracted using an automated rule-based labeler on Stanford Hospital reports (Irvin et al., 2019). Institutional differences in reporting style, phraseology, and parsing rules introduce unavoidable label noise between datasets.
2. **High Proportion of Blank Pneumonia Labels (88.2%)**:
   In CheXpert, 168,496 out of 191,027 frontal images (88.20%) have an unmentioned (blank/NaN) pneumonia label. Under the primary `u_zeros` policy, all unmentioned findings are mapped to negative ($0.0$). If occult pneumonia was present but unrecorded in a small fraction of reports, negative label noise is introduced.
3. **Image Preprocessing, Grayscale Expansion & Aspect-Ratio Trimming**:
   - **Grayscale-to-3-Channel Handling**: As defined in `src/pneumoshift/data/datasets.py` line 116 (`image = Image.open(img_path).convert("RGB")`), single-channel grayscale radiographs in both NIH ChestX-ray14 and CheXpert are converted to 3 identical channels ($L \to (L, L, L)$) via PIL's `.convert("RGB")` prior to downstream tensor transformation and ImageNet normalization.
   - **Resolution and Geometric Trimming**: CheXpert image dimensions are measured at runtime (width and height distributions) and written to `CHEXPERT_COHORT_MEASURED.json`. During inference, both NIH and CheXpert images pass through the exact evaluation transform defined in `src/pneumoshift/data/transforms.py`:
     `transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])`
     While NIH images are natively square ($1024 \times 1024$), CheXpert images possess variable, non-square aspect ratios. When passing through `Resize(256)` followed by `CenterCrop(224)`, `Resize(256)` scales the shorter edge to 256 pixels preserving aspect ratio, and `CenterCrop(224)` crops the centered $224 \times 224$ square window. Consequently, `CenterCrop(224)` trims substantially more anatomical margin along the longer dimension for non-square CheXpert radiographs than for natively square NIH radiographs.
4. **Execution Prerequisite Gate**:
   CheXpert forward inference is blocked by code assertions until `NIH_GATE_PASSED.json` is generated on disk and verified, proving reproduction within numerical tolerance ($\Delta\text{AUROC} \le 10^{-4}$, Spearman $\rho \ge 0.999$) of historical NIH baseline performance on both NIH validation and test sets.
5. **Threshold-Selection Uncertainty Not Propagated**:
   Threshold-selection uncertainty is not propagated. Operating thresholds are fixed per seed from NIH validation (139 positive images in 110 patients) and held fixed within every bootstrap replicate; the sampling uncertainty of the selected thresholds is not included in confidence intervals for threshold-dependent endpoints (Endpoint 3 and secondary operating-point metrics).

---

## 11. Exploratory Subgroup Stratifications

Subgroup evaluations are designated strictly as **exploratory** (hypothesis-generating):
1. **Acquisition Projection**: Anteroposterior (AP) vs. Posteroanterior (PA). Counts and subgroup metrics measured at runtime and written to `CHEXPERT_COHORT_MEASURED.json`.
2. **Patient Sex**: Female vs. Male (measured at runtime).
3. **Patient Age Bands**: `<40`, `40-64`, and `\ge 65` years (measured at runtime).

---

## 12. Protocol Immutability & Deviation Protocol

### 12.1 Deviation Protocol
Any post-hoc addition, modification, or analytical departure from this pre-specified plan after experimental inference results exist must be formally recorded in `DEVIATIONS.md` prior to dissemination.

Each deviation record must state:
1. **Date**: Timestamp of the modification.
2. **Reason**: Clinical, mathematical, or computational rationale for the change.
3. **Original Analysis**: The exact pre-specified protocol specification.
4. **Changed Analysis**: The amended protocol specification.

Both the pre-specified and amended analyses must be reported side-by-side in publication manuscripts to maintain complete scientific integrity.

### 12.2 Provenance Stamping
All output prediction arrays will be hash-stamped with:
- `data_source = 'real_inference'`
- `run_id = 'oldckpt_real_v1'`
- `analysis_plan_sha256`: The runtime SHA-256 hash of this `ANALYSIS_PLAN_TRANSFER.md` file (normalized to LF line endings).

---

## 13. Version History
- **v1.0.0** (Tagged as `prov-hardening-v2`, Commit `add6167`): Initial locked pre-specification protocol prior to cross-domain execution.
- **v2.1.0**: Revision incorporating neutral estimands, two-sided hypotheses, explicit bootstrap $p$-value formula, 6-test multiplicity family (3 endpoints $\times$ 2 architectures) with Benjamini-Hochberg FDR control, unpenalized logistic regression for calibration slope, highest-threshold Youden tie-breaking, corrected image-level prevalence comparisons, runtime image dimension specification, and `valid.csv` descriptive designation.
- **v2.2.0**: Comprehensive specification update incorporating:
  - Programmatically generated checkpoint manifest table and manifest SHA-256 from `configs/checkpoint_manifest.json` (eliminating manual hash transcription).
  - Explicit specification of logistic regression on raw logits $z$ (removing `logit(p)`).
  - Continuity-corrected bootstrap $p$-value formula with $(B+1)$ pseudocount denominator.
  - Locked bootstrap parameters: $B = 2,000$ iterations exactly with fixed seed 42.
  - Three-seed mean paired bootstrap replicates per architecture as primary test statistics.
  - Specification that NIH-test metrics in cross-domain contrasts originate from the same `oldckpt_real_v1` inference run as CheXpert (stored predictions reserved strictly for the NIH gate).
  - Explicit data-loading failure accounting with exclusions logged to `CHEXPERT_COHORT_MEASURED.json`.
  - PIL `.convert("RGB")` grayscale handling documentation citing `src/pneumoshift/data/datasets.py:116` and non-square aspect ratio trimming analysis.
  - Confirmation of $M = 15$ bins as the pre-specified calibration choice.
  - Inclusion of AUPRC, PPV, and NPV in secondary evaluation metrics.
  - Line ending normalization via `.gitattributes` (`text eol=lf`) and CRLF-to-LF stripping in hash calculations.
- **v2.2.1**: Protocol revision removing git tag references from checkpoint manifest specification and computing manifest SHA-256 on LF-normalized bytes; specifying fresh NIH validation and test outputs of the `oldckpt_real_v1` run for threshold derivation and cross-domain contrasts; adding 'Outputs and precision' specification (8-label head index 5, fp16 autocast, GPU batch size 64, CheXpert `Pneumonia` column); separating unreadable image exclusions into `EXCLUSIONS.json` written before metric computation while `CHEXPERT_COHORT_MEASURED.json` is written before inference; and supporting `NIH_GATE_PASSED.json` resolution from attached prior Kaggle inputs (`/kaggle/input/...`).
- **v2.2.2**: Protocol revision renaming document to Pre-Specified Analysis Plan; clarifying execution origin across Kaggle sessions while requiring identical GPU model, PyTorch version, and git commit in prediction stamps; restricting full inference strictly to CUDA GPU; adding methodological limitation 5 explicitly noting that operating threshold selection uncertainty is not propagated across bootstrap replicates; formalizing gate token environment signatures and strict mismatch refusal; recording reference bundle raw SHA-256 digest (`026fb68e71d0cbb944f446a9eb643ef3d7e0c5fb413627c1a6a92a12a246feb8`) covering both validation and test sets with row alignment assertions; and adding `.gitattributes` binary tracking for `configs/*.npz`.

- **v2.2.3**: Protocol revision specifying cluster ID definitions (CheXpert parsed patient ID from Path column vs. NIH split CSV Patient ID column); replacing cryptographically stamped terminology with hash-stamped; replacing duplicate execution origin text in Section 9.2 with cross-reference to Section 2.2; adding exclusion threshold sensitivity guard (if >0.1% of CheXpert images fail to load, primary analysis is repeated with all patients having any failed image removed, reporting both); and updating lineage tag reference in the Kaggle execution notebook to prov-hardening-v3.

*Note on Execution Status*: At all version milestones (`v1.0.0`, `v2.1.0`, `v2.2.0`, `v2.2.1`, `v2.2.2`, and `v2.2.3`), **no CheXpert forward inference has been run**. The git commit hash and tag are hash-stamped in all subsequent prediction stamps.
