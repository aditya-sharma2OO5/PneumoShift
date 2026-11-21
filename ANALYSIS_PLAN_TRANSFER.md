# Pre-Registered Analysis Plan: Cross-Domain Transfer Evaluation (NIH to CheXpert)

**Document Version**: 1.0.0  
**Registration Date**: 2026-11-22  
**Commit Status**: Pre-Experimental Registration (Locked prior to CheXpert forward inference)  
**Target Repository**: `https://github.com/aditya-sharma2OO5/PneumoShift.git`  
**Execution Run ID**: `oldckpt_real_v1`

---

## 1. Study Overview & Objective
This pre-registered protocol specifies the evaluation methodology for assessing the out-of-distribution transferability of deep neural networks trained on the NIH ChestX-ray14 dataset when applied to the external CheXpert dataset. All models, model weights, decision thresholds, label mapping schemes, and primary metrics are locked prior to running CheXpert inference.

---

## 2. Target Evaluation Cohort

### 2.1 Primary Cohort (CheXpert Frontal Train)
- **Source**: `train.csv` from CheXpert (`willarevalo/chexpert-v10-small`).
- **View Selection**: Frontal views strictly (Anteroposterior [AP] and Posteroanterior [PA]). Lateral views (`Frontal/Lateral == "Lateral"`) are excluded.
- **Measured Empirical Size**:
  - **Total Frontal Radiographs ($N$)**: **191,027**
  - **Unique Patients**: **64,534**
  - **Images per Patient**: Mean **2.96** (range: 1 to 94)

### 2.2 Benchmark Validation Cohort
- **Source**: `valid.csv` from CheXpert ($N = 687$ frontal images from 200 patients).
- Evaluated and reported separately as an external benchmark reference against literature standard validation sets.

---

## 3. Unit of Statistical Analysis & Clustering

### 3.1 Patient-Clustered Resampling
Because patients contribute multiple radiographs (mean 2.96 images/patient across 64,534 patient clusters), standard independently sampled bootstrap underestimates variance.
- **Clustering Unit**: Unique patient (`patient_id` parsed from study path).
- **Statistical Method**: Patient-clustered non-parametric bootstrap resampling ($B \ge 2,000$ iterations, fixed random seed 42).
- **Point Estimates & Dispersion**:
  - Sample standard deviation ($ddof = 1$) reported for all multi-seed aggregations across random seeds (42, 1337, 2024).
  - All per-seed values reported explicitly in table rows.

---

## 4. Pneumonia Label Mapping Policies (Pre-Registered)

In the CheXpert labeling pipeline (Irvin et al., 2019), radiological reports are parsed into positive ($1.0$), uncertain ($-1.0$), negative ($0.0$), and unmentioned/blank ($\text{NaN}$). The empirical distribution on the 191,027 frontal images was measured prior to inference:
- **Positive ($1.0$)**: 4,675
- **Uncertain ($-1.0$)**: 15,981
- **Negative ($0.0$)**: 1,875
- **Blank / Missing ($\text{NaN}$)**: 168,496

To prevent post-hoc bias, the following four mutually distinct policies are locked:

### 4.1 Primary Labeling Policy (`u_zeros`)
- **Mapping**: Blank ($\text{NaN}$) $\rightarrow 0.0$; Uncertain ($-1.0$) $\rightarrow 0.0$; Positive ($1.0$) $\rightarrow 1.0$.
- **Clinical Rationale**: In clinical chest radiograph reporting, unmentioned findings indicate absence of clinical abnormality or suspicion; uncertain opacities reflect equivocal findings not confirmed as pneumonia.
- **Cohort Size**: $N = 191,027$
- **Positive Cases ($N_{pos}$)**: **4,675**
- **Prevalence**: **2.447%** ($\approx 2.45\%$)

### 4.2 Sensitivity Analysis 1 (`u_ones`)
- **Mapping**: Uncertain ($-1.0$) $\rightarrow 1.0$; Blank ($\text{NaN}$) $\rightarrow 0.0$; Positive ($1.0$) $\rightarrow 1.0$.
- **Clinical Rationale**: Tests model performance under maximum sensitivity where all suspicious/equivocal opacities are treated as presumptive pneumonia.
- **Cohort Size**: $N = 191,027$
- **Positive Cases ($N_{pos}$)**: **20,656** ($4,675 + 15,981$)
- **Prevalence**: **10.813%** ($\approx 10.81\%$)

### 4.3 Sensitivity Analysis 2 (`u_excluded`, Blank Kept Negative)
- **Mapping**: Uncertain ($-1.0$) excluded entirely ($15,981$ images dropped); Blank ($\text{NaN}$) kept as negative ($0.0$).
- **Clinical Rationale**: Isolates definite positive pneumonia cases against definite or presumed normal cases, removing equivocal label noise while maintaining natural negative representation.
- **Cohort Size**: $N = 175,046$ ($191,027 - 15,981$)
- **Positive Cases ($N_{pos}$)**: **4,675**
- **Prevalence**: **2.671%** ($\approx 2.67\%$)

### 4.4 Sensitivity Analysis 3 (`explicitly_labeled_only`)
- **Mapping**: Excludes both Uncertain ($-1.0$) and Blank ($\text{NaN}$).
- **Clinical Rationale**: Evaluates exclusively the subset of images carrying explicit human radiological mentions for pneumonia ($4,675$ positive, $1,875$ negative).
- **Cohort Size**: $N = 6,550$
- **Positive Cases ($N_{pos}$)**: **4,675**
- **Prevalence**: **71.374%** ($\approx 71.37\%$)
- **Pre-Registration Constraint**: Explicitly labeled and presented as an artificially high-prevalence subgroup, distinct from natural clinical prevalence.

---

## 5. Models, Architectures & Decision Thresholds

### 5.1 Evaluated Checkpoints
All 6 historical NIH checkpoints verified via SHA-256 against `configs/checkpoint_manifest.json`:
1. `densenet121_42`: SHA-256 `6e77c6a6cc...`
2. `densenet121_1337`: SHA-256 `c3f6afccac...`
3. `densenet121_2024`: SHA-256 `494dd05fd5...`
4. `efficientnet_b0_42`: SHA-256 `dc2b37375b...`
5. `efficientnet_b0_1337`: SHA-256 `7772e387d3...`
6. `efficientnet_b0_2024`: SHA-256 `8d8b6f8089...` (or bit-identical root duplicate `5a21c82a99...`)

### 5.2 Operating Threshold Transfer
Operating points are selected strictly on the source NIH validation cohort ($N=13,500$) and transferred without adjustment:
- $\tau_{\text{Youden}}$: Maximum Youden index ($J = \text{Sensitivity} + \text{Specificity} - 1$).
- $\tau_{\text{sens90}}$: Operating threshold achieving $\ge 90\%$ clinical sensitivity on NIH validation.
- $\tau_{\text{sens95}}$: Operating threshold achieving $\ge 95\%$ clinical sensitivity on NIH validation.
- **Transfer Metric**: Transfer Discordance $\Delta \text{Sens} = \text{Sens}_{\text{source}} - \text{Sens}_{\text{target}}$ and $\Delta \text{Spec} = \text{Spec}_{\text{source}} - \text{Spec}_{\text{target}}$.

---

## 6. Primary & Secondary Statistical Metrics

### 6.1 Discrimination
- **AUROC**: Area Under the Receiver Operating Characteristic (empirical discrete trapezoidal calculation via raw empirical coordinates).
- **AUPRC**: Area Under the Precision-Recall Curve (average precision score).

### 6.2 Probability Calibration
- **ECE**: Expected Calibration Error across 15 uniform probability bins (computed on non-clipped probabilities).
- **Brier Score**: Mean squared error between predicted probabilities and binary labels.
- **Calibration Slope & Intercept**: Estimated via univariate logistic regression of true binary labels on logit predictions: $\text{logit}(P(Y=1)) = a + b \cdot z$. Perfect calibration corresponds to $a = 0$ (intercept) and $b = 1$ (slope).

### 6.3 Diagnostic Operating Performance
- Clinical Sensitivity, Specificity, Positive Predictive Value (PPV), and Negative Predictive Value (NPV) at $\tau_{\text{Youden}}$, $\tau_{\text{sens90}}$, and $\tau_{\text{sens95}}$.

---

## 7. Exploratory Subgroup Analyses

The following subgroup stratifications are pre-designated strictly as **exploratory** to distinguish hypothesis-generating findings from primary confirmatory hypotheses:
1. **Acquisition Projection**: AP ($N \approx 138,400$) vs PA ($N \approx 52,600$).
2. **Patient Sex**: Female vs Male.
3. **Patient Age Deciles**: $<40$, $40-64$, $\ge 65$ years.

---

## 8. Protocol Immutability & Scientific Integrity
1. Under no circumstances will any label definition, decision threshold, or primary metric formula be altered after CheXpert inference arrays are produced.
2. In the event of model or inference failure, raw terminal outputs and exceptions will be logged without parameter relaxation.
3. All prediction arrays will be cryptographically stamped with `data_source = 'real_inference'`, `run_id = 'oldckpt_real_v1'`, and the git commit hash of this pre-registered plan.
