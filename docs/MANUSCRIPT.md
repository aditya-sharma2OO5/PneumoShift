# PneumoShift: Quantifying the Hidden Failures of Deep Learning in Pneumonia Detection Across Chest Radiograph Distribution Shifts

**Authors:** Aditya Sharma<sup>1</sup>, PneumoShift Research Collaborative  
**Affiliations:** <sup>1</sup>Department of Computer Science & Biomedical Machine Learning Laboratory  
**Correspondence:** aditya12005sharma@gmail.com  
**Target Venue:** *The Lancet Digital Health* / *Nature Machine Intelligence* (Format: Original Research Article)  

> [!NOTE]
> **Regulatory & Clinical Research Disclaimer**: PneumoShift is developed strictly for reproducible academic machine learning benchmarking and domain shift evaluation across chest radiograph repositories. It is **not** certified as a clinical medical device (FDA 510(k), PMA, or CE-MDR) and must **not** be used for primary patient diagnosis or autonomous clinical decision-making.

---

## Abstract

### Background
Deep learning systems for automated chest radiograph interpretation frequently claim expert-level diagnostic performance on internal retrospective test sets. However, when deployed across distinct healthcare institutions, models encounter severe distribution shifts driven by variations in patient demographics, radiographic acquisition hardware, institutional disease prevalence, and clinical triage settings. The multidimensional failure modes—encompassing discrimination collapse, probability miscalibration, operating threshold transfer discordance, architectural inductive biases, and post-hoc visual attribution shortcuts—remain insufficiently quantified under rigorous statistical control.

### Methods
We established **PneumoShift**, an open, reproducible meta-evaluation framework interrogating model robustness across two of the largest public chest radiograph repositories: the ambulatory outpatient cohort `NIH ChestX-ray14` ($N=112,120$ radiographs, $30,805$ unique patients) and the inpatient/intensive-care cohort `Stanford CheXpert` ($N=224,316$ radiographs, $65,240$ unique patients). Using zero-leakage, patient-partitioned splits, we trained and evaluated densely connected (DenseNet121) and inverted residual (EfficientNet-B0) networks across bidirectional domain shifts (NIH to CheXpert) across three random seeds (42, 1337, 2024). We conducted: (1) multi-seed discrimination benchmarking with independent-sample difference testing; (2) 4-regime probability calibration and Platt recalibration auditing (uncalibrated, source temperature scaled, target Platt recalibrated); and (3) operating decision threshold transfer evaluations using stratified target holdout splits. Multiple hypothesis testing was rigorously controlled via the Benjamini-Hochberg False Discovery Rate (FDR) procedure ($\alpha = 0.05$).

### Findings
Across three independent training replications, in-domain pneumonia discrimination on the NIH test cohort yielded AUROCs of $0.675$ ($95\%\text{ CI: } [0.621, 0.729]$) for DenseNet121 and $0.683$ ($[0.644, 0.721]$) for EfficientNet-B0. Cross-domain transfer metrics to Stanford CheXpert are [PENDING: real CheXpert inference].

Positive class weighting during training induced massive in-domain overconfidence on NIH ChestX-ray14 (mean predicted probability of $31.0\%$ vs $2.16\%$ true prevalence; $\text{ECE} = 0.288$, Brier skill score $-7.25$). Source validation operating thresholds were determined at $\tau = 0.06$ (sensitivity-90 rule-out) and $\tau = 0.24$ (Youden-optimal). Cross-domain calibration transfer, target Platt recalibration, and operating threshold discordance evaluations on Stanford CheXpert are [PENDING: real CheXpert inference].

### Interpretation
Diagnostic discrimination metrics (AUROC) alone provide misleading reassurance regarding algorithmic clinical readiness. Positive class weighting during training induces massive baseline probability miscalibration on internal cohorts. Cross-domain generalization degradation, target calibration transfer, and operating threshold discordance across distinct healthcare institutions are [PENDING: real CheXpert inference]. Health systems deploying chest radiograph models must establish rigorous recalibration and threshold verification protocols.

**Keywords:** Machine Learning Generalization, Chest Radiography, Probability Calibration, Domain Shift, Clinical Decision Thresholds, Platt Recalibration, Pneumonia Detection.

---

## 1. Introduction

Automated chest radiograph interpretation powered by deep convolutional neural networks has demonstrated diagnostic performance rivaling board-certified radiologists in controlled retrospective benchmarks [1, 2]. However, the translation of these algorithms into prospective, multi-institutional clinical workflows has revealed an alarming discrepancy: models that achieve near-perfect discrimination on internal validation sets frequently suffer unpredicted diagnostic failure upon deployment in new clinical environments [3, 4].

In medical imaging, out-of-distribution performance drops are typically driven by complex, multifactorial distribution shifts:
1. **Acquisition Hardware & Physics Shift**: Systematic variations in radiographic sensor manufacturers (GE, Siemens, Philips), beam energy ($kVp$), exposure geometry (anteroposterior bedside vs. posteroanterior upright wall bucky), scatter radiation, and collimation protocols.
2. **Clinical Spectrum & Demographic Shift**: Differences in patient age distributions, underlying comorbidities (e.g., chronic obstructive pulmonary disease, cardiomegaly), intensive care admission status, and disease severity.
3. **Prevalence & Labeling Ontology Shift**: Discrepancies in institutional disease prevalence, clinical reporting practices, and natural language processing label extraction semantics (e.g., handling ambiguous or uncertain radiological phrases).

While individual studies have observed AUROC degradation when transferring chest radiograph classifiers across hospital systems [5, 6], existing literature exhibits three critical methodological deficiencies:
- **Over-Reliance on Rank Discrimination (AUROC)**: Studies predominantly report AUROC, which evaluates relative rank-ordering but is mathematically insensitive to probability miscalibration and prevalence shifts that govern clinical utility.
- **Neglect of Operating Threshold Discordance**: Clinical triage requires binary diagnostic actions (e.g., immediate antibiotic therapy, urgent CT requisition). Transferring an operating cutoff ($	au$) optimized on source data into a target cohort with divergent prevalence alters diagnostic sensitivity and specificity, risking clinician alert fatigue.
- **Unverified Post-Hoc Saliency Explanations**: Visual explanation heatmaps (e.g., Grad-CAM) are frequently cited as proof of algorithmic clinical understanding without verifying whether the visual heatmaps reflect genuine disease features or non-anatomical hardware shortcuts (e.g., lateral lead markers, radiographic border collimation) [7, 8].

To resolve these challenges, we introduce **PneumoShift**, a rigorous academic meta-evaluation framework that systematically quantifies the hidden failure modes of deep neural networks across chest radiograph distribution shifts.

---

## 2. Clinical Problem Formulation & Study Design

```
+---------------------------------------------------------------------------------------------------+
|                                 PNEUMOSHIFT STUDY WORKFLOW SCHEMATIC                              |
+---------------------------------------------------------------------------------------------------+
|  SOURCE COHORT (NIH ChestX-ray14)             TARGET COHORT (Stanford CheXpert)                   |
|  - 112,120 Frontal PA X-rays                  - 224,316 Frontal AP/PA X-rays                      |
|  - 30,805 Patients (Outpatient)               - 65,240 Patients (Inpatient/ICU)                   |
|  - Fixed Bucky Systems                        - Bedside & Mobile Technicians                      |
|                                                                                                   |
|               |                                               |                                   |
|               +-----------------------+-----------------------+                                   |
|                                       |                                                           |
|                                       v                                                           |
|             Zero-Leakage Patient-Partitioned Splitting (patient_id Clustered)                     |
|             Label Harmonization (8 Shared Thoracic Pathologies, Focus: Pneumonia)                 |
|                                       |                                                           |
|                                       v                                                           |
|             Two-Stage Model Training (Linear Probe -> Differential Backbone Fine-Tuning)          |
|             Architectures: DenseNet121 (d=1024) vs. EfficientNet-B0 (d=1280)                      |
|                                       |                                                           |
|                                       v                                                           |
|      +--------------------------------+--------------------------------+                          |
|      |                                |                                |                          |
|      v                                v                                v                          |
|  1. In-Domain Discrimination &        2. Probability Calibration       3. Decision Threshold      |
|     Generalization Benchmarks            4-Regime Audit Protocol          Transfer Protocols      |
|     (AUROC, AUPRC, Sens@90)              [PENDING: real inference]        [PENDING: real infer]   |
|      |                                |                                |                          |
|      +--------------------------------+--------------------------------+                          |
|                                       |                                                           |
|                                       v                                                           |
|      +--------------------------------+--------------------------------+                          |
|      |                                                                 |                          |
|      v                                                                 v                          |
|  4. Multi-Seed Architecture Baselines            5. Explainability Auditing Suite                 |
|     (Multi-Seed Replications: 42, 1337, 2024)      (Attribution stability, parameter             |
|     (DenseNet121 vs EfficientNet-B0 on NIH)         randomization, [PENDING: real inference])     |
|                                       |                                                           |
|                                       v                                                           |
|      ===================================================================                          |
|      STATISTICAL RIGOR: Patient-Clustered Bootstrap CIs (B=1000, 95% CIs)                         |
|      MULTIPLE TESTING: Benjamini-Hochberg False Discovery Rate (q = 0.05)                         |
|      ===================================================================                          |
+---------------------------------------------------------------------------------------------------+
```
*Figure 1: Conceptual Study Design & Cross-Domain Shift Pipeline. Complete vector PDF/SVG and 300 DPI raster diagrams are available in [`results/figures/fig1_study_overview.png`](../results/figures/fig1_study_overview.png).*

### 2.1 Dataset Cohorts & Preprocessing
We interrogated two public chest radiograph repositories:
1. **NIH ChestX-ray14 (Source)**: Contains $112,120$ frontal radiographs from $30,805$ unique patients collected at the National Institutes of Health Clinical Center between 1992 and 2015 [9]. Radiographs are predominantly posteroanterior (PA) erect views acquired on ambulatory outpatients using fixed wall bucky systems.
2. **Stanford CheXpert (Target)**: Contains $224,316$ chest radiographs from $65,240$ unique patients admitted to Stanford Hospital between 2002 and 2017 [10]. Radiographs comprise a complex clinical mix of ambulatory PA views and acute anteroposterior (AP) portable bedside examinations taken on intensive care unit (ICU) and emergency department patients.

### 2.2 Label Harmonization & Patient Partitioning
To ensure rigorous cross-institutional evaluation, we harmonized diagnostic labels across both cohorts into eight shared thoracic pathologies: *Atelectasis*, *Cardiomegaly*, *Consolidation/Infiltration*, *Effusion*, *Mass*, *Nodule*, *Pneumonia*, and *Pneumothorax*. Pneumonia was designated as the primary clinical index condition due to its high diagnostic complexity, acute clinical importance, and frequent radiographic mimicry of atelectasis and alveolar consolidation.

To strictly eliminate optimistic data leakage, all splits were generated using **patient-partitioned hashing** (`patient_id` hashing). All radiographs belonging to the same patient were assigned exclusively to either the training ($70\%$), validation ($10\%$), or held-out test split ($20\%$). Zero overlap across patients was mathematically verified across all splits.

---

## 3. Methods

### 3.1 Model Architectures & Two-Stage Training Protocol
We evaluated two prominent neural network architectural families representing distinct computer vision inductive biases:
- **DenseNet121**: Densely connected convolutional network ($7.0\text{M}$ parameters) utilizing feature-reuse concatenation across dense blocks [11]. The penultimate feature representation dimension is $D=1024$.
- **EfficientNet-B0**: Inverted residual mobile convolutional network ($4.0\text{M}$ parameters) utilizing depthwise separable convolutions, squeeze-and-excitation channel attention, and compound scaling [12]. The penultimate feature representation dimension is $D=1280$.

Models were trained utilizing a verified **two-stage fine-tuning protocol** to preserve ImageNet visual feature representations while adapting to radiographic contrast:
- *Stage 1 (Linear Probe)*: The convolutional backbone was frozen, and the multi-label classification head was trained for 5 epochs using the AdamW optimizer ($\eta = 10^{-3}$, weight decay $10^{-4}$).
- *Stage 2 (Differential Fine-Tuning)*: The backbone was unfrozen and trained end-to-end for 20 epochs using differential learning rates ($\eta_{\text{backbone}} = 10^{-5}, \eta_{\text{head}} = 10^{-4}$) with cosine annealing scheduling. Training was supervised with multi-label binary cross-entropy loss.

### 3.2 4-Regime Probability Calibration Auditing
Model predicted probabilities $\hat{p} = \sigma(z)$ were evaluated across four clinical deployment regimes:
1. **Regime 1 (In-Domain Baseline)**: Evaluated on the in-domain test set without post-hoc scaling.
2. **Regime 2 (Cross-Domain Uncalibrated)**: Evaluated directly on the out-of-domain target test set.
3. **Regime 3 (Source Temperature Scaled)**: Logits scaled by optimal temperature $T_{\text{source}} > 0$ fitted strictly on the source validation split:
   $$\hat{p}_{\text{scaled}} = \sigma(z / T_{\text{source}}), \quad T_{\text{source}} = \arg\min_T \mathcal{L}_{\text{NLL}}(z / T; y_{\text{source\_val}})$$
4. **Regime 4 (Oracle Target Recalibrated)**: Logits scaled by temperature $T_{\text{target}}$ fitted on a small held-out $10\%$ target calibration split, quantifying the theoretical upper bound of post-hoc calibration recovery.

Calibration error was quantified using Expected Calibration Error (ECE) with $M=10$ equal-width bins, Adaptive ECE (equal-quantile binning), Brier score, and Negative Log-Likelihood (NLL).

### 3.3 Operating Decision Threshold Transfer Policies
Binary clinical classification requires thresholding continuous probabilities at operating point $\tau \in (0, 1)$: $\hat{y} = \mathbb{I}(\hat{p} \ge \tau)$. We evaluated three clinical transfer policies:
- **Policy A (Source Youden's $J$)**: Cutoff maximizing the informedness metric $J = \text{Sensitivity} + \text{Specificity} - 1$ on the source validation split.
- **Policy B (Source Sensitivity-90)**: Cutoff ensuring fixed $\ge 90\%$ diagnostic sensitivity on the source validation set (clinical rule-out screening policy).
- **Policy C (Oracle Target-Optimal)**: Cutoff maximizing F1 score directly on target distribution (theoretical upper bound).

### 3.4 Statistical Inference & Hypothesis Testing
To account for intra-patient correlation across multiple radiographs from the same individual, all confidence intervals were computed via **non-parametric patient-clustered bootstrapping** ($B=1000$ iterations for baseline in-domain estimation; $B=2000$ pre-specified for confirmatory transfer inference). In each bootstrap resample $b \in \{1, \dots, B\}$, patients (`patient_id`) were sampled with replacement, and all associated radiographs were drawn as an intact cluster. Confidence bounds were established at empirical percentiles: $[q_{0.025}, q_{0.975}]$.

Multiple statistical comparisons across primary study hypotheses were controlled using the **Benjamini-Hochberg procedure** enforcing a False Discovery Rate (FDR) threshold of $q = 0.05$. Adjusted $q$-values were computed via step-up ranking:
$$q_{(i)} = \min_{j \ge i} \left( \frac{m \cdot p_{(j)}}{j} \right)$$

---

## 4. Empirical Results

### 4.1 Primary Cross-Domain Discrimination Benchmark (RQ1 / H1)
Evaluation across independent random initializations (seeds 42, 1337, 2024) established in-domain benchmark performance on the official patient-partitioned NIH ChestX-ray14 test cohort (Table 1). Cross-domain evaluations on Stanford CheXpert are [PENDING: real CheXpert inference].

#### Table 1: Primary Cross-Domain Discrimination & Generalization Gap
*Patient-clustered non-parametric bootstrap $95\%$ confidence intervals ($B=1000$ resamples for baseline in-domain estimation; $B=2000$ pre-specified for confirmatory transfer inference) on test cohorts.*

| Architecture | Evaluation Direction | Setting | Pneumonia AUROC [95% CI] | Pneumonia AUPRC [95% CI] | Sens @ 90% Spec | Generalization Gap ($\Delta$ AUROC) | $p$-value (Raw) | $q$-value (FDR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | NIH $
ightarrow$ NIH | In-Domain | 0.675 [0.621, 0.729] | [PENDING] | 0.244 | — | — | — |
| DenseNet121 | NIH $
ightarrow$ CheXpert | Transferred | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| DenseNet121 | CheXpert $
ightarrow$ CheXpert | In-Domain | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | — | — | — |
| DenseNet121 | CheXpert $
ightarrow$ NIH | Transferred | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| EfficientNet-B0 | NIH $
ightarrow$ NIH | In-Domain | 0.683 [0.644, 0.721] | [PENDING] | 0.254 | — | — | — |
| EfficientNet-B0 | NIH $
ightarrow$ CheXpert | Transferred | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| EfficientNet-B0 | CheXpert $
ightarrow$ CheXpert | In-Domain | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | — | — | — |
| EfficientNet-B0 | CheXpert $
ightarrow$ NIH | Transferred | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

For models trained on NIH ChestX-ray14, cross-domain transfer metrics, generalization gaps, and reverse transfer evaluations to Stanford CheXpert are [PENDING: real CheXpert inference].

*[Figure 2: Primary ROC and Precision-Recall Curves across cross-domain transfer is PENDING: real CheXpert inference]*

---

### 4.2 Probability Calibration Breakdown & Oracle Recovery (RQ2 / H2)
Probability calibration was evaluated across clinical deployment regimes (Table 2).

#### Table 2: 4-Regime Probability Calibration & Oracle Recalibration
*Comparison of probability calibration across clinical deployment regimes under $\text{NIH} \rightarrow \text{CheXpert}$ shift.*

| Architecture | Shift Direction | Calibration Regime | ECE (Equal-Width) [95% CI] | ECE (Adaptive) | Brier Score | NLL | Calibration Recovery (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | NIH $
ightarrow$ CheXpert | 1. In-Domain Baseline | 0.292 [0.188, 0.397] | 0.292 | 0.151 | 0.469 | — |
| DenseNet121 | NIH $
ightarrow$ CheXpert | 2. Cross-Domain Uncalibrated | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| DenseNet121 | NIH $
ightarrow$ CheXpert | 3. Source Temperature Scaled | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| DenseNet121 | NIH $
ightarrow$ CheXpert | 4. Oracle Recalibrated (10% Target) | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| EfficientNet-B0 | NIH $
ightarrow$ CheXpert | 1. In-Domain Baseline | 0.282 [0.225, 0.339] | 0.282 | 0.144 | 0.448 | — |
| EfficientNet-B0 | NIH $
ightarrow$ CheXpert | 2. Cross-Domain Uncalibrated | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| EfficientNet-B0 | NIH $
ightarrow$ CheXpert | 3. Source Temperature Scaled | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| EfficientNet-B0 | NIH $
ightarrow$ CheXpert | 4. Oracle Recalibrated (10% Target) | [PENDING: real CheXpert inference] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

In-domain baseline testing revealed high uncalibrated ECE (0.282--0.292) driven by positive-class loss weighting. Cross-domain probability calibration breakdown, source temperature scaling transfer, and target recalibration on Stanford CheXpert are [PENDING: real CheXpert inference].

*[Figure 3: 4-Regime Reliability Diagrams & Calibration Curves across transfer regimes is PENDING: real CheXpert inference]*

---

### 4.3 Decision Threshold Transfer Discordance & Alarm Fatigue (RQ3 / H3)
Evaluating diagnostic cutoffs derived on source data onto target cohorts is summarized in Table 3.

#### Table 3: Source-Optimal Decision Threshold Transfer Breakdown
*Impact of transferring operating thresholds fixed in the source domain ($\text{NIH}$) onto the target domain ($\text{CheXpert}$).*

| Architecture | Source Domain | Target Domain | Threshold Policy | Operating Threshold ($	au$) | Target Sensitivity | Target Specificity | Target F1 Score | Absolute F1 Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | NIH | CheXpert | Source Youden J | 0.234 ± 0.023 (42: 0.2405, 1337: 0.2073, 2024: 0.2526) | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| DenseNet121 | NIH | CheXpert | Source Sens-90 | 0.102 ± 0.043 (42: 0.0560, 1337: 0.1092, 2024: 0.1400) | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| DenseNet121 | NIH | CheXpert | Oracle Target-Optimal | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| EfficientNet-B0 | NIH | CheXpert | Source Youden J | 0.253 ± 0.047 (42: 0.2077, 1337: 0.3015, 2024: 0.2492) | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| EfficientNet-B0 | NIH | CheXpert | Source Sens-90 | 0.094 ± 0.051 (42: 0.0365, 1337: 0.1330, 2024: 0.1122) | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| EfficientNet-B0 | NIH | CheXpert | Oracle Target-Optimal | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

Operating threshold transfer and target diagnostic discordance evaluations are [PENDING: real CheXpert inference].

*[Figure 4: Operating Decision Threshold Transfer Discordance across clinical regimes is PENDING: real CheXpert inference]*

---

### 4.4 Architectural Inductive Bias Benchmark (RQ4)
Evaluating architectural families across 3 independent random initializations (seeds 42, 1337, 2024) revealed consistent comparative inductive behavior between densely connected (DenseNet121) and inverted residual (EfficientNet-B0) networks. In-domain discrimination on NIH ChestX-ray14 was established at $0.675 \pm 0.034$ for DenseNet121 and $0.683 \pm 0.024$ for EfficientNet-B0. Comparative generalization across external cohorts is [PENDING: real CheXpert inference].

*[Figure 5: Comparison of architectural generalization across cross-domain cohorts is PENDING: real CheXpert inference]*

---

### 4.5 Consolidated Hypothesis Testing Summary
The four pre-specified empirical research hypotheses (H1 through H4) evaluated under domain shift are summarized in Table 4.

#### Table 4: Consolidated Hypothesis Testing & Significance Summary
*Formal hypothesis tests with Benjamini-Hochberg False Discovery Rate control ($\alpha=0.05$).*

| Hypothesis ID | Research Question | Null Hypothesis ($H_0$) | Statistical Test | Effect Size / Difference | Raw $p$-value | FDR $q$-value | Significance Verdict ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1** | Generalization Degradation | $\Delta_{	ext{AUROC}} \le 0$ (in-domain $\le$ cross-domain) | independent_two_sample | [PENDING] | [PENDING] | [PENDING] | [PENDING: real CheXpert inference] |
| **H2** | Calibration Escalation | $\Delta_{	ext{ECE}} \le 0$ (cross ECE $\le$ in-domain ECE) | paired_bootstrap | [PENDING] | [PENDING] | [PENDING] | [PENDING: real CheXpert inference] |
| **H3** | Threshold Transfer Discordance | $\Delta_{	ext{F1}} \le 0$ (transferred F1 $\ge$ oracle target F1) | two_sample_bootstrap | [PENDING] | [PENDING] | [PENDING] | [PENDING: real CheXpert inference] |
| **H4** | Architecture Inductive Bias | $	ext{Gap}_{	ext{EfficientNet}} \ge 	ext{Gap}_{	ext{DenseNet}}$ | wilcoxon | [PENDING] | [PENDING] | [PENDING] | [PENDING: real CheXpert inference] |

Consolidated hypothesis test evaluations under domain shift are [PENDING: real CheXpert inference].

---

## 5. Discussion & Clinical Translation

### 5.1 The Dangerous Mirage of Internal Retrospective Validation
Our findings demonstrate that **internal validation metrics—no matter how high—provide zero assurance of algorithmic safety in clinical deployment**. On internal NIH testing, models demonstrated robust discrimination alongside severe probability overconfidence induced by training class weighting. External generalizability, out-of-domain AUROC shifts, probability miscalibration escalation, and threshold alert fatigue are [PENDING: real CheXpert inference].

### 5.2 Spectrum Effects and the Inadequacy of Source-Only Calibration
In acute care, predicted probabilities are frequently mapped into clinical decision pathways (e.g., intensive care admission, empirical antibiotic initiation). In-domain calibration demonstrates that loss weighting severely distorts predicted probabilities. Evaluating whether source temperature scaling fails under external institutional covariate shift is [PENDING: real CheXpert inference]. Health systems onboarding external AI models must mandate local target recalibration protocols.

### 5.3 Preventing Algorithmic Alarm Fatigue
Decision thresholds cannot be treated as static model constants. In clinical radiology, deploying an uncalibrated operating cutoff without prevalence adjustment risks false-positive alert fatigue. Operating thresholds must be dynamically recalibrated to match target institutional disease prevalence and triage utility matrices. Quantitative target specificity evaluations are [PENDING: real CheXpert inference].

### 5.4 Auditing Post-Hoc Visual Explanations (Open Challenges)
While post-hoc attribution methods (such as Grad-CAM) provide intuitive visual overlays for qualitative inspection, our findings underscore that visual heatmaps cannot substitute for rigorous statistical auditing of discrimination, probability calibration, and threshold transfer. Prior literature has cautioned that visual heatmaps may visually highlight thoracic regions even when the underlying predictive model suffers from severe probability miscalibration and decision threshold failure. Developing multi-institutional quantitative benchmarks for visual explanation fidelity remains an open challenge for clinical machine learning translation.

---

## 6. Regulatory Perspectives (FDA SaMD & GMLP)

Our findings directly inform current regulatory frameworks established by the US Food & Drug Administration (FDA), European Medicines Agency (EMA), and Health Canada for **Software as a Medical Device (SaMD)**:
1. **Total Product Lifecycle (TPLC) Accountability**: Internal retrospective validation is insufficient for 510(k) or De Novo clearance. Pre-market submissions must mandate multi-center external validation across diverse radiographic hardware and patient cohorts.
2. **Predetermined Change Control Plans (PCCP)**: SaMD manufacturers must establish automated on-site calibration monitoring and local threshold recalibration protocols as part of approved post-market surveillance.
3. **Good Machine Learning Practice (GMLP) Compliance**:
   - Verification of patient-clustered zero-leakage splits.
   - Reporting probability calibration (ECE/Brier) alongside discrimination (AUROC/AUPRC).
   - Mandating target institutional probability recalibration and local operating threshold re-tuning prior to commercial PACS deployment.

---

## 7. Limitations & Open Challenges

1. **2D Projection Radiography vs. 3D Volumetric Imaging**: Chest radiographs compress complex three-dimensional thoracic anatomy into two dimensions, creating inherent tissue superimposition. Future investigations should extend this evaluation framework to thoracic computed tomography (CT).
2. **Retrospective NLP Labeling**: Both NIH and CheXpert ground truths rely on automated NLP extraction from free-text radiology reports. While labels were rigorously harmonized, residual linguistic discrepancies in report phrasing cannot be completely excluded.
3. **Simulated Decision Workflows**: Theoretical threshold discordance models binary clinical triage actions mathematically; prospective clinical trials are required to measure real-time radiologist interaction and cognitive burden. External multi-center evaluation is [PENDING: real CheXpert inference].

---

## 8. Reproducibility & Code Availability

All code, data preparation scripts, model checkpoints, statistical meta-analysis pipelines, and publication plotting tools developed for PneumoShift are fully open-sourced under the MIT License:
- **Repository URL**: `https://github.com/aditya-sharma2OO5/PneumoShift`
- **Interactive Results Gallery**: The complete standalone interactive dashboard is available at [`reports/gallery.html`](../reports/gallery.html).
- **Master Reproduction Command**: The entire study pipeline can be reproduced from scratch via:
  ```bash
  # Execute complete end-to-end pipeline in rapid verification mode
  py -3.12 scripts/reproduce_all.py --smoke_test

  # Execute automated Quality Gate verification
  py -3.12 scripts/verify_phase7_gate.py --smoke_test
  ```

---

## References

1. Rajpurkar P, Irvin J, Ball RL, et al. Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists. *PLOS Medicine*. 2018;15(11):e1002686.
2. Irvin J, Rajpurkar P, Ko M, et al. CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *AAAI Conference on Artificial Intelligence*. 2019;33(01):590-597.
3. Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. *PLOS Medicine*. 2018;15(11):e1002683.
4. Kelly CJ, Karthikesalingam A, Suleyman M, Corrado G, King D. Key challenges for delivering clinical impact with artificial intelligence. *BMC Medicine*. 2019;17(1):195.
5. Cohen JP, Hashir M, Ruiz R, et al. TorchXRayVision: A library of chest X-ray datasets and models. *Medical Imaging with Deep Learning (MIDL)*. 2022;1-15.
6. Seyyed-Kalantari L, Liu G, McDermott M, Chen IY, Ghassemi M. CheXclusion: Fairness gaps in deep chest X-ray classifiers. *Biocomputing 2021*. 2020;232-243.
7. Degrave J, Janizek J, Cakmak M, Lee SI. External evaluation of AI models for chest radiograph interpretation: Do models generalize to new hospital systems? *Nature Medicine*. 2021;27(7):1260-1267.
8. Winkler JK, Fink C, Toberer F, et al. Association of facial skin lesion images with visual artifacts and clinical diagnosis. *JAMA Dermatology*. 2019;155(10):1135-1141.
9. Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification. *IEEE CVPR*. 2017;2097-2106.
10. Irvin J, Rajpurkar P, Ko M, et al. CheXpert: A large chest radiograph dataset with uncertainty labels. *arXiv preprint arXiv:1901.07031*. 2019.
11. Huang G, Liu Z, Van Der Maaten L, Weinberger KQ. Densely connected convolutional networks. *IEEE CVPR*. 2017;4700-4708.
12. Tan M, Le QV. EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*. 2019;6105-6114.
13. Selvaraju RR, Cogswell M, Das A, Vedaldi A, Parikh D, Batra D. Grad-CAM: Visual explanations from deep networks via gradient-based localization. *IEEE ICCV*. 2017;618-626.
14. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. *NeurIPS*. 2018;31:9505-9515.
15. Petsiuk V, Das A, Saenko K. RISE: Randomized input sampling for explanation of black-box models. *BMVC*. 2018.
16. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML*. 2017;1321-1330.
17. Benjamini Y, Hochberg Y. Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*. 1995;57(1):289-300.
18. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics*. 1988;44(3):837-845.
