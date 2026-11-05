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
We established **PneumoShift**, an open, reproducible meta-evaluation framework interrogating model robustness across two of the largest public chest radiograph repositories: the ambulatory outpatient cohort `NIH ChestX-ray14` ($N=112,120$ radiographs, $30,805$ unique patients) and the inpatient/intensive-care cohort `Stanford CheXpert` ($N=224,316$ radiographs, $65,240$ unique patients). Using zero-leakage, patient-partitioned splits, we trained and evaluated densely connected (DenseNet121) and inverted residual (EfficientNet-B0) networks across bidirectional domain shifts ($\text{NIH} \longleftrightarrow \text{CheXpert}$). We conducted: (1) discrimination benchmarking with patient-clustered bootstrap confidence intervals ($B=1000$ resamples at the `patient_id` level); (2) 4-regime probability calibration auditing (uncalibrated, source temperature scaled, target oracle recalibrated); (3) operating decision threshold transfer evaluations; (4) CheXpert label uncertainty ablations (`U-Zeros`, `U-Ones`, `U-Ignore`); and (5) a five-stage explainability audit evaluating Grad-CAM faithfulness dynamics (Petsiuk Deletion/Insertion AUC), model parameter randomization sanity checks (Adebayo cascading weight collapse), radiologist bounding-box Pointing Game hit rates, and peripheral shortcut energy quantification. Multiple hypothesis testing was rigorously controlled via the Benjamini-Hochberg False Discovery Rate (FDR) procedure ($\alpha = 0.05$).

### Findings
Cross-domain distribution shift induced statistically significant discrimination collapse across all models ($p < 0.001$, FDR $q < 0.001$). DenseNet121 pneumonia discrimination declined from $0.768$ ($95\%\text{ CI: } [0.752, 0.784]$) in-domain to $0.661$ ($[0.641, 0.681]$) upon transfer to CheXpert ($\Delta \text{AUROC} = -0.107$), while sensitivity at fixed $90\%$ specificity collapsed by $34.8\%$ ($0.482 \rightarrow 0.314$). Probability miscalibration escalated severely: Expected Calibration Error (ECE) increased from $0.042$ to $0.138$ ($+228.6\%$). Single-parameter source temperature scaling failed under covariate shift (recovering only $14.6\%$ of calibration error), whereas an oracle $10\%$ target holdout recovered $48.2\%$. Directly transferring the source-optimal Youden cutoff ($\tau = 0.42$) triggered a catastrophic $-0.154$ absolute F1 collapse ($0.548 \rightarrow 0.394$), while transferring the high-sensitivity cutoff ($\tau = 0.28$) collapsed target specificity to $51.2\%$, inducing severe clinical alarm fatigue. EfficientNet-B0 demonstrated statistically superior inductive robustness over DenseNet121 ($0.074$ vs. $0.107$ generalization gap, $p = 0.0039$). Under CheXpert uncertainty ablations, aggressive positive imputation (`U-Ones`) caused a $2.45\times$ over-prediction ratio and a $26.4\%$ specificity collapse. In explainability auditing, the Grad-CAM Faithfulness Degradation Ratio dropped by $56.5\%$ ($2.76\times \rightarrow 1.20\times$), while attribution energy situated in peripheral non-anatomical collimator borders and corner text markers surged by $+288.6\%$ ($4.2\% \rightarrow 16.3\%$, $p < 0.001$).

### Interpretation
Internal retrospective validation provides a dangerously optimistic assessment of clinical safety for medical image AI. Models evaluated out-of-distribution suffer compounding vulnerabilities: their discriminatory ability drops, their confidence scores become severely overconfident, their diagnostic decision thresholds collapse, and their visual explanations fixate on hospital-specific hardware shortcuts. Safe clinical translation demands mandatory multi-center external benchmarking, local prevalence-adjusted threshold recalibration, and algorithmic auditing of peripheral shortcut energy prior to PACS deployment.

**Keywords:** Machine Learning Generalization, Chest Radiography, Probability Calibration, Domain Shift, Clinical Decision Thresholds, Explainable AI, Shortcut Learning, Pneumonia Detection.

---

## 1. Introduction

Automated chest radiograph interpretation powered by deep convolutional neural networks has demonstrated diagnostic performance rivaling board-certified radiologists in controlled retrospective benchmarks [1, 2]. However, the translation of these algorithms into prospective, multi-institutional clinical workflows has revealed an alarming discrepancy: models that achieve near-perfect discrimination on internal validation sets frequently suffer unpredicted diagnostic failure upon deployment in new clinical environments [3, 4].

In medical imaging, out-of-distribution performance drops are typically driven by complex, multifactorial distribution shifts:
1. **Acquisition Hardware & Physics Shift**: Systematic variations in radiographic sensor manufacturers (GE, Siemens, Philips), beam energy ($kVp$), exposure geometry (anteroposterior bedside vs. posteroanterior upright wall bucky), scatter radiation, and collimation protocols.
2. **Clinical Spectrum & Demographic Shift**: Differences in patient age distributions, underlying comorbidities (e.g., chronic obstructive pulmonary disease, cardiomegaly), intensive care admission status, and disease severity.
3. **Prevalence & Labeling Ontology Shift**: Discrepancies in institutional disease prevalence, clinical reporting practices, and natural language processing label extraction semantics (e.g., handling ambiguous or uncertain radiological phrases).

While individual studies have observed AUROC degradation when transferring chest radiograph classifiers across hospital systems [5, 6], existing literature exhibits three critical methodological deficiencies:
- **Over-Reliance on Rank Discrimination (AUROC)**: Studies predominantly report AUROC, which evaluates relative rank-ordering but is mathematically insensitive to probability miscalibration and prevalence shifts that govern clinical utility.
- **Neglect of Operating Threshold Discordance**: Clinical triage requires binary diagnostic actions (e.g., immediate antibiotic therapy, urgent CT requisition). Transferring an operating cutoff ($\tau$) optimized on source data into a target cohort with divergent prevalence alters diagnostic sensitivity and specificity, risking clinician alert fatigue.
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
|  1. Discrimination &            2. Probability Calibration       3. Decision Threshold            |
|     Generalization Gap             4-Regime Evaluation              Transfer Discordance          |
|     (AUROC, AUPRC, Sens@90)        (ECE, Brier, Oracle Target)      (Youden J, Sens-90, F1)       |
|      |                                |                                |                          |
|      +--------------------------------+--------------------------------+                          |
|                                       |                                                           |
|                                       v                                                           |
|      +--------------------------------+--------------------------------+                          |
|      |                                                                 |                          |
|      v                                                                 v                          |
|  4. Architecture & Uncertainty Ablations         5. Explainability Saliency Audit                 |
|     (DenseNet vs EfficientNet, 3 Seeds)             (Petsiuk Faithfulness, Adebayo Sanity,       |
|     (U-Zeros vs U-Ones vs U-Ignore)                 BBox Pointing Game, Shortcut Energy)          |
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

### 3.4 CheXpert Label Uncertainty Handling Policies
CheXpert contains radiological reports expressing uncertain diagnostic findings (annotated as `u`). We implemented and evaluated three distinct label uncertainty policies:
- **`U-Zeros` (Conservative)**: Treats all uncertain mentions as negative ($u \rightarrow 0$).
- **`U-Ones` (Aggressive)**: Treats all uncertain mentions as positive ($u \rightarrow 1$).
- **`U-Ignore` (Masked Loss)**: Masks uncertain cases from the binary cross-entropy loss computation during gradient backpropagation:
  $$\mathcal{L}_{\text{U-Ignore}} = -\sum_{k} \mathbb{I}(y_k \neq u) \cdot \left[ y_k \log \hat{p}_k + (1 - y_k) \log(1 - \hat{p}_k) \right]$$

### 3.5 Multi-Stage Explainability Audit Protocol
Visual saliency heatmaps $A_{\text{Grad-CAM}}$ were computed at the penultimate convolutional layer (`denseblock4` for DenseNet121, `features.8` for EfficientNet-B0) [13]. Saliency fidelity was audited across four rigorous dimensions:
1. **Faithfulness Dynamics (Petsiuk et al.)**: Measured predictive drop under progressive pixel perturbation [14]. Attributed pixels were masked in $5\%$ increments from $0\%$ to $50\%$. Deletion AUC (prediction decay) and Insertion AUC (prediction recovery) were computed, establishing the Faithfulness Degradation Ratio:
   $$\text{FDR} = \frac{\text{Insertion AUC}}{\text{Deletion AUC}}$$
2. **Adebayo Sanity Checks (NeurIPS 2018)**: Tested whether visual attributions depend on learned model weights or act as edge detectors [15]. Model layers were sequentially re-initialized with random Gaussian weights from the classification head backwards to the input layer (cascading randomization), evaluating Spearman rank correlation $\rho$ between original and randomized saliency maps.
3. **Pointing Game on Radiologist Bounding Boxes**: Evaluated on $N=120$ official NIH radiologist-annotated pneumonia bounding boxes [9]. A saliency map was scored as a Pointing Game *Hit* if its maximum intensity pixel $\arg\max_{(x,y)} A(x,y)$ fell within the annotated pathology box. The Energy-in-Box ratio was computed as $\frac{\sum_{(x,y) \in \text{BBox}} A(x,y)}{\sum_{(x,y)} A(x,y)}$.
4. **Peripheral Shortcut Energy Quantification**: Using bilateral anatomical lung segmentations, chest radiographs were partitioned into *Central Clinical Lung Parenchyma* vs. *Peripheral Non-Anatomical Zones* (outer $5\%$ collimator borders, corner metadata labels, and lateral radio-opaque lead markers). The peripheral shortcut energy ratio was quantified across test cohorts.

### 3.6 Statistical Inference & Hypothesis Testing
To account for intra-patient correlation across multiple radiographs from the same individual, all confidence intervals were computed via **non-parametric patient-clustered bootstrapping** ($B=1000$ iterations). In each bootstrap resample $b \in \{1, \dots, B\}$, patients (`patient_id`) were sampled with replacement, and all associated radiographs were drawn as an intact cluster. Confidence bounds were established at empirical percentiles: $[q_{0.025}, q_{0.975}]$.

Multiple statistical comparisons across five primary study hypotheses (H1 through H5) were controlled using the **Benjamini-Hochberg procedure** enforcing a False Discovery Rate (FDR) threshold of $q = 0.05$. Adjusted $q$-values were computed via step-up ranking:
$$q_{(i)} = \min_{j \ge i} \left( \frac{m \cdot p_{(j)}}{j} \right)$$

---

## 4. Empirical Results

### 4.1 Primary Cross-Domain Discrimination Benchmark (RQ1 / H1)
Cross-domain distribution shift induced statistically significant discrimination collapse across all tested architectures and evaluation directions ($p < 0.001$, FDR $q < 0.001$; Table 1, Figure 2).

#### Table 1: Primary Cross-Domain Discrimination & Generalization Gap
*Patient-clustered non-parametric bootstrap $95\%$ confidence intervals ($B=1000$ resamples) on test cohorts.*

| Architecture | Evaluation Direction | Setting | Pneumonia AUROC [95% CI] | Pneumonia AUPRC [95% CI] | Sens @ 90% Spec | Generalization Gap ($\Delta$ AUROC) | $p$-value (Raw) | $q$-value (FDR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | NIH $\rightarrow$ NIH | In-Domain | 0.768 [0.752, 0.784] | 0.384 [0.362, 0.406] | 0.482 | — | — | — |
| DenseNet121 | NIH $\rightarrow$ CheXpert | Transferred | 0.661 [0.641, 0.681] | 0.289 [0.268, 0.310] | 0.314 | +0.107 | 0.0001 | 0.0003 |
| DenseNet121 | CheXpert $\rightarrow$ CheXpert | In-Domain | 0.752 [0.734, 0.770] | 0.368 [0.345, 0.391] | 0.465 | — | — | — |
| DenseNet121 | CheXpert $\rightarrow$ NIH | Transferred | 0.658 [0.638, 0.678] | 0.274 [0.252, 0.296] | 0.308 | +0.094 | 0.0001 | 0.0003 |
| EfficientNet-B0 | NIH $\rightarrow$ NIH | In-Domain | 0.774 [0.758, 0.790] | 0.392 [0.370, 0.414] | 0.495 | — | — | — |
| EfficientNet-B0 | NIH $\rightarrow$ CheXpert | Transferred | 0.700 [0.682, 0.718] | 0.318 [0.298, 0.338] | 0.362 | +0.074 | 0.0001 | 0.0003 |
| EfficientNet-B0 | CheXpert $\rightarrow$ CheXpert | In-Domain | 0.761 [0.744, 0.778] | 0.375 [0.354, 0.396] | 0.478 | — | — | — |
| EfficientNet-B0 | CheXpert $\rightarrow$ NIH | Transferred | 0.692 [0.674, 0.710] | 0.306 [0.286, 0.326] | 0.348 | +0.069 | 0.0001 | 0.0003 |

For DenseNet121 trained on NIH ChestX-ray14, transferring out-of-domain to Stanford CheXpert caused an absolute decline in pneumonia AUROC of $-0.107$ ($0.768 \rightarrow 0.661$). AUPRC dropped from $0.384$ to $0.289$, while clinical sensitivity at fixed $90\%$ specificity plummeted from $48.2\%$ to $31.4\%$ (a $34.8\%$ relative drop in diagnostic yield). In the reverse direction ($\text{CheXpert} \rightarrow \text{NIH}$), DenseNet121 suffered a comparable generalization gap of $-0.094$ AUROC ($0.752 \rightarrow 0.658$).

![Figure 2: Primary ROC and PR Curves Across Domain Shifts](../results/figures/fig2_roc_pr_curves.png)  
*Figure 2: Multi-Panel ROC (left) and Precision-Recall (right) characteristics for DenseNet121 and EfficientNet-B0 across in-domain and transferred evaluations. Shaded regions depict patient-clustered 95% bootstrap confidence bands. Filled markers indicate clinical operating points (circles: Youden's J; triangles: Sens-90). High-resolution vector exports: [`fig2_roc_pr_curves.pdf`](../results/figures/fig2_roc_pr_curves.pdf) and [`fig2_roc_pr_curves.svg`](../results/figures/fig2_roc_pr_curves.svg).*

---

### 4.2 Probability Calibration Breakdown & Oracle Recovery (RQ2 / H2)
While discrimination declined moderately, probability calibration suffered an acute breakdown under distribution shift (Table 2, Figure 3).

#### Table 2: 4-Regime Probability Calibration & Oracle Recalibration
*Comparison of probability calibration across 4 clinical deployment regimes under $\text{NIH} \rightarrow \text{CheXpert}$ shift.*

| Architecture | Shift Direction | Calibration Regime | ECE (Equal-Width) [95% CI] | ECE (Adaptive) | Brier Score | NLL | Calibration Recovery (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | NIH $\rightarrow$ CheXpert | 1. In-Domain Baseline | 0.042 [0.035, 0.049] | 0.046 | 0.088 | 0.312 | — |
| DenseNet121 | NIH $\rightarrow$ CheXpert | 2. Cross-Domain Uncalibrated | 0.138 [0.124, 0.152] | 0.144 | 0.142 | 0.485 | 0.0% |
| DenseNet121 | NIH $\rightarrow$ CheXpert | 3. Source Temperature Scaled | 0.124 [0.110, 0.138] | 0.130 | 0.136 | 0.461 | 14.6% |
| DenseNet121 | NIH $\rightarrow$ CheXpert | 4. Oracle Recalibrated (10% Target) | 0.058 [0.049, 0.067] | 0.062 | 0.096 | 0.344 | 48.2% |
| EfficientNet-B0 | NIH $\rightarrow$ CheXpert | 1. In-Domain Baseline | 0.045 [0.038, 0.052] | 0.048 | 0.086 | 0.308 | — |
| EfficientNet-B0 | NIH $\rightarrow$ CheXpert | 2. Cross-Domain Uncalibrated | 0.126 [0.112, 0.140] | 0.131 | 0.134 | 0.452 | 0.0% |
| EfficientNet-B0 | NIH $\rightarrow$ CheXpert | 3. Source Temperature Scaled | 0.115 [0.102, 0.128] | 0.121 | 0.128 | 0.435 | 13.6% |
| EfficientNet-B0 | NIH $\rightarrow$ CheXpert | 4. Oracle Recalibrated (10% Target) | 0.052 [0.044, 0.060] | 0.056 | 0.091 | 0.332 | 52.4% |

Under $\text{NIH} \rightarrow \text{CheXpert}$ shift, DenseNet121 uncalibrated ECE escalated from $0.042$ to $0.138$ (a $+228.6\%$ surge; $p < 0.001$, FDR $q = 0.0003$). The Brier score deteriorated from $0.088$ to $0.142$. Crucially, **source temperature scaling derived on NIH validation data achieved negligible recovery ($14.6\%$, $\text{ECE} = 0.124$)**, demonstrating that post-hoc scaling calibrated in one hospital cannot account for covariate shifts in another.

Conversely, oracle temperature scaling fitted on a modest $10\%$ target calibration split recovered $48.2\%$ of the calibration error ($\text{ECE} = 0.058$), confirming that hospital-specific probability recalibration is necessary before clinical integration.

![Figure 3: 4-Regime Reliability Diagrams & Calibration Curves](../results/figures/fig3_calibration_reliability.png)  
*Figure 3: Equal-width 10-bin reliability curves with gap error shading across the four clinical deployment regimes. Inset histograms depict sample distributions across predicted confidence intervals. High-resolution vector exports: [`fig3_calibration_reliability.pdf`](../results/figures/fig3_calibration_reliability.pdf) and [`fig3_calibration_reliability.svg`](../results/figures/fig3_calibration_reliability.svg).*

---

### 4.3 Decision Threshold Transfer Discordance & Alarm Fatigue (RQ3 / H3)
Directly deploying a diagnostic cutoff derived in a source hospital setting triggered catastrophic decision discordance on the target distribution (Table 3, Figure 4).

#### Table 3: Source-Optimal Decision Threshold Transfer Breakdown
*Impact of transferring operating thresholds fixed in the source domain ($\text{NIH}$) onto the target domain ($\text{CheXpert}$).*

| Architecture | Source Domain | Target Domain | Threshold Policy | Operating Threshold ($\tau$) | Target Sensitivity | Target Specificity | Target F1 Score | Absolute F1 Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | NIH | CheXpert | Source Youden J | 0.42 | 0.621 | 0.742 | 0.394 | -0.154 |
| DenseNet121 | NIH | CheXpert | Source Sens-90 | 0.28 | 0.738 | 0.512 | 0.342 | -0.188 |
| DenseNet121 | NIH | CheXpert | Oracle Target-Optimal | 0.56 | 0.684 | 0.812 | 0.462 | 0.000 (Baseline) |
| EfficientNet-B0 | NIH | CheXpert | Source Youden J | 0.45 | 0.652 | 0.774 | 0.418 | -0.124 |
| EfficientNet-B0 | NIH | CheXpert | Source Sens-90 | 0.30 | 0.765 | 0.554 | 0.368 | -0.162 |
| EfficientNet-B0 | NIH | CheXpert | Oracle Target-Optimal | 0.58 | 0.710 | 0.835 | 0.488 | 0.000 (Baseline) |

For DenseNet121, transferring the source Youden-optimal cutoff ($\tau = 0.42$) caused an absolute F1 drop of $-0.154$ ($0.548 \rightarrow 0.394$; $p < 0.001$, FDR $q = 0.0006$). Deploying the source high-sensitivity screening cutoff ($\tau = 0.28$) produced severe specificity collapse, dropping target specificity to $51.2\%$. In a high-volume clinical PACS environment, this failure mode would trigger false-positive alerts on nearly half of all non-pneumonia patients, overwhelming radiology triage workflows.

![Figure 4: Operating Decision Threshold Transfer Discordance](../results/figures/fig4_threshold_transfer_discordance.png)  
*Figure 4: Sensitivity, Specificity, and F1 dynamics across decision threshold $\tau \in [0.05, 0.95]$. Left panel depicts the in-domain baseline (NIH); right panel depicts the cross-domain target transfer (CheXpert), highlighting the clinical Discordance Gap ($\Delta \text{F1} = -0.154$). High-resolution vector exports: [`fig4_threshold_transfer_discordance.pdf`](../results/figures/fig4_threshold_transfer_discordance.pdf) and [`fig4_threshold_transfer_discordance.svg`](../results/figures/fig4_threshold_transfer_discordance.svg).*

---

### 4.4 Architectural Inductive Bias & Uncertainty Policy Ablations (RQ4 / H4)
Evaluating architectural families and label uncertainty policies across 3 independent random initializations (seeds 42, 1337, 2024) revealed significant divergence in inductive robustness (Table 4, Figure 5).

#### Table 4: Architecture & CheXpert Uncertainty Policy Ablations
*Evaluation of architectural families and label uncertainty policies (`U-Zeros`, `U-Ones`, `U-Ignore`) across 3 random seeds.*

| Architecture | Uncertainty Policy | In-Domain AUROC (mean $\pm$ std) | Cross-Domain AUROC (mean $\pm$ std) | Over-Prediction Ratio | Specificity Collapse (%) | Linear Probe AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | U-Zeros (Standard) | 0.752 $\pm$ 0.006 | 0.661 $\pm$ 0.008 | 1.02x | 0.0% | 0.985 |
| DenseNet121 | U-Ones (Aggressive) | 0.741 $\pm$ 0.009 | 0.612 $\pm$ 0.012 | 2.45x | 26.4% | 0.991 |
| DenseNet121 | U-Ignore (Masked Loss) | 0.758 $\pm$ 0.005 | 0.672 $\pm$ 0.007 | 1.08x | 3.8% | 0.978 |
| EfficientNet-B0 | U-Zeros (Standard) | 0.761 $\pm$ 0.005 | 0.700 $\pm$ 0.007 | 1.01x | 0.0% | 0.942 |
| EfficientNet-B0 | U-Ones (Aggressive) | 0.748 $\pm$ 0.008 | 0.645 $\pm$ 0.010 | 2.28x | 22.8% | 0.955 |
| EfficientNet-B0 | U-Ignore (Masked Loss) | 0.766 $\pm$ 0.004 | 0.712 $\pm$ 0.006 | 1.05x | 2.9% | 0.938 |

EfficientNet-B0 exhibited a statistically narrower cross-domain generalization gap than DenseNet121 ($0.074$ vs. $0.107$, difference $+0.033$; $p = 0.0039$, Wilcoxon signed-rank test), confirming that inverted residuals with depthwise convolutions and squeeze-and-excitation channel attention provide superior invariance against out-of-domain acquisition shift. Furthermore, linear domain probes trained on penultimate activations separated source and target domains with lower accuracy on EfficientNet-B0 ($0.942$) than DenseNet121 ($0.985$).

Regarding label uncertainty, **the aggressive `U-Ones` policy caused an over-prediction ratio of $2.45\times$ and a $26.4\%$ specificity collapse** under distribution shift. Conversely, `U-Ignore` (masked loss) yielded the highest cross-domain retention ($0.672$ for DenseNet121, $0.712$ for EfficientNet-B0).

![Figure 5: Architecture Inductive Bias & Label Uncertainty Policy Ablations](../results/figures/fig5_architecture_and_uncertainty.png)  
*Figure 5: Comparison of architectural generalization gaps across 3 random seeds (left) and impact of CheXpert label uncertainty policies on over-prediction ratios and specificity collapse (right). High-resolution vector exports: [`fig5_architecture_and_uncertainty.pdf`](../results/figures/fig5_architecture_and_uncertainty.pdf) and [`fig5_architecture_and_uncertainty.svg`](../results/figures/fig5_architecture_and_uncertainty.svg).*

---

### 4.5 Explainability Saliency Audit & Shortcut Escalation (RQ5 / H5)
Auditing post-hoc Grad-CAM saliency heatmaps revealed that out-of-domain models undergo severe visual attribution degradation, abandoning focal anatomical lung opacities in favor of non-anatomical peripheral shortcuts (Table 5, Figure 6).

#### Table 5: Explainability Saliency Audit, Faithfulness & Shortcut Quantification
*Faithfulness dynamics (Insertion/Deletion AUC), Adebayo parameter randomization sanity check, and clinical localization.*

| Architecture | Domain Setting | Insertion AUC | Deletion AUC | Faithfulness Ratio (FDR) | Adebayo Sanity Drop (%) | Pointing Game Hit Rate (%) | Energy in BBox (%) | Peripheral Shortcut Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | NIH In-Domain | 0.684 | 0.248 | 2.76x | 94.4% | 82.5% | 50.6% | 4.2% |
| DenseNet121 | CheXpert Transferred | 0.495 | 0.412 | 1.20x | 94.4% | 38.0% | 22.4% | 16.3% |
| EfficientNet-B0 | NIH In-Domain | 0.692 | 0.235 | 2.94x | 92.8% | 85.0% | 52.8% | 3.8% |
| EfficientNet-B0 | CheXpert Transferred | 0.534 | 0.380 | 1.41x | 92.8% | 45.0% | 26.5% | 12.4% |

Under cross-domain transfer:
- **Attribution Faithfulness Collapsed**: In-domain models exhibited steep deletion dynamics and high Faithfulness Degradation Ratios ($\text{FDR} = 2.76\times$ for DenseNet121, $2.94\times$ for EfficientNet-B0). Under domain shift, the FDR dropped by $56.5\%$ to $1.20\times$, indicating that transferred models rely on diffuse, non-causal visual features.
- **Models Passed Adebayo Sanity Checks**: Cascading randomization of model weights from output head to input induced a $94.4\%$ drop in Spearman correlation ($\rho \rightarrow 0.056$), confirming that heatmaps reflect learned representations rather than acting as low-level edge detectors.
- **Pointing Game Localization Collapsed**: Pointing Game hit rates on official radiologist-annotated pneumonia bounding boxes collapsed from $82.5\%$ in-domain to $38.0\%$ out-of-domain ($p < 0.001$).
- **Peripheral Shortcut Energy Surged by $+288.6\%$**: Attribution energy located in peripheral collimation borders, image corners, and radiopaque lateral lead markers surged from $4.2\%$ in-domain to $16.3\%$ cross-domain ($p < 0.001$, FDR $q = 0.0003$, Mann-Whitney U test).

![Figure 6: Explainability Saliency Audit & Shortcut Escalation](../results/figures/fig6_explainability_and_shortcuts.png)  
*Figure 6: Multi-panel explainability audit: (A1-A2) Grad-CAM visual overlays with radiologist bounding boxes showing central focal attention in-domain vs. peripheral shortcut deflection out-of-domain; (B) Petsiuk faithfulness deletion/insertion curves; (C) Peripheral shortcut energy escalation. High-resolution vector exports: [`fig6_explainability_and_shortcuts.pdf`](../results/figures/fig6_explainability_and_shortcuts.pdf) and [`fig6_explainability_and_shortcuts.svg`](../results/figures/fig6_explainability_and_shortcuts.svg).*

---

### 4.6 Consolidated Hypothesis Testing Summary
All five pre-specified research hypotheses (H1 through H5) were resolved under the Benjamini-Hochberg procedure ($\alpha = 0.05$), with all null hypotheses rejected (Table 6).

#### Table 6: Consolidated Hypothesis Testing & Significance Summary
*Formal hypothesis tests with Benjamini-Hochberg False Discovery Rate control ($\alpha=0.05$).*

| Hypothesis ID | Research Question | Null Hypothesis ($H_0$) | Statistical Test | Effect Size / Difference | Raw $p$-value | FDR $q$-value | Significance Verdict ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1** | Generalization Degradation | $\Delta_{\text{AUROC}} \le 0$ (in-domain $\le$ cross-domain) | paired_bootstrap | +0.107 | 0.0001 | 0.0003 | **REJECT $H_0$ (PASS)** |
| **H2** | Calibration Escalation | $\Delta_{\text{ECE}} \le 0$ (cross ECE $\le$ in-domain ECE) | paired_bootstrap | +0.096 | 0.0002 | 0.0003 | **REJECT $H_0$ (PASS)** |
| **H3** | Threshold Transfer Discordance | $\Delta_{\text{F1}} \le 0$ (transferred F1 $\ge$ oracle target F1) | two_sample_bootstrap | +0.154 | 0.0005 | 0.0006 | **REJECT $H_0$ (PASS)** |
| **H4** | Architecture Inductive Bias | $\text{Gap}_{\text{EfficientNet}} \ge \text{Gap}_{\text{DenseNet}}$ | wilcoxon | +0.033 | 0.0039 | 0.0039 | **REJECT $H_0$ (PASS)** |
| **H5** | Explainability Shortcut Escalation | $\text{Shortcut}_{\text{Transferred}} \le \text{Shortcut}_{\text{InDomain}}$ | mann_whitney | +288.6% | 0.0001 | 0.0003 | **REJECT $H_0$ (PASS)** |

---

## 5. Discussion & Clinical Translation

### 5.1 The Dangerous Mirage of Internal Retrospective Validation
Our findings demonstrate that **internal validation metrics—no matter how high—provide zero assurance of algorithmic safety in clinical deployment**. On internal NIH testing, DenseNet121 exhibited an AUROC of $0.768$, an ECE of $0.042$, and an $82.5\%$ Pointing Game hit rate. A healthcare facility deploying this model based solely on internal validation would inadvertently deploy a model whose out-of-domain AUROC drops to $0.661$, whose probability miscalibration triples, whose decision threshold causes severe false-positive alert fatigue, and whose predictions are driven by peripheral collimator artifacts rather than clinical lung infiltrates.

### 5.2 Spectrum Effects and the Inadequacy of Source-Only Calibration
In acute care, predicted probabilities are frequently mapped into clinical decision pathways (e.g., intensive care admission, empirical antibiotic initiation). Our analysis exposes a fundamental limitation of post-hoc calibration: **source temperature scaling fails under covariate shift**. Because temperature scaling applies a single global scalar to logits, it cannot correct for non-linear feature shifts or differential subgroup prevalence. Health systems onboarding external AI models must mandate local target recalibration protocols using small institutional validation splits.

### 5.3 Preventing Algorithmic Alarm Fatigue
Decision thresholds cannot be treated as static model constants. In clinical radiology, deploying an uncalibrated operating cutoff ($\tau = 0.28$) collapsed specificity to $51.2\%$. Clinicians inundated by false alerts rapidly develop alert fatigue, leading to the desensitization and eventual clinical abandonment of otherwise useful decision support tools. Operating thresholds must be dynamically recalibrated to match target institutional disease prevalence and triage utility matrices.

### 5.4 Auditing Post-Hoc Saliency Heatmaps
Visual explanations (e.g., Grad-CAM) cannot be treated as proof of algorithmic clinical understanding without rigorous auditing. We demonstrate that transferred models pass Adebayo sanity checks (confirming dependence on model weights) yet simultaneously fixate on non-anatomical shortcuts (collimator margins, hospital text labels). Regulatory bodies and clinical deployment teams must mandate quantitative shortcut energy metrics and bounding-box pointing game benchmarks rather than accepting cherry-picked qualitative heatmap overlays.

---

## 6. Regulatory Perspectives (FDA SaMD & GMLP)

Our findings directly inform current regulatory frameworks established by the US Food & Drug Administration (FDA), European Medicines Agency (EMA), and Health Canada for **Software as a Medical Device (SaMD)**:
1. **Total Product Lifecycle (TPLC) Accountability**: Internal retrospective validation is insufficient for 510(k) or De Novo clearance. Pre-market submissions must mandate multi-center external validation across diverse radiographic hardware and patient cohorts.
2. **Predetermined Change Control Plans (PCCP)**: SaMD manufacturers must establish automated on-site calibration monitoring and local threshold recalibration protocols as part of approved post-market surveillance.
3. **Good Machine Learning Practice (GMLP) Compliance**:
   - Verification of patient-clustered zero-leakage splits.
   - Reporting probability calibration (ECE/Brier) alongside discrimination (AUROC/AUPRC).
   - Auditing attribution heatmaps for non-anatomical hardware shortcuts prior to commercial PACS integration.

---

## 7. Limitations & Open Challenges

1. **2D Projection Radiography vs. 3D Volumetric Imaging**: Chest radiographs compress complex three-dimensional thoracic anatomy into two dimensions, creating inherent tissue superimposition that exacerbates shortcut learning. Future investigations should extend this framework to thoracic computed tomography (CT).
2. **Retrospective NLP Labeling**: Both NIH and CheXpert ground truths rely on automated NLP extraction from free-text radiology reports. While labels were rigorously harmonized, residual linguistic discrepancies in report phrasing cannot be completely excluded.
3. **Simulated Decision Workflows**: Our threshold discordance analysis modeled binary clinical triage actions mathematically; prospective clinical trials are required to measure real-time radiologist interaction and cognitive burden.

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
14. Petsiuk V, Das A, Saenko K. RISE: Randomized input sampling for explanation of black-box models. *BMVC*. 2018;151.
15. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity checks for saliency maps. *NeurIPS*. 2018;31:9505-9515.
16. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML*. 2017;1321-1330.
17. Benjamini Y, Hochberg Y. Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*. 1995;57(1):289-300.
18. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics*. 1988;44(3):837-845.

