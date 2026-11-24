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
Deep learning systems for automated chest radiograph interpretation frequently claim strong diagnostic discrimination on internal retrospective test sets. When deployed across distinct healthcare institutions, however, classifiers encounter coupled shifts in patient demographics, inpatient acuity, radiographic projection geometry (anteroposterior bedside portable versus posteroanterior upright), and reporting terminology. Reporting Area Under the Receiver Operating Characteristic curve (AUROC) alone obscures how external shift alters calibration, operating specificity at source-locked screening thresholds, and sensitivity to uncertain radiological labels.

### Methods
We evaluated six locked, hash-verified NIH ChestX-ray14 models—three random seeds (`42`, `1337`, `2024`) each of DenseNet-121 and EfficientNet-B0—zero-shot across the external Stanford CheXpert cohort under a pre-specified, SHA-256-locked analysis plan (`ANALYSIS_PLAN_TRANSFER.md` v2.2.3, execution run `oldckpt_real_v1`). Models were trained strictly on patient-disjoint NIH ChestX-ray14 splits (`N = 73,024` train across `23,807` patients) with inverse-frequency positive class weighting and evaluated after an automated NIH Sanity Gate on fresh NIH validation (`N = 13,500`, `4,201` patients), NIH test (`N = 22,744`, `2,518` patients), and all frontal CheXpert radiographs (`N = 191,027`, `64,534` patients; zero decode exclusions). Primary inference (`u_zeros` pneumonia policy) tested six confirmatory hypotheses (three endpoints $\times$ two architectures): $\Delta\text{AUROC}$, $\Delta\text{Calibration Slope}$ on raw logits, and $\Delta\text{Sensitivity}$ at source-validation $\tau_{\text{sens90}}$, using three-seed mean paired contrasts with $B = 2,000$ independent patient-clustered bootstrap resamples (`seed = 42`) and Benjamini-Hochberg False Discovery Rate (FDR) control ($q < 0.05$). Secondary analyses evaluated 15-bin calibration, source-fitted temperature and Platt scaling, 20-split patient-disjoint Target-10% Platt recalibration, all four CheXpert label policies, exploratory subgroups, and pre-specified sensitivity to two decoded static-noise CheXpert outliers (`PRE-E-001`).

### Findings
Both architectures suffered large, statistically significant external discrimination losses under NIH-to-CheXpert transfer: DenseNet-121 declined from `0.675` (`[0.652, 0.700]`) on NIH test to `0.591` (`[0.584, 0.599]`) on CheXpert ($\Delta\text{AUROC} = +0.0837$ `[+0.0588, +0.1082]`, $p = 0.0010$, $q_{\text{BH}} = 0.0015$), and EfficientNet-B0 declined from `0.683` (`[0.658, 0.707]`) to `0.608` (`[0.601, 0.616]`, $\Delta\text{AUROC} = +0.0741$ `[+0.0494, +0.0995]`, $p = 0.0010$, $q_{\text{BH}} = 0.0015$). Calibration slope compressed significantly for DenseNet-121 (`0.581 -> 0.406`, $\Delta = +0.1755$, $q_{\text{BH}} = 0.0015$) but remained stable for EfficientNet-B0 (`0.519 -> 0.502`, $\Delta = +0.0168$, $q_{\text{BH}} = 0.6857$). Inverse-frequency training induced severe baseline overconfidence in both domains (mean predicted probability `0.304–0.334` vs `2.16%–2.45%` prevalence; CheXpert $\text{ECE}_{15} = 0.310 \pm 0.097$ DenseNet-121, $0.303 \pm 0.104$ EfficientNet-B0). Critically, source-fitted single-parameter temperature scaling failed to resolve intercept overconfidence (CheXpert $\text{ECE}_{15} = 0.235$ / $0.233$), whereas **source-fitted Platt scaling** reduced CheXpert $\text{ECE}_{15}$ to `0.0079 +- 0.0031` (DenseNet-121) and `0.0083 +- 0.0048` (EfficientNet-B0) with zero target labels—closely matching 20-split Target-10% Platt recalibration (`ECE = 0.0011`). At fixed NIH-validation $\tau_{\text{sens90}}$, CheXpert sensitivity increased modestly (`94.6% -> 96.6%`, $q_{\text{BH}} = 0.0240$; `93.8% -> 97.0%`, $q_{\text{BH}} = 0.0015$), but **operating specificity collapsed sharply** (`19.4% -> 6.5%` DenseNet-121; `20.1% -> 6.2%` EfficientNet-B0; $p = 0.0010$).

### Interpretation
Even when disease base rates match closely between hospitals (`2.16%` NIH test vs `2.45%` CheXpert `u_zeros`), out-of-domain deployment incurs nearly 8 AUROC points of discrimination loss, extreme false-positive triage expansion at source rule-out cutoffs, and architectural divergence in calibration slope drift. Furthermore, single-parameter temperature scaling fails when models carry training-induced intercept offsets, whereas two-parameter source-fitted Platt scaling transfers accurately without target labels when institutional prevalences align. Robust hospital deployment mandates external validation, affine post-hoc calibration auditing, and site-specific specificity verification prior to clinical integration.

**Keywords:** Machine Learning Generalization, Chest Radiography, Probability Calibration, Platt Scaling, Domain Shift, Clinical Decision Thresholds, Pneumonia Detection.

---

## 1. Introduction

Automated chest radiograph interpretation using convolutional neural networks has achieved high retrospective performance on curated internal benchmarks [1, 2]. Nevertheless, multi-center evaluations consistently show that diagnostic models trained in one hospital experience substantial degradation when applied to external patient populations [3–6]. Understanding why this degradation occurs—and which failure modes are structural versus remediable through post-hoc calibration—requires evaluating discrimination, probability calibration, and thresholded triage utility jointly under rigorous patient-clustered statistical inference.

In chest radiography, external distribution shift stems from three coupled clinical and technical mechanisms:
1. **Acquisition Geometry & Care Setting Shift**: NIH ChestX-ray14 reflects clinical center radiographs dominated by ambulatory posteroanterior (PA) views (`1024 x 1024` native resolution) [9], whereas Stanford CheXpert frontal examinations are dominated (`84.6%`) by anteroposterior (AP) portable bedside studies from emergency and intensive care unit (ICU) patients with variable aspect ratios and acute support hardware [10].
2. **Population Acuity & Comorbidity Spectrum**: Inpatient ICU cohorts exhibit markedly older age distributions and higher burdens of confounding cardiopulmonary opacities (cardiomegaly, edema, effusion, atelectasis) that radiographically mimic pneumonia.
3. **NLP Labeling Ontology & Uncertainty Handling**: NIH uses binary labels extracted from clinical reports, whereas CheXpert explicitly distinguishes positive (`1.0`), uncertain (`-1.0`), negative (`0.0`), and unmentioned/blank (`NaN`, accounting for `88.20%` of frontal radiographs) mentions [10].

Existing cross-domain imaging literature exhibits three recurring analytical gaps that limit clinical translation:
- **Disproportionate Focus on AUROC Alone**: Rank-ordering metrics obscure massive probability miscalibration (`ECE > 0.25`) caused by class-weighted loss functions commonly used on imbalanced multi-label datasets, even when disease prevalence matches across institutions.
- **Unquantified Operating Point Distortion**: Thresholds calibrated for rule-out screening ($\ge 90\%$ validation sensitivity) on one cohort can produce severe specificity collapse upon transfer, overwhelming clinical workflows with false-positive alerts without lowering sensitivity.
- **Inadequate Uncertainty & Confounder Accounting**: Treating image-level evaluations as independent ignores strong patient clustering (up to `9.03` radiographs/patient in NIH test and `2.96` in CheXpert), inflating statistical confidence, while post-hoc Platt vs temperature recalibration behavior under weighted-loss intercept bias remains rarely contrasted directly.

To address these gaps rigorously without post-hoc outcome selection, we executed **PneumoShift** (`run_id = oldckpt_real_v1`) under a pre-specified, SHA-256-locked transfer protocol across all `191,027` frontal radiographs in Stanford CheXpert.

---

## 2. Clinical Problem Formulation & Study Design

The PneumoShift transfer evaluation follows an end-to-end pre-specified analytical pipeline linking six locked NIH ChestX-ray14 models to fresh forward-inference evaluation across NIH validation, NIH test, and Stanford CheXpert frontal cohorts ([Figure 1](../results/figures/fig1_study_overview.png)).

![Figure 1: Conceptual Study Design & Empirical Transfer Summary](../results/figures/fig1_study_overview.png)
*Figure 1: End-to-end study architecture and pre-specified cross-domain transfer summary (`run_id = oldckpt_real_v1`, `ANALYSIS_PLAN_TRANSFER.md` v2.2.3). Source validation (`N = 13,500`) fixes thresholds and source scalers; confirmatory contrasts compare fresh NIH test (`N = 22,744`) against all frontal Stanford CheXpert radiographs (`N = 191,027`) under independent $B = 2,000$ patient-clustered bootstrapping and 6-test Benjamini-Hochberg FDR control. Vector formats available at [`fig1_study_overview.pdf`](../results/figures/fig1_study_overview.pdf) and [`fig1_study_overview.svg`](../results/figures/fig1_study_overview.svg).*

### 2.1 Dataset Cohorts & Measurement Verification
We evaluated transfer from **NIH ChestX-ray14** (Source) [9] to **Stanford CheXpert** (`chexpert-v10-small`, Target) [10]:
1. **NIH ChestX-ray14 (`112,120` total radiographs, `30,805` unique patients)**: Partitioned cleanly by `Patient ID` in `data/splits/nih/nih_manifest.json` with verified zero patient overlap across four splits:
   - *Training (`nih_train.csv`)*: `73,024` images (`23,807` patients, pneumonia prevalence `1.01%`).
   - *Validation (`nih_val.csv`)*: `13,500` images (`4,201` patients; `139` positive images / `1.03%`, `110` positive patients / `2.62%`). Used strictly for early stopping, operating threshold selection, and source recalibration fitting.
   - *Test (`nih_test.csv`)*: `22,744` images (`2,518` patients; `491` positive images / `2.16%`, `277` positive patients / `11.00%`; mean `9.03` images/patient).
   - *Held-Out Reference (`nih_target_cal.csv`)*: `2,852` images (`279` patients).
2. **Stanford CheXpert Frontal (`224,316` total raw radiographs, `65,240` patients)**: Filtered strictly to frontal radiographs (`Frontal/Lateral != "Lateral"`), yielding **`191,027` frontal radiographs across `64,534` unique patients** (mean `2.96` images/patient), measured prior to inference and written to `CHEXPERT_COHORT_MEASURED.json`. Zero images failed decoding (`EXCLUSIONS.json`: `n_excluded = 0`), leaving all `191,027` frontal radiographs intact. Measured projection distribution comprises `161,590` AP (`84.6%`), `29,420` PA (`15.4%`), and `17` lacking explicit projection tags; sex comprises `112,151` Male and `78,875` Female (`1` unknown); age strata comprise `<40` yr (`N = 26,268`), `40–64` yr (`N = 79,586`), and `>=65` yr (`N = 85,173`).

Crucially, image-level pneumonia prevalence matches closely across evaluation cohorts (`2.16%` in NIH test vs `2.45%` in CheXpert under primary `u_zeros`), isolating covariate, projection, and labeling shift from coarse image-level base-rate mismatch.

### 2.2 Harmonized Multi-Label Head & Image Preprocessing
All models were trained jointly over the 8 canonical shared pathologies (`SHARED_CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax", "No Finding"]`), with Pneumonia evaluated strictly from logit column index `5` ($z_5$). In both cohorts, single-channel grayscale radiographs are converted to 3 identical RGB channels via PIL `.convert("RGB")`, resized on the shorter edge to `256` pixels, center-cropped to `224 x 224`, and ImageNet-normalized. As pre-stated in `ANALYSIS_PLAN_TRANSFER.md` §10.3, `CenterCrop(224)` preserves the full central field for square NIH inputs (`1024 x 1024`) while trimming longer-edge peripheral anatomy on non-square CheXpert scans.

### 2.3 Pre-Specified CheXpert Uncertainty Label Policies
Empirical frontal CheXpert Pneumonia mentions comprise `4,675` Positive (`1.0`), `15,981` Uncertain (`-1.0`), `1,875` Negative (`0.0`), and `168,496` Blank/Unmentioned (`NaN`, `88.20%`). We evaluated four pre-specified evaluation policies (§3.1–3.4):
1. **`u_zeros` (Primary Confirmatory Policy)**: `1.0 -> 1`, `-1.0 / NaN -> 0` (`N = 191,027`, `4,675` pos, image prevalence `2.45%`, patient prevalence `6.09%`).
2. **`u_ones` (Sensitivity 1)**: `1.0 / -1.0 -> 1`, `NaN -> 0` (`N = 191,027`, `20,656` pos, image prevalence `10.81%`, patient prevalence `21.45%`).
3. **`u_excluded` (Sensitivity 2)**: Uncertain `-1.0` dropped (`15,981` removed), `NaN -> 0` (`N = 175,046` across `61,490` patients, `4,675` pos, image prevalence `2.67%`).
4. **`explicitly_labeled_only` (Sensitivity 3)**: Drops both `-1.0` and `NaN`, isolating explicit mentions (`N = 6,550` across `5,522` patients, `4,675` pos, artificial prevalence `71.37%`).

---

## 3. Methods

### 3.1 Model Architectures, Training Configuration, and Provenance Verification
We evaluated two convolutional families initialized from ImageNet weights across three deterministic training seeds (`42`, `1337`, `2024`):
- **DenseNet-121** [11]: Densely connected convolutional architecture with $D = 1024$ pooled representation before a linear classification head (`dropout = 0.2`).
- **EfficientNet-B0** [12]: Compound-scaled mobile inverted bottleneck architecture with squeeze-and-excitation blocks and $D = 1280$ pooled representation (`dropout = 0.2`).

As specified in `configs/train/nih_densenet121.yaml` and `configs/train/nih_efficientnet_b0.yaml`, models were trained end-to-end on NIH training data for up to `10` epochs with AdamW (`lr = 1e-4`, `weight_decay = 1e-5`), `ReduceLROnPlateau` scheduling (`factor = 0.5`, `patience = 2`), gradient clipping (`1.0`), early stopping on `val_pneumonia_auroc` (`patience = 3`), and inverse class-frequency positive-weighted binary cross-entropy (`pos_weight`) to address severe class imbalance (`~1%` pneumonia). Note that `CheXpert -> CheXpert` and `CheXpert -> NIH` cross-trained models were not included in this evaluation; inference exclusively tested zero-shot transfer of the six SHA-256-locked NIH checkpoints verified against `configs/checkpoint_manifest.json` (`4ae0b9fc...`).

Before any CheXpert inference executed on Kaggle (`Tesla T4`, `PyTorch 2.11.0+cu130`, `git_commit = d79bc756ba9a60d628674f2af1a683bbc25e2e6e`), an automated **NIH Sanity Gate** verified fresh NIH validation and test logits against `configs/nih_test_reference_logits.npz` (`026fb68e...`) within tolerance ($\Delta\text{AUROC} \le 10^{-4}$, Spearman $\rho \ge 0.999$). All prediction bundles carry matching `data_source == "real_inference"`, `run_id == "oldckpt_real_v1"`, `analysis_plan_sha256` (`10baae5f...`), identical GPU/PyTorch environment stamps, and verified row ordering across all six checkpoints.

### 3.2 Numerically Stable Logit-Space Protocol (`DEVIATIONS.md` `PRE-E-001`)
During label-blind pre-analysis metadata/QC inspection prior to computing any outcome metrics, exactly two decoded CheXpert frontal images in `efficientnet_b0_2024` exhibited extreme finite logits (`|logit| > 80`): row `68292` (`patient20683/study2/view1_frontal.jpg`, PA, Pneumonia logit `-54.97`, Edema logit `-98.44`) and row `76123` (`patient23121/study4/view2_frontal.jpg`, AP, Pneumonia logit `-123.38`). Visual source inspection confirmed both were pure static-noise images without visible thoracic anatomy. Under `float32` sigmoid conversion, `logit = -123.38` underflows to exact `0.0`, which would yield `inf` or arbitrary clipping distortion in `log(y_score)`. As logged in `DEVIATIONS.md` (`PRE-E-001`), the primary analysis retains all $N = 191,027$ radiographs (`0` clipped or winsorized) and computes all probabilities ($\hat{p} = \sigma(z)$ via `scipy.special.expit`), NLL (`logaddexp`), recalibration fits, and logistic calibration regressions strictly in `float64` directly from `y_logits`, supplemented by a pre-specified sensitivity comparison excluding rows `(68292, 76123)` ($N = 191,025$).

### 3.3 Probability Calibration & Post-Hoc Recalibration Protocols
Calibration was evaluated with $M = 15$ bins (`src/pneumoshift/evaluation/calibration.py`) across both equal-width (`ECE_uniform`) and equal-mass quantile (`ECE_adaptive`) schemes, alongside Brier score, Brier Skill Score ($\text{BSS} = 1 - \text{BS} / [\bar{y}(1-\bar{y})]$), NLL, and logistic calibration slope ($b$) and intercept ($a$) estimated by unpenalized binomial GLM on raw logits $z$: $\text{logit}(P(Y=1)) = a + b \cdot z$ (perfect calibration: $a = 0, b = 1$). We evaluated five regimes per architecture:
1. **In-Domain Uncalibrated (NIH Test)**: Raw logits $\hat{p} = \sigma(z)$.
2. **Target Uncalibrated (CheXpert Frontal)**: Raw transferred logits $\hat{p} = \sigma(z)$.
3. **Source Temperature Scaled (NIH Validation $\to$ CheXpert)**: Single scalar $T > 0$ fitted by minimizing NLL on fresh NIH validation logits ($N = 13,500$), applied unmodified as $\sigma(z / T)$.
4. **Source Platt Scaled (NIH Validation $\to$ CheXpert)**: Two-parameter affine logit model $\sigma(a z + b)$ fitted via unpenalized maximum likelihood on fresh NIH validation logits ($N = 13,500$), applied unmodified to CheXpert.
5. **Target-10% Platt Recalibration (Institutional Adaptation Benchmark)**: Across `20` random patient-level splits (`seeds = 100..119`, zero patient overlap), affine Platt parameters $(a, b)$ were fitted on `10%` of CheXpert patients and evaluated on the remaining `90%`, reporting mean and SD across splits.

### 3.4 Clinical Operating Decision Thresholds
Thresholds were selected individually per seed strictly on fresh NIH validation predictions under $\hat{y} = \mathbb{I}(\sigma(z) \ge \tau)$ with largest-threshold tie-breaking:
- **$\tau_{\text{sens90}}$ (Primary Rule-Out Policy)**: $\max\{\tau : \text{Sensitivity}_{\text{val}}(\tau) \ge 0.90\}$.
- **$\tau_{\text{Youden}}$ (Balanced Informedness)**: $\max\{\arg\max_\tau (\text{Sensitivity}_{\text{val}}(\tau) + \text{Specificity}_{\text{val}}(\tau) - 1)\}$.
- **$\tau_{\text{sens95}}$ (High-Sensitivity Rule-Out)**: $\max\{\tau : \text{Sensitivity}_{\text{val}}(\tau) \ge 0.95\}$.
Per `ANALYSIS_PLAN_TRANSFER.md` §9.4, target-optimal F1 thresholding ($\tau_{\text{max-F1}}$) was explicitly excluded from confirmatory and secondary analyses because F1 cutoffs depend directly on population prevalence.

### 3.5 Statistical Inference, Bootstrap Design, and Multiplicity Control
For each architecture, confirmatory primary contrasts compute three-seed mean differences $\Delta\bar{\theta} = \frac{1}{3}\sum_s (\hat{\theta}_{\text{NIH test}, s} - \hat{\theta}_{\text{CheXpert}, s})$ across three primary endpoints: $\Delta\text{AUROC}$, $\Delta\text{Slope}$, and $\Delta\text{Sens}(\tau_{\text{sens90}})$. Uncertainty and two-sided $p$-values were derived from $B = 2,000$ patient-clustered bootstrap resamples (`seed = 42`), independently resampling NIH test patient clusters (`2,518`) and CheXpert patient clusters (`64,534`) at each iteration $b$ while pairing exact patient indices across all seeds and architectures. Continuity-corrected two-sided empirical $p$-values are defined as:
$$p = \min\left(1.0,\; 2 \cdot \min\left( \frac{1 + \sum_{b=1}^B \mathbb{I}(\Delta\bar{\theta}^{*(b)} \le 0)}{B + 1},\; \frac{1 + \sum_{b=1}^B \mathbb{I}(\Delta\bar{\theta}^{*(b)} \ge 0)}{B + 1} \right)\right)$$
Between-seed standard deviation ($ddof = 1$) is reported separately from bootstrap 95% percentile intervals ($[2.5^{\text{th}}, 97.5^{\text{th}}]$) to distinguish training initialization dispersion from finite cohort sampling uncertainty. Benjamini-Hochberg FDR control ($\alpha = 0.05$) was applied across the exact 6-test confirmatory family ($3 \text{ endpoints} \times 2 \text{ architectures}$). Unpaired DeLong two-sample tests [18] are reported strictly as a secondary sensitivity comparison with an independence caveat. All remaining secondary metrics, label policies (`u_ones`, `u_excluded`, `explicitly_labeled_only`), and demographic/projection subgroups (`AP`/`PA`, `Female`/`Male`, `<40`/`40–64`/`>=65`) are explicitly non-confirmatory.

---

## 4. Empirical Results

### 4.1 Primary Confirmatory Cross-Domain Endpoints & Discrimination Curves
Cross-domain deployment onto Stanford CheXpert induced highly consistent, statistically significant discrimination loss across every seed and both architectures ([Table 1](#table-1-primary-confirmatory-cross-domain-endpoints--per-seed-reference), [Figure 2](../results/figures/fig2_roc_pr_curves.png)). For DenseNet-121, three-seed mean AUROC fell from `0.675` (`95% CI [0.652, 0.700]`; seed SD `0.0338`) on NIH test to `0.591` (`[0.584, 0.599]`; seed SD `0.0352`) on CheXpert ($\Delta\text{AUROC} = +0.0837$ `[+0.0588, +0.1082]`, $p = 0.0010$, $q_{\text{BH}} = 0.0015$), with remarkably tight between-seed gap SD (`0.0016`: `+0.0818`, `+0.0846`, `+0.0846` across seeds `42`, `1337`, `2024`). EfficientNet-B0 fell from `0.683` (`[0.658, 0.707]`; seed SD `0.0240`) to `0.608` (`[0.601, 0.616]`; seed SD `0.0165`), giving $\Delta\text{AUROC} = +0.0741$ `[+0.0494, +0.0995]` ($p = 0.0010$, $q_{\text{BH}} = 0.0015$, gap seed SD `0.0131`: `+0.0780`, `+0.0848`, `+0.0595`). Unclustered two-sample DeLong sensitivity tests agreed across every model (`max p = 4.82e-11` DenseNet-121; `max p = 5.98e-6` EfficientNet-B0). Precision-recall area similarly declined from `0.043 -> 0.036` (DenseNet-121) and `0.045 -> 0.038` (EfficientNet-B0) against comparable baseline prevalences (`2.16%` vs `2.45%`).

In contrast, raw-logit calibration slope showed clear architectural divergence: DenseNet-121 experienced significant logit slope flattening on CheXpert (`0.581 -> 0.406`, $\Delta\text{Slope} = +0.1755$ `[+0.0767, +0.2806]`, $q_{\text{BH}} = 0.0015$), whereas EfficientNet-B0 preserved essentially identical calibration slope across institutions (`0.519 -> 0.502`, $\Delta\text{Slope} = +0.0168$ `[-0.0760, +0.1129]`, $p = 0.6857$, $q_{\text{BH}} = 0.6857$). For Endpoint 3, sensitivity at NIH-validation $\tau_{\text{sens90}}$ increased slightly upon transfer in both DenseNet-121 (`0.946 -> 0.966`, $\Delta = -0.0199$, $q_{\text{BH}} = 0.0240$) and EfficientNet-B0 (`0.938 -> 0.970`, $\Delta = -0.0322$, $q_{\text{BH}} = 0.0015$). Overall, 5 of 6 primary hypothesis tests rejected $H_0$ after Benjamini-Hochberg correction.

#### Table 1: Primary Confirmatory Cross-Domain Endpoints & Per-Seed Reference
*Three-seed means (`42`, `1337`, `2024`) under CheXpert primary `u_zeros` (`N = 191,027`, `64,534` patients) vs fresh NIH test (`N = 22,744`, `2,518` patients), $B = 2,000$ patient-clustered bootstrap (`seed = 42`), `ddof = 1` seed SD, and 6-test Benjamini-Hochberg FDR control.*

| Architecture | Primary Endpoint | NIH Test [95% CI] | CheXpert Frontal [95% CI] | Contrast $\Delta$ [95% CI] | $\Delta$ Seed SD | $p$-value (2-sided) | BH FDR $q$-value | Decision ($q<0.05$) | Per-Seed DeLong Max $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet-121 | AUROC (NIH test vs CheXpert) | `0.675 [0.652, 0.700]` | `0.591 [0.584, 0.599]` | `+0.0837 [+0.0588, +0.1082]` | `0.0016` | `0.0010` | `0.0015` | Reject $H_0$ | `4.82e-11` |
| DenseNet-121 | Calibration Slope (raw logits) | `0.581 [0.490, 0.683]` | `0.406 [0.368, 0.443]` | `+0.1755 [+0.0767, +0.2806]` | `0.1054` | `0.0010` | `0.0015` | Reject $H_0$ | — |
| DenseNet-121 | Sensitivity at NIH-val $\tau_{\text{sens90}}$ | `0.946 [0.929, 0.963]` | `0.966 [0.962, 0.970]` | `-0.0199 [-0.0379, -0.0027]` | `0.0183` | `0.0200` | `0.0240` | Reject $H_0$ | — |
| EfficientNet-B0 | AUROC (NIH test vs CheXpert) | `0.683 [0.658, 0.707]` | `0.608 [0.601, 0.616]` | `+0.0741 [+0.0494, +0.0995]` | `0.0131` | `0.0010` | `0.0015` | Reject $H_0$ | `5.98e-06` |
| EfficientNet-B0 | Calibration Slope (raw logits) | `0.519 [0.439, 0.604]` | `0.502 [0.455, 0.549]` | `+0.0168 [-0.0760, +0.1129]` | `0.0215` | `0.6857` | `0.6857` | Fail to reject | — |
| EfficientNet-B0 | Sensitivity at NIH-val $\tau_{\text{sens90}}$ | `0.938 [0.920, 0.954]` | `0.970 [0.966, 0.973]` | `-0.0322 [-0.0503, -0.0156]` | `0.0243` | `0.0010` | `0.0015` | Reject $H_0$ | — |

![Figure 2: Empirical ROC and Precision-Recall Curves Across Transfer](../results/figures/fig2_roc_pr_curves.png)
*Figure 2: Empirical ROC (**A**) and Precision–Recall (**B**) curves computed strictly from real `oldckpt_real_v1` logits (`scipy.special.expit` + `roc_curve` / `precision_recall_curve`) across seeds `42`, `1337`, and `2024` (mean curves with $\pm 1\text{ SD}$ shaded across-seed bands). Legend brackets display patient-clustered 95% CIs from `results.json`. Triangular and circular markers indicate mean operating points at $\tau_{\text{sens90}}$ and $\tau_{\text{Youden}}$. Vector files: [`fig2_roc_pr_curves.pdf`](../results/figures/fig2_roc_pr_curves.pdf), [`fig2_roc_pr_curves.svg`](../results/figures/fig2_roc_pr_curves.svg).*

---

### 4.2 Probability Calibration Breakdown & Source vs Target Platt Recalibration
Uncalibrated models displayed severe overconfidence in both in-domain NIH test (`ECE_uniform = 0.2924 +- 0.0654`, `mean_pred = 0.3140` vs `2.16%` prevalence for DenseNet-121; `ECE_uniform = 0.2819 +- 0.0355`, `mean_pred = 0.3035` for EfficientNet-B0) and external CheXpert deployment (`ECE_uniform = 0.3097 +- 0.0967`, `NLL = 0.4720`, `BSS = -5.372` for DenseNet-121; `ECE_uniform = 0.3027 +- 0.1041`, `NLL = 0.4485`, `BSS = -4.976` for EfficientNet-B0), driven directly by `pos_weight` intercept shifts ($a_{\text{int}} \approx -3.35$ to $-3.45$, where logits centered near zero correspond to over-predicted probabilities) ([Table 2](#table-2-15-bin-calibration--post-hoc-recalibration-regimes), [Figure 3](../results/figures/fig3_calibration_reliability.png)).

A striking practical distinction emerged between **source-fitted Temperature Scaling** and **source-fitted Platt Scaling** (both fitted exclusively on NIH validation, $N = 13,500$, with zero CheXpert access):
- **Failure of Single-Parameter Temperature Scaling**: Dividing logits by scalar $T$ (fitted $T \in [0.258, 0.926]$) left CheXpert severely miscalibrated (`ECE_uniform = 0.2350 +- 0.1795`, `NLL = 0.4255` DenseNet-121; `ECE_uniform = 0.2330 +- 0.0794`, `NLL = 0.3889` EfficientNet-B0), because temperature scaling preserves $\sigma(0 / T) = 0.5$ and cannot translate the class-weighted intercept offset toward clinical prevalence.
- **Success of Zero-Target Source Platt Scaling**: Two-parameter affine Platt scaling fitted only on NIH validation ($a = 0.754 \pm 0.297, b = -3.630 \pm 0.444$ DenseNet-121; $a = 0.726 \pm 0.240, b = -3.716 \pm 0.170$ EfficientNet-B0) collapsed CheXpert `ECE_uniform` by over `97%` to **`0.0079 +- 0.0031`** (DenseNet-121, `NLL = 0.1163`) and **`0.0083 +- 0.0048`** (EfficientNet-B0, `NLL = 0.1163`), restoring `BSS >= 0.0` without seeing any CheXpert labels.
- **Oracle Target-10% Platt Benchmark (20 Patient Splits)**: Fitting Platt parameters on `10%` of CheXpert patients slightly improved calibration further (`ECE_uniform = 0.0011`, `ECE_adaptive = 0.0023 / 0.0025`, `NLL = 0.1136 / 0.1133`) and corrected residual slope compression (`Slope = 1.066` DenseNet-121, `1.012` EfficientNet-B0).

#### Table 2: 15-Bin Calibration & Post-Hoc Recalibration Regimes
*Three-seed mean $\pm$ SD (`ddof = 1`) across $M = 15$ uniform and adaptive bins evaluated in raw logit space (`expit` / `logaddexp`, `PRE-E-001`). Target-10% Platt summarizes 3-seed mean $\pm$ between-seed SD of 20 patient-disjoint CheXpert holdout splits (`90%` evaluation).*

| Architecture | Cohort | Calibration Regime | $\text{ECE}_{\text{uniform}}$ ($M=15$) | $\text{ECE}_{\text{adaptive}}$ ($M=15$) | Brier Score | BSS | NLL | Cal Slope | Cal Intercept | Mean Pred |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet-121 | NIH Test | 1. In-Domain Uncalibrated | `0.2924 +- 0.0654` | `0.2924 +- 0.0654` | `0.1514 +- 0.0555` | `-6.167 +- 2.626` | `0.4685 +- 0.1267` | `0.581 +- 0.240` | `-3.389 +- 0.376` | `0.3140 +- 0.0654` |
| DenseNet-121 | CheXpert Frontal | 2. Target Uncalibrated | `0.3097 +- 0.0967` | `0.3097 +- 0.0967` | `0.1521 +- 0.0689` | `-5.372 +- 2.887` | `0.4720 +- 0.1512` | `0.406 +- 0.171` | `-3.367 +- 0.280` | `0.3342 +- 0.0967` |
| DenseNet-121 | CheXpert Frontal | 3. Source Temp Scaled (NIH val) | `0.2350 +- 0.1795` | `0.2369 +- 0.1765` | `0.1402 +- 0.1043` | `-4.874 +- 4.370` | `0.4255 +- 0.2578` | `0.203 +- 0.092` | `-3.367 +- 0.280` | `0.2566 +- 0.1841` |
| DenseNet-121 | CheXpert Frontal | 4. Source Platt Scaled (NIH val) | `0.0079 +- 0.0031` | `0.0082 +- 0.0026` | `0.0239 +- 0.0000` | `-0.000 +- 0.001` | `0.1163 +- 0.0000` | `0.563 +- 0.195` | `-1.318 +- 0.903` | `0.0166 +- 0.0031` |
| DenseNet-121 | CheXpert (90% eval) | 5. Target-10% Platt (20 splits) | `0.0011 +- 0.0001` | `0.0023 +- 0.0002` | `0.0238 +- 0.0001` | `0.003 +- 0.002` | `0.1136 +- 0.0010` | `1.066 +- 0.044` | `0.234 +- 0.162` | — |
| EfficientNet-B0 | NIH Test | 1. In-Domain Uncalibrated | `0.2819 +- 0.0355` | `0.2819 +- 0.0355` | `0.1469 +- 0.0128` | `-5.954 +- 0.604` | `0.4553 +- 0.0285` | `0.519 +- 0.178` | `-3.450 +- 0.088` | `0.3035 +- 0.0355` |
| EfficientNet-B0 | CheXpert Frontal | 2. Target Uncalibrated | `0.3027 +- 0.1041` | `0.3027 +- 0.1041` | `0.1427 +- 0.0513` | `-4.976 +- 2.149` | `0.4485 +- 0.1320` | `0.502 +- 0.161` | `-3.350 +- 0.176` | `0.3272 +- 0.1041` |
| EfficientNet-B0 | CheXpert Frontal | 3. Source Temp Scaled (NIH val) | `0.2330 +- 0.0794` | `0.2335 +- 0.0791` | `0.1222 +- 0.0522` | `-4.119 +- 2.188` | `0.3889 +- 0.1314` | `0.258 +- 0.054` | `-3.350 +- 0.176` | `0.2575 +- 0.0794` |
| EfficientNet-B0 | CheXpert Frontal | 4. Source Platt Scaled (NIH val) | `0.0083 +- 0.0048` | `0.0084 +- 0.0048` | `0.0239 +- 0.0001` | `0.000 +- 0.003` | `0.1163 +- 0.0027` | `0.697 +- 0.051` | `-0.758 +- 0.436` | `0.0161 +- 0.0049` |
| EfficientNet-B0 | CheXpert (90% eval) | 5. Target-10% Platt (20 splits) | `0.0011 +- 0.0000` | `0.0025 +- 0.0005` | `0.0238 +- 0.0000` | `0.004 +- 0.002` | `0.1133 +- 0.0007` | `1.012 +- 0.025` | `0.040 +- 0.090` | — |

![Figure 3: 15-Bin Calibration Reliability & Recalibration Regimes](../results/figures/fig3_calibration_reliability.png)
*Figure 3: Empirical $M=15$-bin reliability diagrams on CheXpert (`seed = 42` curves with 3-seed mean ECE in legends; low-probability zoom `[0, 0.16]` inset) for DenseNet-121 (**A**) and EfficientNet-B0 (**B**), alongside 3-seed mean calibration error across all five regimes (**C**). Affine source Platt scaling resolves intercept bias and lowers CheXpert ECE from `~0.31` to `~0.008` without target labels. Vector files: [`fig3_calibration_reliability.pdf`](../results/figures/fig3_calibration_reliability.pdf), [`fig3_calibration_reliability.svg`](../results/figures/fig3_calibration_reliability.svg).*

---

### 4.3 Operating Threshold Transfer Discordance & Specificity Collapse
Locking thresholds on fresh NIH validation ($N = 13,500$, `139` positives) and deploying unmodified onto Stanford CheXpert ($N = 191,027$) exposed severe **specificity collapse rather than sensitivity loss** ([Table 3](#table-3-source-derived-operating-threshold-transfer-breakdown), [Figure 4](../results/figures/fig4_threshold_transfer_discordance.png)). At the primary $\tau_{\text{sens90}}$ cutoff (`0.1017 +- 0.0425` DenseNet-121; `0.0939 +- 0.0507` EfficientNet-B0), NIH test sensitivity was `0.946` (`[0.929, 0.963]`) and `0.938` (`[0.920, 0.954]`), rising slightly on CheXpert to `0.966` (`[0.962, 0.970]`) and `0.970` (`[0.966, 0.973]`). However, operating specificity collapsed from `0.194 [0.180, 0.208]` to **`0.065 [0.064, 0.066]`** for DenseNet-121 ($\Delta\text{Spec} = +0.129, p = 0.0010$) and from `0.201 [0.188, 0.215]` to **`0.062 [0.061, 0.063]`** for EfficientNet-B0 ($\Delta\text{Spec} = +0.139, p = 0.0010$). Clinically, classifying over `93.5%` of pneumonia-negative radiographs as positive yields a target PPV of just `2.53%` (nearly indistinguishable from background prevalence `2.45%`)—generating severe alarm burden despite meeting sensitivity guarantees. Similar specificity drops occurred across Youden's $J$ (`0.428 -> 0.330` DenseNet-121; `0.504 -> 0.347` EfficientNet-B0; $p = 0.0010$) and $\tau_{\text{sens95}}$ (`0.096 -> 0.018` DenseNet-121; `0.132 -> 0.034` EfficientNet-B0; $p = 0.0010$).

#### Table 3: Source-Derived Operating Threshold Transfer Breakdown
*Operating cutoffs fitted individually per seed on fresh NIH validation (`N = 13,500`) and evaluated unmodified on NIH test and CheXpert (`u_zeros`) with $B = 2,000$ patient-clustered 95% CIs.*

| Architecture | Threshold Rule | NIH Val Cutoff (mean $\pm$ SD) | NIH Test Sens [95% CI] | CheXpert Sens [95% CI] | NIH Test Spec [95% CI] | CheXpert Spec [95% CI] | $\Delta$ Spec (NIH $-$ Chx) | CheXpert PPV / NPV |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DenseNet-121 | Source Sens-90 ($\tau_{\text{sens90}}$) | `0.1017 +- 0.0425` | `0.946 [0.929, 0.963]` | `0.966 [0.962, 0.970]` | `0.194 [0.180, 0.208]` | `0.065 [0.064, 0.066]` | `+0.129 (p=0.0010)` | `2.53% / 98.71%` |
| DenseNet-121 | Source Youden $J$ ($\tau_{\text{Youden}}$) | `0.2334 +- 0.0234` | `0.793 [0.758, 0.829]` | `0.777 [0.767, 0.786]` | `0.428 [0.407, 0.451]` | `0.330 [0.327, 0.333]` | `+0.098 (p=0.0010)` | `2.89% / 98.31%` |
| DenseNet-121 | Source Sens-95 ($\tau_{\text{sens95}}$) | `0.0586 +- 0.0296` | `0.981 [0.971, 0.990]` | `0.992 [0.990, 0.994]` | `0.096 [0.088, 0.105]` | `0.018 [0.018, 0.019]` | `+0.078 (p=0.0010)` | `2.47% / 98.92%` |
| EfficientNet-B0 | Source Sens-90 ($\tau_{\text{sens90}}$) | `0.0939 +- 0.0507` | `0.938 [0.920, 0.954]` | `0.970 [0.966, 0.973]` | `0.201 [0.188, 0.215]` | `0.062 [0.061, 0.063]` | `+0.139 (p=0.0010)` | `2.53% / 98.78%` |
| EfficientNet-B0 | Source Youden $J$ ($\tau_{\text{Youden}}$) | `0.2528 +- 0.0470` | `0.747 [0.708, 0.786]` | `0.765 [0.757, 0.774]` | `0.504 [0.483, 0.526]` | `0.347 [0.345, 0.349]` | `+0.157 (p=0.0010)` | `2.99% / 98.38%` |
| EfficientNet-B0 | Source Sens-95 ($\tau_{\text{sens95}}$) | `0.0606 +- 0.0345` | `0.965 [0.952, 0.976]` | `0.983 [0.981, 0.986]` | `0.132 [0.123, 0.142]` | `0.034 [0.033, 0.034]` | `+0.099 (p=0.0010)` | `2.49% / 98.90%` |

![Figure 4: Empirical Decision Threshold Transfer & Specificity Collapse](../results/figures/fig4_threshold_transfer_discordance.png)
*Figure 4: Empirical sensitivity and specificity swept directly across operating threshold $\tau \in [0.01, 0.60]$ for DenseNet-121 (**A**) and EfficientNet-B0 (**B**) on NIH test vs CheXpert frontal (`3-seed` mean curves). Vertical rules mark source-derived cutoffs ($\tau_{\text{sens95}}$, $\tau_{\text{sens90}}$, $\tau_{\text{Youden}}$), showing target specificity collapsing to `6.5%` / `6.2%` at $\tau_{\text{sens90}}$. Vector files: [`fig4_threshold_transfer_discordance.pdf`](../results/figures/fig4_threshold_transfer_discordance.pdf), [`fig4_threshold_transfer_discordance.svg`](../results/figures/fig4_threshold_transfer_discordance.svg).*

---

### 4.4 CheXpert Label Uncertainty Policy Sensitivity & `PRE-E-001` Outlier Check
Evaluating the same six checkpoints across all four pre-specified CheXpert labeling ontologies revealed substantial sensitivity to uncertainty and blank-label handling ([Table 4](#table-4-chexpert-label-uncertainty-policy-sensitivity--pre-e-001-outlier-check), [Figure 5](../results/figures/fig5_primary_endpoints_and_policies.png)). Moving from primary `u_zeros` (`2.45%` prevalence) to `u_ones` (`10.81%`, mapping `15,981` uncertain reports as positive) raised AUROC slightly (`0.591 -> 0.623` DenseNet-121; `0.608 -> 0.627` EfficientNet-B0) and raised AUPRC linearly with prevalence (`0.036 -> 0.166`; `0.038 -> 0.167`). Dropping uncertain labels (`u_excluded`, `N = 175,046`, `2.67%` prev) improved discrimination modestly (`0.602` / `0.619`). Finally, restricting to `explicitly_labeled_only` (`N = 6,550`, `71.37%` prev)—where clinicians explicitly documented pneumonia presence or absence rather than leaving pneumonia unmentioned—elevated AUROC to `0.700 +- 0.019` (DenseNet-121) and `0.652 +- 0.031` (EfficientNet-B0) and AUPRC to `0.836` / `0.809`. This confirms that much of the apparent cross-domain AUROC deficit reflects label noise and unmentioned/equivocal radiographs within the `168,496` blank records.

In the pre-specified `DEVIATIONS.md` `PRE-E-001` outlier sensitivity check on `efficientnet_b0_2024`, excluding corrupted static-noise rows `68292` and `76123` reduced positives from `4,675` to `4,674` (`191,027 -> 191,025`) with negligible effect on any endpoint under stable logit evaluation: AUROC (`0.5981 -> 0.5983`), AUPRC (`0.0347 -> 0.0347`), Uncalibrated NLL (`0.5645 -> 0.5642`), Uncalibrated $\text{ECE}_{15}$ (`0.3852 -> 0.3852`), and Source Platt NLL (`0.1146 -> 0.1143`).

#### Table 4: CheXpert Label Uncertainty Policy Sensitivity & PRE-E-001 Outlier Check
*Three-seed mean $\pm$ SD (`ddof = 1`) across all four CheXpert uncertainty policies (top) and side-by-side `PRE-E-001` comparison on `efficientnet_b0_2024` (bottom).*

| Policy / Sensitivity | Architecture / Model | Images (Positives) | Image Prev | AUROC | AUPRC | Sens @ $\tau_{\text{sens90}}$ | Spec @ $\tau_{\text{sens90}}$ | Uncal $\text{ECE}_{15}$ | Uncal NLL |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `u_zeros` (Primary) | DenseNet-121 | `191,027 (4,675)` | `2.45%` | `0.591 +- 0.035` | `0.036 +- 0.005` | `0.966 +- 0.011` | `0.065 +- 0.022` | `0.310 +- 0.097` | `0.472 +- 0.151` |
| `u_zeros` (Primary) | EfficientNet-B0 | `191,027 (4,675)` | `2.45%` | `0.608 +- 0.016` | `0.038 +- 0.004` | `0.970 +- 0.029` | `0.062 +- 0.061` | `0.303 +- 0.104` | `0.449 +- 0.132` |
| `u_ones` | DenseNet-121 | `191,027 (20,656)` | `10.81%` | `0.623 +- 0.023` | `0.166 +- 0.016` | `0.978 +- 0.010` | `0.069 +- 0.023` | `0.226 +- 0.097` | `0.511 +- 0.115` |
| `u_ones` | EfficientNet-B0 | `191,027 (20,656)` | `10.81%` | `0.627 +- 0.002` | `0.167 +- 0.004` | `0.973 +- 0.034` | `0.065 +- 0.063` | `0.223 +- 0.097` | `0.497 +- 0.085` |
| `u_excluded` | DenseNet-121 | `175,046 (4,675)` | `2.67%` | `0.602 +- 0.037` | `0.042 +- 0.007` | `0.966 +- 0.011` | `0.069 +- 0.023` | `0.301 +- 0.097` | `0.461 +- 0.149` |
| `u_excluded` | EfficientNet-B0 | `175,046 (4,675)` | `2.67%` | `0.619 +- 0.016` | `0.044 +- 0.004` | `0.970 +- 0.029` | `0.065 +- 0.063` | `0.295 +- 0.105` | `0.440 +- 0.132` |
| `explicitly_labeled_only` | DenseNet-121 | `6,550 (4,675)` | `71.37%` | `0.700 +- 0.019` | `0.836 +- 0.014` | `0.966 +- 0.011` | `0.140 +- 0.034` | `0.358 +- 0.081` | `0.887 +- 0.149` |
| `explicitly_labeled_only` | EfficientNet-B0 | `6,550 (4,675)` | `71.37%` | `0.652 +- 0.031` | `0.809 +- 0.016` | `0.970 +- 0.029` | `0.089 +- 0.050` | `0.353 +- 0.081` | `0.923 +- 0.252` |
| `PRE-E-001` Full Cohort | EfficientNet-B0 (seed `2024`) | `191,027 (4,675)` | `2.45%` | `0.5981` | `0.0347` | — | `Slope = 0.4946` | `0.3852` | `0.5645` (Platt `0.1146`) |
| `PRE-E-001` Excl. `68292 & 76123` | EfficientNet-B0 (seed `2024`) | `191,025 (4,674)` | `2.45%` | `0.5983` | `0.0347` | — | `Slope = 0.5254` | `0.3852` | `0.5642` (Platt `0.1143`) |

![Figure 5: Confirmatory 6-Test BH Family & Label Policy Sensitivity](../results/figures/fig5_primary_endpoints_and_policies.png)
*Figure 5: (**A**) Forest plot of all 6 confirmatory hypothesis tests (`3-seed` mean contrasts with patient-clustered 95% CIs, individual seed markers `42`/`1337`/`2024`, and BH $q$-values). (**B**) CheXpert pneumonia AUROC across all four pre-specified evaluation label policies compared against NIH test references. Vector files: [`fig5_primary_endpoints_and_policies.pdf`](../results/figures/fig5_primary_endpoints_and_policies.pdf), [`fig5_primary_endpoints_and_policies.svg`](../results/figures/fig5_primary_endpoints_and_policies.svg).*

---

### 4.5 Exploratory Demographic & Acquisition Subgroup Stratification
Stratified exploratory evaluation under `u_zeros` ([Table 5](#table-5-exploratory-demographic-and-acquisition-subgroup-stratification), [Figure 6](../results/figures/fig6_subgroups_and_outlier_sensitivity.png)) revealed three hypothesis-generating patterns:
1. **Projection View (`AP` vs `PA`)**: CheXpert `PA` radiographs exhibited roughly double the pneumonia prevalence of `AP` exams (`4.07%` vs `2.15%`) alongside modestly higher AUPRC (`0.061 / 0.059` vs `0.033 / 0.035`), while AUROC differed little (`0.602` vs `0.604` DenseNet-121; `0.615` vs `0.597` EfficientNet-B0).
2. **Age Gradient in Specificity & Overconfidence**: Across `<40`, `40–64`, and `>=65` yr strata, older patients experienced progressive AUROC erosion (`0.602 -> 0.592 -> 0.585` DenseNet-121; `0.620 -> 0.609 -> 0.600` EfficientNet-B0), escalating uncalibrated overconfidence (`ECE_uniform`: `0.255 -> 0.308 -> 0.328`), and severe specificity collapse at fixed $\tau_{\text{sens90}}$—falling from `15.5%` (`<40` yr) to **`3.4%`** (`>=65` yr) in DenseNet-121 and from `12.2%` to **`4.2%`** in EfficientNet-B0.
3. **Patient Sex (`Female` vs `Male`)**: Discrimination was symmetric (`AUROC 0.591` vs `0.592` DenseNet-121; `0.610` vs `0.609` EfficientNet-B0), though male radiographs showed lower fixed-threshold specificity (`5.4% / 5.1%` vs `8.1% / 7.7%`).

#### Table 5: Exploratory Demographic and Acquisition Subgroup Stratification
*Strictly exploratory (`non-confirmatory`) evaluation under CheXpert `u_zeros` (`3-seed` mean $\pm$ SD, `ddof = 1`). Note `17` frontal rows lack `AP/PA` tags and `1` row lacks binary sex.*

| Subgroup Stratum | Architecture | Images (Positives) | Prev | AUROC | AUPRC | Sens @ $\tau_{\text{sens90}}$ | Spec @ $\tau_{\text{sens90}}$ | Uncal $\text{ECE}_{15}$ | Cal Slope |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Projection: `AP` | DenseNet-121 | `161,590 (3,476)` | `2.15%` | `0.602 +- 0.035` | `0.033 +- 0.005` | `0.973 +- 0.007` | `0.057 +- 0.018` | `0.321 +- 0.104` | `0.447 +- 0.110` |
| Projection: `AP` | EfficientNet-B0 | `161,590 (3,476)` | `2.15%` | `0.615 +- 0.015` | `0.035 +- 0.002` | `0.970 +- 0.029` | `0.061 +- 0.063` | `0.307 +- 0.109` | `0.554 +- 0.183` |
| Projection: `PA` | DenseNet-121 | `29,420 (1,198)` | `4.07%` | `0.604 +- 0.014` | `0.061 +- 0.006` | `0.948 +- 0.030` | `0.107 +- 0.058` | `0.248 +- 0.058` | `0.456 +- 0.163` |
| Projection: `PA` | EfficientNet-B0 | `29,420 (1,198)` | `4.07%` | `0.597 +- 0.009` | `0.059 +- 0.005` | `0.970 +- 0.030` | `0.066 +- 0.050` | `0.281 +- 0.078` | `0.426 +- 0.179` |
| Sex: `Female` | DenseNet-121 | `78,875 (1,947)` | `2.47%` | `0.591 +- 0.033` | `0.036 +- 0.004` | `0.955 +- 0.009` | `0.081 +- 0.022` | `0.296 +- 0.086` | `0.379 +- 0.149` |
| Sex: `Female` | EfficientNet-B0 | `78,875 (1,947)` | `2.47%` | `0.610 +- 0.015` | `0.038 +- 0.003` | `0.960 +- 0.035` | `0.077 +- 0.074` | `0.287 +- 0.105` | `0.490 +- 0.162` |
| Sex: `Male` | DenseNet-121 | `112,151 (2,728)` | `2.43%` | `0.592 +- 0.037` | `0.037 +- 0.005` | `0.974 +- 0.012` | `0.054 +- 0.023` | `0.319 +- 0.105` | `0.433 +- 0.186` |
| Sex: `Male` | EfficientNet-B0 | `112,151 (2,728)` | `2.43%` | `0.609 +- 0.018` | `0.039 +- 0.004` | `0.976 +- 0.024` | `0.051 +- 0.052` | `0.314 +- 0.103` | `0.521 +- 0.162` |
| Age: `<40` yr | DenseNet-121 | `26,268 (611)` | `2.33%` | `0.602 +- 0.029` | `0.037 +- 0.006` | `0.919 +- 0.029` | `0.155 +- 0.050` | `0.255 +- 0.052` | `0.415 +- 0.215` |
| Age: `<40` yr | EfficientNet-B0 | `26,268 (611)` | `2.33%` | `0.620 +- 0.025` | `0.039 +- 0.006` | `0.932 +- 0.045` | `0.122 +- 0.071` | `0.254 +- 0.087` | `0.487 +- 0.125` |
| Age: `40-64` yr | DenseNet-121 | `79,586 (1,757)` | `2.21%` | `0.592 +- 0.037` | `0.032 +- 0.005` | `0.962 +- 0.013` | `0.068 +- 0.026` | `0.308 +- 0.094` | `0.390 +- 0.124` |
| Age: `40-64` yr | EfficientNet-B0 | `79,586 (1,757)` | `2.21%` | `0.609 +- 0.012` | `0.035 +- 0.003` | `0.969 +- 0.029` | `0.064 +- 0.064` | `0.300 +- 0.103` | `0.488 +- 0.161` |
| Age: `>=65` yr | DenseNet-121 | `85,173 (2,307)` | `2.71%` | `0.585 +- 0.038` | `0.039 +- 0.006` | `0.982 +- 0.009` | `0.034 +- 0.016` | `0.328 +- 0.113` | `0.413 +- 0.197` |
| Age: `>=65` yr | EfficientNet-B0 | `85,173 (2,307)` | `2.71%` | `0.600 +- 0.018` | `0.041 +- 0.004` | `0.981 +- 0.025` | `0.042 +- 0.055` | `0.320 +- 0.111` | `0.523 +- 0.178` |

![Figure 6: Exploratory Subgroup Stratification & PRE-E-001 Sensitivity](../results/figures/fig6_subgroups_and_outlier_sensitivity.png)
*Figure 6: (**A**) Exploratory subgroup stratification under `u_zeros` showing `3-seed` mean AUROC ($\pm 1\text{ SD}$) and mean specificity at $\tau_{\text{sens90}}$ (in parentheses) across projection, sex, and age bands. (**B**) `PRE-E-001` corrupted-scan sensitivity on `efficientnet_b0_2024` comparing all `191,027` frontal radiographs against excluding static-noise rows `68292` and `76123` (`N = 191,025`). Vector files: [`fig6_subgroups_and_outlier_sensitivity.pdf`](../results/figures/fig6_subgroups_and_outlier_sensitivity.pdf), [`fig6_subgroups_and_outlier_sensitivity.svg`](../results/figures/fig6_subgroups_and_outlier_sensitivity.svg).*

---

### 4.6 Consolidated Hypothesis Testing Summary
The six pre-specified confirmatory hypotheses (`H1`–`H6`) evaluated under Benjamini-Hochberg FDR control ($\alpha = 0.05$) are summarized in [Table 6](#table-6-consolidated-summary-of-primary-hypotheses-and-analytical-findings).

#### Table 6: Consolidated Summary of Primary Hypotheses and Analytical Findings
*Benjamini-Hochberg FDR control across the pre-specified 6-test confirmatory family (`B = 2,000` paired patient-clustered bootstrap iterations, `seed = 42`).*

| Test ID | Pre-Specified Hypothesis | Estimand ($\text{NIH}_{\text{test}} - \text{CheXpert}$) | Raw $p$ ($B=2,000$) | BH $q$-value | Verdict ($\alpha=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| H1 (DenseNet-121) | $\Delta\text{AUROC}$ | `+0.0837 [+0.0588, +0.1082]` | `0.0010` | `0.0015` | **REJECT $H_0$ (Significant)** |
| H2 (DenseNet-121) | $\Delta\text{Calibration Slope}$ | `+0.1755 [+0.0767, +0.2806]` | `0.0010` | `0.0015` | **REJECT $H_0$ (Significant)** |
| H3 (DenseNet-121) | $\Delta\text{Sensitivity}$ at $\tau_{\text{sens90}}$ | `-0.0199 [-0.0379, -0.0027]` | `0.0200` | `0.0240` | **REJECT $H_0$ (Significant)** |
| H4 (EfficientNet-B0) | $\Delta\text{AUROC}$ | `+0.0741 [+0.0494, +0.0995]` | `0.0010` | `0.0015` | **REJECT $H_0$ (Significant)** |
| H5 (EfficientNet-B0) | $\Delta\text{Calibration Slope}$ | `+0.0168 [-0.0760, +0.1129]` | `0.6857` | `0.6857` | FAIL TO REJECT (NS) |
| H6 (EfficientNet-B0) | $\Delta\text{Sensitivity}$ at $\tau_{\text{sens90}}$ | `-0.0322 [-0.0503, -0.0156]` | `0.0010` | `0.0015` | **REJECT $H_0$ (Significant)** |

---

## 5. Discussion & Clinical Translation

### 5.1 Why Internal Validation Misleads Clinical Readiness
Our empirical findings demonstrate that internal validation provides a falsely reassuring view of cross-hospital generalization even when crude disease prevalence scarcely changes (`2.16%` NIH test vs `2.45%` CheXpert `u_zeros`). On transfer to Stanford CheXpert, discrimination dropped by `0.084` AUROC in DenseNet-121 and `0.074` AUROC in EfficientNet-B0 (`AUROC 0.591` and `0.608`, `q = 0.0015`). Moreover, restricting evaluation strictly to explicitly labeled CheXpert studies (`N = 6,550`, `71.37%` positive) restored AUROC toward `0.700` for DenseNet-121, showing that institutional differences in unmentioned (`88.2%` blank) and equivocal (`8.4%` uncertain) clinical reporting interact strongly with covariate shift.

### 5.2 Class-Weighting Bias and Why Source Platt Succeeds Where Temperature Scaling Fails
A central mechanistic insight of this study is the stark divergence between post-hoc **temperature scaling** [16] and **Platt scaling** [19]. Training with inverse-frequency `pos_weight` shifts uncalibrated predicted probabilities upward (~`0.30–0.33` average vs `~2.2%–2.5%` true prevalence), corresponding to a large negative logit intercept bias (`a ~ -3.4`). Because scalar temperature scaling ($z / T$) fixes logits at $0 \mapsto 0.5$, optimizing $T$ on validation cannot relocate the intercept—and in models where validation calibration slopes vary, small $T$ (`0.26–0.53`) fails to eliminate massive probability overconfidence on CheXpert (`ECE = 0.233–0.235`). By contrast, two-parameter source Platt scaling learns both the slope attenuation and the prevalence-correcting intercept (`b ~ -3.6` to `-3.7`), reducing CheXpert `ECE` by `>97%` (`to ~0.008`) without touching a single CheXpert label. Because Target-10% Platt refitting further improves slope alignment (`1.066` / `1.012`) and reaches `ECE = 0.0011`, source Platt provides an immediately deployable zero-target-label safeguard whenever institutional prevalence is stable, whereas lightweight target-site recalibration remains necessary to correct domain-specific slope compression (especially in DenseNet-121).

### 5.3 Algorithmic Alarm Fatigue Through Specificity Collapse Rather Than Sensitivity Drop
Contrary to the intuitive expectation that out-of-domain shift always causes missed cases (sensitivity collapse), our results show that uncalibrated overconfident scores paired with acuity-heavy inpatient populations (`84.6%` AP bedside exams; `>=65` yr elderly stratum) actually pushed sensitivity at NIH-derived $\tau_{\text{sens90}}$ slightly **upward** (`96.6%` and `97.0%`) at the catastrophic expense of **specificity (`6.5%` DenseNet-121; `6.2%` EfficientNet-B0)**. In clinical deployment on `191,027` frontal radiographs, flagging over `93%` of healthy/non-pneumonia examinations would flood radiology queues with false positives (`PPV = 2.53%`), creating acute alarm fatigue. Safe clinical triage therefore requires locking operating thresholds only *after* site-calibrated probability transformation and directly auditing target-site specificity and age-stratified false-positive burden.

### 5.4 Open Scope Boundaries: Reverse Transfer and Quantitative XAI Auditing
Visual explanation methods (Grad-CAM [13], sanity checks [14], insertion/deletion [15]) were intentionally omitted from quantitative claims here because no forward Grad-CAM faithfulness or shortcut-energy audit was executed during `oldckpt_real_v1` inference on real CheXpert data. Similarly, bidirectional training (`CheXpert -> NIH`) and training-time uncertainty ablations (`U-Ignore` trained on CheXpert) remain outside this evaluation scope. Maintaining strict boundaries between observed real-inference evidence and unexecuted experiments protects clinical interpretation against unverified extrapolation.

---

## 6. Regulatory Perspectives

Our findings directly inform current regulatory frameworks established by the US Food & Drug Administration (FDA), European Medicines Agency (EMA), and Health Canada for **Software as a Medical Device (SaMD)**:
1. **Total Product Lifecycle (TPLC) External Evidence**: Internal retrospective test metrics fail to predict out-of-domain discrimination (`-7.4` to `-8.4` AUROC points) and specificity loss (`~6%` at rule-out cutoffs). Multi-center external benchmarking with patient-clustered variance estimation should be mandatory prior to clearance.
2. **Predetermined Change Control Plans (PCCP) for Site Recalibration**: Rather than freezing raw uncalibrated probability outputs or single-parameter temperature scalers across institutions, PCCPs should explicitly authorize verified affine recalibration (`Platt` parameters) monitored continuously against local prevalence and `NLL`/`ECE` drift.
3. **Good Machine Learning Practice (GMLP) Safeguards**:
   - Enforcing zero-leakage patient-disjoint evaluation splits and cluster-aware CIs (`B >= 2,000`).
   - Preserving numerically stable logit pipelines (`PRE-E-001`) against extreme finite corrupted inputs (`static noise` JPEGs).
   - Disclosing age-, sex-, and projection-stratified false-positive operating behavior under explicit label-uncertainty policies (`u_zeros`, `u_ones`, `u_excluded`).

---

## 7. Limitations & Open Challenges

1. **Automated NLP Ground-Truth Noise**: Both NIH (`ChestX-ray14`) and Stanford CheXpert depend on automated rule-based radiology report extractors [9, 10]. In CheXpert, `88.20%` of frontal studies lack explicit pneumonia mentions (`blank/NaN`); comparing `u_zeros` (`AUROC 0.591–0.608`) against `explicitly_labeled_only` (`0.652–0.700`) proves that ambiguous and unmentioned reporting accounts for substantial measured uncertainty.
2. **Non-Square Geometric Margin Trimming**: Fixed `Resize(256)` + `CenterCrop(224)` preprocessing trims horizontal or vertical borders more aggressively on variable-aspect-ratio CheXpert images than on natively square NIH images.
3. **Fixed Validation-Threshold Variance**: Operating thresholds ($\tau_{\text{sens90}}$, $\tau_{\text{Youden}}$, $\tau_{\text{sens95}}$) were estimated on `13,500` NIH validation scans (`139` positives) and held fixed inside bootstrap replicates; propagating validation threshold estimation variance would slightly widen threshold-dependent CIs.
4. **Unidirectional Scope & Missing CheXpert Training / XAI Audits**: This evaluation quantifies zero-shot `NIH -> CheXpert` transfer for six locked NIH models; symmetric `CheXpert -> NIH` transfer, training-time uncertainty policies, and empirical CheXpert Grad-CAM shortcut localization remain to be executed under an equally strict pre-specified protocol.
5. **Decoded Corrupted-Content Scans**: Label-blind QC detected two non-anatomical static-noise CheXpert scans (`rows 68292, 76123`) triggering extreme finite logits in one seed (`efficientnet_b0_2024`), demonstrating that PIL decode checks alone cannot filter corrupted clinical content—making `float64` logit-space metrics (`PRE-E-001`) indispensable.

---

## 8. Reproducibility & Code Availability

All analysis code, hash-stamped provenance verification utilities, pre-specified protocols, and publication scripts are open-sourced under the MIT License:
- **Repository**: `https://github.com/aditya-sharma2OO5/PneumoShift`
- **Pre-Specified Protocol & Deviation Log**: [`ANALYSIS_PLAN_TRANSFER.md`](../ANALYSIS_PLAN_TRANSFER.md) (`v2.2.3`, tag `prov-hardening-v3`) and [`DEVIATIONS.md`](../DEVIATIONS.md) (`PRE-E-001`)
- **Saved Transfer Outputs**: `outputs/transfer_oldckpt_real_v1/` (`results.json`, stamped CSVs, `paper_tables/`, and `figures/`)
- **Primary Analysis, Paper Table, & Empirical Figure Commands**:
  ```bash
  # 1. Execute audited transfer statistical pipeline (B=2,000, M=15, PRE-E-001)
  python scripts/analyze_transfer.py --preds-dir <OLD_CKPT_PREDS_DIR> --expected-commit d79bc756ba9a60d628674f2af1a683bbc25e2e6e

  # 2. Export publication tables (CSV, JSON, Markdown, LaTeX booktabs)
  python scripts/export_paper_tables.py

  # 3. Render empirical publication Figures 1-6 (PDF, SVG, 300 DPI PNG)
  python scripts/generate_transfer_figures.py --preds-dir <OLD_CKPT_PREDS_DIR> --expected-commit d79bc756ba9a60d628674f2af1a683bbc25e2e6e
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
19. Platt JC. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*. MIT Press; 1999:61-74.
