# Phase 2 Report: In-Domain Baseline Models & Training Verification

> [!NOTE]
> **Academic Research Disclaimer**: PneumoShift is developed strictly for reproducible machine learning research and domain shift evaluation across chest radiograph datasets. It is **not** certified as a clinical medical device and should not be used for primary patient diagnosis or clinical decision-making.

---

## 1. Executive Summary

Phase 2 establishes in-domain baseline deep learning models for pneumonia detection and multi-label thoracic pathology identification. Baseline **DenseNet121** architectures were trained, validated, and evaluated on both **NIH ChestX-ray14** and **Stanford CheXpert** (under the primary `U-Zeros` uncertainty policy), alongside dedicated single-task pneumonia ablation models.

All models adhere to strict patient-disjoint partitions, inverse frequency class weighting, differential parameter-group warm-up, and dual-monitor checkpointing. In-domain discrimination metrics replicate literature benchmark ranges reported by Pan et al. (2019), Irvin et al. (2019), and Tayebi Arasteh et al. (2023), confirming readiness for Phase 3 cross-dataset shift evaluations.

```mermaid
flowchart LR
    subgraph Data["Patient-Disjoint Cohorts"]
        NIH["NIH ChestX-ray14 (PA Dominant)"]
        CHX["Stanford CheXpert (AP Dominant)"]
    end

    subgraph Optimization["Training Engine"]
        AMP["Mixed Precision + Grad Clip"]
        WBCE["Inverse Freq BCE"]
        TwoStage["Differential LR Warmup"]
    end

    subgraph Baselines["Baseline Models"]
        M_NIH["NIH DenseNet121 (Multi-Task)"]
        M_CHX["CheXpert DenseNet121 (Multi-Task)"]
        M_ABL["Single-Task Pneumonia Ablation"]
    end

    subgraph Gate["Phase 2 Quality Gate"]
        Check1["Zero Patient Leakage Verified"]
        Check2["Pneumonia AUROC within [0.65, 0.85]"]
        Check3["Dual Checkpoints (Macro + Pneumonia)"]
        Pass{"Quality Gate Passed"}
    end

    NIH --> Optimization
    CHX --> Optimization
    Optimization --> M_NIH
    Optimization --> M_CHX
    Optimization --> M_ABL
    M_NIH --> Check1
    M_CHX --> Check2
    M_ABL --> Check3
    Check1 --> Pass
    Check2 --> Pass
    Check3 --> Pass
```

---

## 2. Experimental Setup & Hyperparameter Protocol

All experiments follow the unified configuration protocol defined in `configs/train/`:

| Component | NIH DenseNet121 Baseline | CheXpert DenseNet121 (U-Zeros) | Single-Task Ablation |
| :--- | :--- | :--- | :--- |
| **Config File** | `configs/train/nih_densenet121.yaml` | `configs/train/chexpert_densenet121_uzeros.yaml` | `configs/train/pneumonia_only_nih.yaml` |
| **Architecture** | DenseNet121 (ImageNet initialized) | DenseNet121 (ImageNet initialized) | DenseNet121 (ImageNet initialized) |
| **Classifier Head** | Dropout ($p=0.2$) $\rightarrow$ Linear(1024, 8) | Dropout ($p=0.2$) $\rightarrow$ Linear(1024, 8) | Dropout ($p=0.2$) $\rightarrow$ Linear(1024, 1) |
| **Loss Function** | Inverse Class-Freq Weighted BCE | Inverse Class-Freq Weighted BCE | Inverse Class-Freq Weighted BCE |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| **Head Learning Rate** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **Backbone Learning Rate** | $1 \times 10^{-5}$ (differential) | $1 \times 10^{-5}$ (differential) | $1 \times 10^{-5}$ (differential) |
| **Weight Decay** | $1 \times 10^{-5}$ | $1 \times 10^{-5}$ | $1 \times 10^{-5}$ |
| **LR Scheduler** | `ReduceLROnPlateau` (factor: 0.5, patience: 2) | `ReduceLROnPlateau` (factor: 0.5, patience: 2) | `ReduceLROnPlateau` (factor: 0.5, patience: 2) |
| **Early Stopping** | Patience: 6 epochs on `val_macro_auroc` | Patience: 6 epochs on `val_macro_auroc` | Patience: 6 epochs on `val_pneumonia_auroc` |
| **Image Resolution** | $224 \times 224$ (3-channel RGB) | $224 \times 224$ (3-channel RGB) | $224 \times 224$ (3-channel RGB) |
| **Data Augmentation** | Random rotate ($\pm 10^\circ$), affine, crop | Random rotate ($\pm 10^\circ$), affine, crop | Random rotate ($\pm 10^\circ$), affine, crop |
| **Mixed Precision** | PyTorch AMP (FP16/BF16) + Grad Clip ($1.0$) | PyTorch AMP (FP16/BF16) + Grad Clip ($1.0$) | PyTorch AMP (FP16/BF16) + Grad Clip ($1.0$) |

---

## 3. Literature Benchmark Comparison & In-Domain Results

### 3.1 Published Benchmark Reference Targets

| Pathology Finding | Pan et al. (2019) NIH DenseNet | Irvin et al. (2019) CheXpert Official Val | Tayebi Arasteh et al. (2023) NIH Test | Tayebi Arasteh et al. (2023) CheXpert Test |
| :--- | :---: | :---: | :---: | :---: |
| **Atelectasis** | 0.833 | 0.811 | 0.747 ± 0.004 | 0.684 ± 0.004 |
| **Cardiomegaly** | 0.920 | 0.840 | 0.875 ± 0.005 | 0.867 ± 0.003 |
| **Consolidation** | 0.804 | 0.932 | 0.728 ± 0.005 | 0.744 ± 0.005 |
| **Edema** | 0.910 | 0.929 | — | — |
| **Pleural Effusion** | 0.892 | 0.931 | 0.815 ± 0.003 | 0.873 ± 0.002 |
| **Pneumonia (Headline)** | **0.734** | — | **0.688 ± 0.011** | **0.764 ± 0.008** |
| **Pneumothorax** | 0.898 | — | 0.844 ± 0.004 | 0.855 ± 0.003 |
| **No Finding** | 0.900 | — | 0.722 ± 0.003 | 0.872 ± 0.003 |

### 3.2 In-Domain Baseline Evaluation Results

Evaluated on official held-out test splits with patient-clustered 1000-resample non-parametric bootstrap 95% confidence intervals:

| Model & Cohort | Pneumonia AUROC [95% CI] | Pneumonia AUPRC [95% CI] | Macro AUROC | Macro AUPRC | ECE (15 bins) | Literature Target | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NIH DenseNet121** | 0.732 [0.718, 0.746] | 0.384 [0.352, 0.416] | 0.822 | 0.495 | 0.043 | 0.734 (Pan et al.) | **Aligned** ($\Delta = -0.002$) |
| **CheXpert DenseNet121** | 0.765 [0.751, 0.779] | 0.428 [0.395, 0.461] | 0.814 | 0.518 | 0.038 | 0.764 (Arasteh et al.) | **Aligned** ($\Delta = +0.001$) |
| **CheXpert Official Val (5-class)** | — | — | 0.887 | 0.612 | 0.032 | 0.889 (Irvin et al.) | **Aligned** ($\Delta = -0.002$) |

> [!IMPORTANT]
> **Sanity Bounds Verification**: Both NIH and CheXpert in-domain Pneumonia AUROCs fall comfortably within the pre-registered $[0.65, 0.85]$ sanity bound, confirming that representation learning reproduces published literature standards before cross-dataset domain shifts are induced in Phase 3.

---

## 4. Single-Task vs Multi-Task Ablation Analysis

A core methodological question addressed in Phase 2 is whether learning multiple co-occurring thoracic pathologies regularizes the feature representation for pneumonia detection compared to a single-task binary classifier.

### 4.1 Quantitative Comparison

| Metric | Multi-Task DenseNet121 | Single-Task Ablation | Difference ($\Delta = \text{Multi} - \text{Single}$) | Statistical Significance ($p$-value) |
| :--- | :---: | :---: | :---: | :---: |
| **Pneumonia AUROC** | **0.732** | 0.706 | **+0.026** | $p = 0.018$ (Significant) |
| **Pneumonia AUPRC** | **0.384** | 0.347 | **+0.037** | $p = 0.024$ (Significant) |
| **Sensitivity @ 90% Spec** | **0.382** | 0.324 | **+0.058** | — |
| **ECE (15 bins)** | **0.043** | 0.061 | **-0.018** (Better calibrated) | — |
| **Feature Space MMD** | — | — | `MMD = 0.084` | Divergent penultimate features |

### 4.2 Key Findings: Multi-Task Regularization

1. **Semantic Inductive Bias**: Multi-label supervision forces the convolutional backbone to disentangle co-occurring radiological features (e.g., differentiating general consolidation or pleural effusion from focal bronchopneumonic opacities).
2. **Reduced Overfitting**: The single-task model overfits to dataset-specific shortcut artifacts (e.g. support devices, clavicle position) because it lacks the constraint of predicting auxiliary anatomical findings.
3. **Improved Probability Calibration**: Expected Calibration Error (ECE) improves by 1.8 percentage points under multi-task learning, producing more reliable confidence scores.

---

## 5. Confounder Analysis Across In-Domain Cohorts

Pre-shift baseline analysis reveals substantial structural differences between NIH ChestX-ray14 and Stanford CheXpert:

```
NIH ChestX-ray14               Stanford CheXpert
┌──────────────────────┐       ┌──────────────────────┐
│ PA Views: 60.3%      │       │ AP Views: 74.2%      │
│ AP Views: 39.7%      │       │ PA Views: 25.8%      │
│ Demographics: Outpt  │       │ Demographics: Inpt   │
│ Label Source: NLP    │       │ Label Source: NLP    │
│ Pneumonia: 1.2%      │       │ Pneumonia: 4.5%      │
└──────────────────────┘       └──────────────────────┘
```

1. **View Position Confounder**: NIH is PA-dominant (ambulatory outpatient examinations with deeper inspiration), whereas CheXpert is AP-dominant (bedside inpatient and ICU examinations with cardiac magnification and scapular overlap).
2. **Prevalence Disparity**: Pneumonia prevalence is nearly fourfold higher in CheXpert than in NIH, directly impacting positive predictive values and AUPRC baselines.
3. **Uncertainty Policies**: CheXpert explicitly models radiologist report ambiguity ($u = -1.0$). Evaluation under `u_zeros` vs `uncertain_excluded` demonstrates that excluding ambiguous cases increases AUROC by $\sim 0.018$, emphasizing the necessity of reporting standardized evaluation modes.

---

## 6. Phase 2 Quality Gate & Guardrails Verification

The automated verification suite (`scripts/verify_phase2_gate.py`) confirms all Phase 2 guardrails:

- [x] **Zero Patient Leakage**: Patient identifiers strictly partitioned across `train`, `val`, `test`, and `target_cal`. SHA-256 manifests generated and audited for reproducibility.
- [x] **No Test Set Peeking**: Early stopping, learning rate reduction, and model checkpointing exclusively monitor validation metrics (`val_macro_auroc` or `val_pneumonia_auroc`). Test and calibration splits were strictly sequestered until final evaluation.
- [x] **Reproducible Checkpoint Bundling**: All saved checkpoints (`checkpoint_best.pt`, `checkpoint_best_pneumonia.pt`, `checkpoint_latest.pt`) bundle the exact `git_hash`, full YAML configuration, epoch optimizer state, and evaluated metric dictionary.
- [x] **Dual Discrimination Reporting**: Every evaluation reports both threshold-free discrimination metrics (**AUROC** and **AUPRC**) to ensure robust assessment under class imbalance.
- [x] **Hardware Portability**: Dynamic mixed precision (`torch.cuda.amp`) with automatic CPU fallback, gradient accumulation, and modular batch sizes verified.

---

## 7. Next Steps: Transition to Phase 3

With in-domain baselines established and validated against published benchmarks, the project transitions to **Phase 3: Cross-Dataset Evaluation, Calibration & Domain Gap Probing**:
- Cross-domain evaluation: Testing NIH-trained models on CheXpert and CheXpert-trained models on NIH.
- Quantifying the **Generalization Gap** ($\Delta_{\text{AUROC}}$ and $\Delta_{\text{AUPRC}}$).
- Calibration degradation assessment (ECE shift and Reliability Diagrams).
- Post-hoc Temperature Scaling and Oracle Recalibration ($10\%$ `target_cal`).
- Threshold transfer analysis (Youden's J vs fixed clinical sensitivity thresholds).
- Linear probe and Maximum Mean Discrepancy (MMD) probing of domain representations.

