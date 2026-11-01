# PneumoShift

**Beyond Accuracy: Evaluating the Robustness and Explainability of Deep Learning Models for Pneumonia Detection Across Chest X-Ray Datasets**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. Overview & Research Objectives

PneumoShift is a reproducible research framework designed to rigorously evaluate deep learning models for pneumonia detection when evaluated under dataset shift between major clinical chest radiograph cohorts (NIH ChestX-ray14 and Stanford CheXpert).

### Core Research Questions
- **RQ1 (Discrimination Shift)**: How much does pneumonia-detection performance (AUROC, AUPRC, fixed-operating-point metrics) degrade when models trained on one chest X-ray cohort are evaluated on another?
- **RQ2 (Calibration & Threshold Transfer)**: How do probability calibration (ECE, MCE, Brier score) and source-selected decision thresholds degrade under domain shift?
- **RQ3 (Explainability Faithfulness & Stability)**: Do Grad-CAM visual explanations remain faithful, stable, and anatomically plausible under domain shift, and does saliency quality degrade in tandem with predictive accuracy?
- **RQ4 (Architectural Influence)**: Does the inductive bias of different network families (DenseNet121 vs EfficientNet-B0) influence generalization, calibration stability, and shortcut reliance?

### Tested Hypotheses
- **H1**: Cross-dataset AUROC is strictly lower than in-domain AUROC across all models.
- **H2**: Expected Calibration Error (ECE) escalates under dataset shift even when AUROC drops only modestly.
- **H3**: Source-optimal classification thresholds transfer poorly to target domains, causing severe sensitivity/specificity imbalances.
- **H4**: Saliency attributions shift towards non-lung shortcut features (borders, lateral markers, image artifacts, support devices) when tested out-of-domain.

---

## 2. Repository Layout

```text
PneumoShift/
├── configs/                # Experiment configurations (YAML)
│   ├── default.yaml        # Base experiment configuration
│   ├── smoke_test.yaml     # Fast end-to-end smoke test config
│   ├── data/               # Dataset configurations (NIH, CheXpert, Synthetic)
│   ├── model/              # Model backbone configs (DenseNet121, EfficientNet)
│   ├── train/              # Training & optimizer parameters
│   ├── eval/               # Evaluation & bootstrap metrics
│   └── xai/                # Grad-CAM, faithfulness & shortcut analysis
├── docs/                   # Documentation & protocol notes
│   ├── ASSUMPTIONS.md      # Methodological decisions & assumptions
│   ├── METHODS.md          # Formal statistical & experimental protocol
│   └── DATA_GUIDE.md       # Dataset acquisition & folder organization
├── notebooks/              # Analysis & interactive exploratory notebooks
│   ├── 01_data_audit.ipynb
│   ├── 02_results_overview.ipynb
│   └── 03_xai_gallery.ipynb
├── results/                # Output artifacts (CSVs, figures, tables, logs)
├── scripts/                # Command-line entry points
│   ├── prepare_splits.py   # Patient-stratified split generator
│   ├── train.py            # Model training entry point
│   ├── evaluate.py         # Evaluation & calibration benchmark
│   ├── run_xai.py          # Grad-CAM, faithfulness & stability pipeline
│   ├── aggregate_results.py# Master tables & statistical significance
│   └── make_figures.py     # Publication-grade figures
├── src/pneumoshift/        # Core library source code
│   ├── data/               # Data loaders, harmonization, splits, transforms
│   ├── models/             # Architecture factory and classifier heads
│   ├── training/           # Trainer, weighted losses, callbacks
│   ├── evaluation/         # Metrics, calibration, threshold transfer, probes
│   ├── xai/                # Grad-CAM, faithfulness, sanity checks, lung masks
│   ├── viz/                # Plotting utilities for ROC/PR, ECE, galleries
│   └── utils/              # Seeding, logging, I/O helpers
├── tests/                  # Unit tests and CI leakage checks
├── Makefile                # Standardized developer commands
├── pyproject.toml          # Package configuration and tool settings
└── requirements.txt        # Pinned project dependencies
```

---

## 3. Quickstart & Installation

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/aditya-sharma2OO5/PneumoShift.git
cd PneumoShift

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Smoke Test Pipeline (No Real Data Required)
A synthetic generator creates small dummy radiographs with ground-truth vectors to test the entire data loader, training, evaluation, and Grad-CAM suite:
```bash
# Generate synthetic dataset and verify zero-leakage splits
python scripts/prepare_splits.py --smoke_test

# Run end-to-end smoke test
pytest tests/test_smoke.py
```

---

## 4. Phase-by-Phase Roadmap

- [x] **P0 Setup**: Repository scaffolding, modular structure, configs, CI tests, and synthetic smoke test pipeline.
- [x] **P1 Data**: Frontal filtering, label harmonization (8 shared classes), patient-partitioned splits with zero overlap, and distribution audit.
- [ ] **P2 Training**: In-domain DenseNet121 baseline models on NIH ChestX-ray14 and Stanford CheXpert.
- [ ] **P3 Cross-Dataset Evaluation**: Cross-domain generalization gap, ECE/MCE calibration, temperature scaling, oracle recalibration, threshold transfer, and penultimate feature probes.
- [ ] **P4 Architecture & Uncertainty Ablation**: EfficientNet-B0 evaluation, multi-seed replication (seeds 42, 1337, 2024), and CheXpert uncertainty ablations (U-Zeros vs U-Ones vs U-Ignore).
- [ ] **P5 Explainability Suite**: Grad-CAM attribution, deletion/insertion faithfulness curves, parameter randomization sanity checks, stability across seeds, BBox pointing game, and lung-field plausibility.
- [ ] **P6 Results Aggregation**: Master summary tables with patient-bootstrap 95% CIs and FDR-adjusted hypothesis tests.
- [ ] **P7 Paper Deliverables**: Automated markdown results report, publication figures, and reproducibility checklist.

---

## 5. Disclaimer

This software is developed strictly for academic machine learning research and reproducible benchmark evaluation. It is **not** certified as a clinical medical device and should not be used for primary patient diagnosis or clinical decision-making.

