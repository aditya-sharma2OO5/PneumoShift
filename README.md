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
├── docs/                   # Documentation, manuscript & reports
│   ├── MANUSCRIPT.md       # Camera-ready academic research manuscript
│   ├── DATA_GUIDE.md       # Dataset acquisition & folder organization
│   ├── METHODS.md          # Formal statistical & experimental protocol
│   ├── ASSUMPTIONS.md      # Methodological decisions & assumptions
│   ├── PHASE3_REPORT.md    # Cross-domain shift evaluation report
│   ├── PHASE4_REPORT.md    # Architecture & uncertainty ablation report
│   ├── PHASE5_REPORT.md    # Explainability & shortcut audit report
│   └── PHASE6_REPORT.md    # Master results aggregation & meta-analysis
├── reports/                # Standalone interactive deliverables
│   └── gallery.html        # Zero-dependency interactive HTML dashboard
├── results/                # Output artifacts (CSVs, figures, tables, logs)
│   ├── figures/            # Canonical Figures 1-6 (.pdf, .svg, .png 300 DPI)
│   └── tables/master/      # Master Tables 1-6 (.csv, .json, .md, .tex)
├── scripts/                # Command-line entry points
│   ├── reproduce_all.py    # Master one-click end-to-end reproduction CLI
│   ├── setup_real_data.py  # Automated dataset ingestion & extraction CLI
│   ├── generate_figures.py # Publication figures rendering engine
│   ├── generate_gallery.py # Standalone HTML dashboard generator
│   ├── verify_phase7_gate.py# Automated Phase 7 Quality Gate CLI
│   ├── prepare_splits.py   # Patient-stratified split generator
│   ├── train.py            # Model training entry point
│   ├── evaluate_cross_domain.py # Unified Phase 3 evaluation CLI
│   ├── run_phase4_ablations.py # Unified Phase 4 ablations CLI
│   ├── run_phase5_xai.py   # Unified Phase 5 explainability CLI
│   └── aggregate_results.py# Master tables & statistical meta-analysis
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
- [x] **P2 Training**: In-domain DenseNet121 baseline models on NIH ChestX-ray14 and Stanford CheXpert.
- [x] **P3 Cross-Dataset Evaluation**: Cross-domain generalization gap ($-0.107$ AUROC on Pneumonia), 4-regime ECE calibration and oracle recalibration ($48\%$ recovery on 10% `target_cal`), threshold transfer collapse, penultimate feature probes (0.985 AUROC), and subgroup confounders (see [`docs/PHASE3_REPORT.md`](docs/PHASE3_REPORT.md)).
- [x] **P4 Architecture & Uncertainty Ablation**: EfficientNet-B0 comparison (generalization gap narrowing, $D=1280$ vs $1024$ linear domain probe, MMD shift), CheXpert label uncertainty ablations (U-Zeros vs U-Ones vs U-Ignore, over-prediction ratio, specificity collapse), and 3-seed statistical significance replication (seeds 42, 1337, 2024; paired t-test and Wilcoxon; see [`docs/PHASE4_REPORT.md`](docs/PHASE4_REPORT.md)).
- [x] **P5 Explainability Suite**: Multi-architecture Grad-CAM attribution, deletion/insertion faithfulness curves (FDR ratio), Adebayo parameter randomization sanity checks (94.4% collapse), multi-seed attribution reproducibility (seeds 42, 1337, 2024), ground-truth BBox pointing game (82.5% in-domain vs 38.0% transferred), and anatomical lung-field plausibility vs peripheral shortcut escalation (+288.6% border/marker energy; see [`docs/PHASE5_REPORT.md`](docs/PHASE5_REPORT.md)).
- [x] **P6 Results Aggregation**: Consolidated results aggregation, patient-clustered non-parametric bootstrap uncertainty quantification ($B=1000, 95\%$ CIs), Benjamini-Hochberg False Discovery Rate control ($q=0.05$), and 6 master publication tables exported across 4 formats (.csv, .json, .md, .tex with booktabs) (see [`docs/PHASE6_REPORT.md`](docs/PHASE6_REPORT.md)).
- [x] **P7 Paper Deliverables**: Complete publication deliverables package: 6 canonical publication figures in dual vector (`.pdf`, `.svg`) and 300 DPI raster (`.png`), zero-dependency interactive standalone HTML dashboard ([`reports/gallery.html`](reports/gallery.html)), camera-ready research manuscript ([`docs/MANUSCRIPT.md`](docs/MANUSCRIPT.md)), real dataset onboarding CLI & user guide ([`docs/DATA_GUIDE.md`](docs/DATA_GUIDE.md)), master one-click reproduction CLI ([`scripts/reproduce_all.py`](scripts/reproduce_all.py)), and automated Phase 7 Quality Gate ([`scripts/verify_phase7_gate.py`](scripts/verify_phase7_gate.py)).

---

## 5. Disclaimer

This software is developed strictly for academic machine learning research and reproducible benchmark evaluation. It is **not** certified as a clinical medical device and should not be used for primary patient diagnosis or clinical decision-making.

