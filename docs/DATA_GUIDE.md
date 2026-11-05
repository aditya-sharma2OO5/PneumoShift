# PneumoShift Data Guide: Acquisition, Organization & Real Dataset Ingestion Manual

> [!NOTE]
> **Academic Research Notice**: PneumoShift is designed for rigorous domain shift benchmarking across multi-center chest radiographs. Models can be evaluated in two scalable modes:
> 1. **Synthetic Verification Mode (`--smoke_test`)**: Zero external downloads required; executes 100% of mathematical and algorithmic pipelines in seconds.
> 2. **Real Dataset Mode**: Ingests real hospital cohorts from `NIH ChestX-ray14` (~45 GB) and `Stanford CheXpert` (~11 GB) with automated zero-leakage patient partitioning.

---

## 1. Storage & Hardware Budgeting

Before acquiring raw radiograph repositories, ensure your local filesystem or external storage meets the recommended disk allocations:

| Dataset / Execution Mode | Raw Archive Size | Extracted Size | Recommended Free Space | Target Hardware |
| :--- | :---: | :---: | :---: | :--- |
| **NIH ChestX-ray14 (Full)** | 42.1 GB (12 ZIPs) | ~45.0 GB ($112,120$ PNGs) | $\ge 90\text{ GB}$ | Single / Multi-GPU ($\ge 8\text{GB}$ VRAM) |
| **Stanford CheXpert-v1.0-small** | 10.8 GB (1 ZIP) | ~11.5 GB ($224,316$ JPGs) | $\ge 25\text{ GB}$ | Single / Multi-GPU ($\ge 8\text{GB}$ VRAM) |
| **Both Full Cohorts (Combined)** | ~53.0 GB | ~56.5 GB | $\ge 130\text{ GB}$ | Dedicated SSD Recommended |
| **Curated Subsample (`--subsample`)** | Shared with raw | ~1.5 GB ($20,000$ images) | $\ge 15\text{ GB}$ | Fast turnaround on laptops / single GPUs |
| **Synthetic Smoke Test (`--smoke_test`)**| 0.0 MB | ~2.0 MB (mock tensors) | $\ge 100\text{ MB}$ | Continuous Integration / CPU |

---

## 2. Directory Hierarchy Layout

PneumoShift expects raw datasets and generated patient splits structured according to the canonical hierarchy below:

```text
data/
├── nih_cxr/                                       # NIH ChestX-ray14 root
│   ├── images/                                    # Extracted 112,120 PNG images
│   │   ├── 00000001_000.png
│   │   ├── 00000001_001.png
│   │   └── ...
│   ├── Data_Entry_2017_v2020.csv                  # Official NIH metadata table
│   ├── train_val_list.txt                         # Official NIH training partition list
│   ├── test_list.txt                              # Official NIH held-out test list
│   └── BBox_List_2017.csv                         # Radiologist pneumonia bounding boxes
│
├── chexpert/                                      # Stanford CheXpert root
│   └── CheXpert-v1.0-small/                       # Downsampled standard resolution cohort
│       ├── train.csv                              # CheXpert multi-label training metadata
│       ├── valid.csv                              # Official radiologist-consensus validation
│       ├── train/                                 # Patient subfolders (patient00001...)
│       └── valid/                                 # Validation subfolders
│
├── splits/                                        # Patient-partitioned zero-leakage splits
│   ├── nih/
│   │   ├── train.csv                              # 70% patient-clustered training split
│   │   ├── val.csv                                # 10% patient-clustered validation split
│   │   └── test.csv                               # 20% patient-clustered test split
│   └── chexpert/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
└── synthetic/                                     # Rapid CI dry-run fixtures
    ├── images/
    └── metadata.csv
```

---

## 3. Automated Ingestion via Kaggle CLI (Recommended)

The fastest, most automated way to acquire and set up the datasets is via the integrated onboarding script `scripts/setup_real_data.py`.

### Step 3.1: Configure Kaggle API Token
1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. Navigate to your profile: **Account &rarr; API &rarr; Create New Token**.
3. A file named `kaggle.json` will be downloaded to your machine.
4. Place `kaggle.json` in your user directory:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux / macOS**: `~/.kaggle/kaggle.json` (ensure `chmod 600 ~/.kaggle/kaggle.json`)

### Step 3.2: Execute Automated Download & Restructuring
Run the setup CLI:
```bash
# 1. Download and extract both datasets automatically
py -3.12 scripts/setup_real_data.py --download_kaggle

# 2. Automatically generate zero-leakage patient-partitioned splits
py -3.12 scripts/setup_real_data.py --prepare_splits

# 3. Verify dataset installation status
py -3.12 scripts/setup_real_data.py --verify_only
```

---

## 4. Manual Download Options

If you prefer downloading files manually through your web browser or an institutional transfer protocol:

### Option A: NIH ChestX-ray14
- **Official Host**: National Institutes of Health Clinical Center Box repository.
- **URL**: `https://nihcc.app.box.com/v/ChestXray-NIHCC`
- **Downloadables**:
  - `images_001.zip` through `images_012.zip` (12 batch archives).
  - `Data_Entry_2017_v2020.csv`
  - `train_val_list.txt` & `test_list.txt`
  - `BBox_List_2017.csv`
- Extract all 12 archives into a single flat directory: `data/nih_cxr/images/`.

### Option B: Stanford CheXpert
- **Official Host**: Stanford Machine Learning Group / AIMI Center.
- **URL**: `https://stanfordmlgroup.github.io/competitions/chexpert/`
- **Recommended Variant**: `CheXpert-v1.0-small.zip` (~11 GB downsampled resolution).
- Extract into: `data/chexpert/CheXpert-v1.0-small/`.

Once downloaded, run the extraction and split pipeline:
```bash
py -3.12 scripts/setup_real_data.py --archives_dir "C:\Downloads\CXR_Archives" --extract_archives --prepare_splits
```

---

## 5. Execution Modes: Full Scale vs. Curated Subsample

PneumoShift supports two primary operational scales for real-data experimentation:

### Mode 1: Full-Cohort Scale (`--full`)
- Evaluates on the complete cohorts ($112,120$ NIH images + $224,316$ CheXpert images).
- Recommended for high-performance multi-GPU workstation or cloud cluster environments (e.g., NVIDIA A100, RTX 4090).
- Standard training time: ~3.5 hours per model epoch on a single high-end GPU.

### Mode 2: Curated Stratified Subsample (`--subsample`)
- Generates a patient-stratified subset of **10,000 images per cohort** preserving disease prevalence and zero patient overlap.
- Designed for rapid single-GPU or local laptop experimentation.
- Standard training time: ~15 minutes per epoch on single consumer GPUs.
```bash
py -3.12 scripts/setup_real_data.py --subsample --subsample_size 10000 --prepare_splits
```

---

## 6. Zero-Leakage Split Verification & Data Audit

After extracting the datasets, always execute the automated patient leakage verification audit to guarantee scientific integrity:
```bash
# Verify splits and run clinical distribution audit
py -3.12 scripts/prepare_splits.py --dataset all --audit
```

The script performs a mathematical intersection check:
$$\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{val}} = \emptyset, \quad \text{Patients}_{\text{train}} \cap \text{Patients}_{\text{test}} = \emptyset$$
Any patient overlap triggers a hard `AssertionError` preventing model training.

---

## 7. Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: "Filenames exceed Windows 260-character path limit"
- **Cause**: Windows historically restricts absolute paths to 260 characters (`MAX_PATH`).
- **Fix**: Enable Long Paths in Windows:
  1. Open PowerShell as Administrator.
  2. Run: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`
  3. Alternatively, keep your project directory close to the drive root (e.g., `C:\PneumoShift`).

### Q2: "Kaggle 401 Unauthorized Error"
- **Cause**: Missing or expired `kaggle.json`.
- **Fix**: Regenerate an API token on `kaggle.com` and ensure it is saved at `~/.kaggle/kaggle.json` with user read permissions.

### Q3: "Can I use external pre-trained checkpoints?"
- **Answer**: Yes. In `scripts/train.py` and `scripts/reproduce_all.py`, pass `--checkpoint <path_to_weights.pt>` to bypass training from scratch and jump straight to cross-domain evaluation, calibration auditing, and explainability heatmaps.
