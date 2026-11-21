# Kaggle Inference Guide: Real Checkpoints on CheXpert Frontal Cohort

**Run ID**: `oldckpt_real_v1`  
**Notebook**: [`notebooks/kaggle_infer_old_checkpoints.ipynb`](file:///C:/Users/adity/Github%20Repo/PneumoShift/notebooks/kaggle_infer_old_checkpoints.ipynb)  
**Warning**: *Do not launch the full execution run until the user confirms.*

---

## 1. Overview
This protocol establishes real cross-domain empirical predictions by running the 6 verified historical NIH model checkpoints directly on real CheXpert images (`willarevalo/chexpert-v10-small`).

### Checkpoints to Evaluate
| Key | Architecture | Seed | Best Stored Epoch | SHA-256 | NIH Val AUROC |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `densenet121_42` | DenseNet-121 | 42 | 5 | `6e77c6a6cc3e10cb5150924dd81ff74c5eb44359466ae2d3f56519c875d23a59` | 0.747424 |
| `densenet121_1337` | DenseNet-121 | 1337 | 1 | `c3f6afccac87254c980a88ea72c71fb13631364483401cbed5bc8b2052b946a0` | 0.747471 |
| `densenet121_2024` | DenseNet-121 | 2024 | 1 | `494dd05fd54cd0effdc3393534d99d6a821b2ff93f63bf0c4841493eb0c851c3` | 0.748308 |
| `efficientnet_b0_42` | EfficientNet-B0 | 42 | 5 | `dc2b37375b6909b814830dd57eb384765f5c18ed8fdc34ced349fa48872458bc` | 0.756032 |
| `efficientnet_b0_1337` | EfficientNet-B0 | 1337 | 1 | `7772e387d3d2b4000825da5e70f858c09ce81862340d1f6199b0d0aad7f958b9` | 0.762341 |
| `efficientnet_b0_2024` | EfficientNet-B0 | 2024 | 1 | `8d8b6f80892e9854001ac24bfadea81f25703c5d617ea83c24eab9935788815c` | 0.758065 |

---

## 2. Step 1: Upload Checkpoints to Kaggle as a Private Dataset

### Method A: Kaggle Web UI (Recommended)
1. In your local terminal, generate or verify `checkpoint_manifest.json`:
   ```powershell
   python scripts/create_checkpoint_manifest.py
   ```
2. Create a staging folder and stage the 6 checkpoints alongside the manifest:
   ```powershell
   mkdir -p staging_checkpoints
   cp results/runs/nih_densenet121/checkpoints/seed_42/checkpoint_best.pt staging_checkpoints/checkpoint_best_densenet121_42.pt
   cp results/runs/nih_densenet121/checkpoints/seed_1337/checkpoint_best.pt staging_checkpoints/checkpoint_best_densenet121_1337.pt
   cp results/runs/nih_densenet121/checkpoints/seed_2024/checkpoint_best.pt staging_checkpoints/checkpoint_best_densenet121_2024.pt
   cp results/runs/nih_efficientnet_b0/checkpoints/seed_42/checkpoint_best.pt staging_checkpoints/checkpoint_best_efficientnet_b0_42.pt
   cp results/runs/nih_efficientnet_b0/checkpoints/seed_1337/checkpoint_best.pt staging_checkpoints/checkpoint_best_efficientnet_b0_1337.pt
   cp results/runs/nih_efficientnet_b0/checkpoints/seed_2024/checkpoint_best.pt staging_checkpoints/checkpoint_best_efficientnet_b0_2024.pt
   cp checkpoint_manifest.json staging_checkpoints/
   ```
3. Navigate to [Kaggle Datasets](https://www.kaggle.com/datasets) -> Click **New Dataset**.
4. Set Title: `pneumoshift-checkpoints` (Slug: `pneumoshift-checkpoints`).
5. Drag and drop all files from `staging_checkpoints/`.
6. Select **Private** -> Click **Create**.

### Method B: Kaggle CLI
```bash
kaggle datasets init -p staging_checkpoints/
# In dataset-metadata.json, set "title": "pneumoshift-checkpoints", "id": "<username>/pneumoshift-checkpoints"
kaggle datasets create -p staging_checkpoints/ -u
```

---

## 3. Step 2: Configure Kaggle Notebook
1. Go to [Kaggle Code](https://www.kaggle.com/code) -> Click **New Notebook**.
2. **File** -> **Import Notebook** -> Upload [`notebooks/kaggle_infer_old_checkpoints.ipynb`](file:///C:/Users/adity/Github%20Repo/PneumoShift/notebooks/kaggle_infer_old_checkpoints.ipynb).
3. In the right-hand **Notebook settings** panel:
   - **Accelerator**: GPU P100 (or GPU T4 x2)
   - **Internet**: On (for initial package downloads if needed)
4. Under **Input Datasets** (Add Input):
   - Add your private dataset: `pneumoshift-checkpoints`
   - Add CheXpert: `willarevalo/chexpert-v10-small`
   - Add NIH: `data-cxr` (or official NIH ChestX-ray14)

---

## 4. Step 3: Execution & Sanity Verification

### Dry Run (Smoke Mode)
- Ensure line in Cell 2: `SMOKE_TEST = True`.
- Click **Run All**.
- Verifies:
  1. All 6 checkpoint SHA-256 hashes match bit-for-bit.
  2. NIH Sanity Gate executes and validates AUROC agreement to 4 decimals.
  3. CheXpert cohort formulation measures label distribution and frontal slice.
  4. 1 checkpoint runs on 256 images, saves stamped `.npz`, and creates `chexpert_predictions_oldckpt_real_v1.zip` in `< 3 minutes`.

### Full Production Run
- Once confirmed, set: `SMOKE_TEST = False`.
- Click **Run All** (estimated runtime: ~25-35 minutes for all 6 models on ~191k CheXpert frontal images with GPU P100).
- Download `chexpert_predictions_oldckpt_real_v1.zip` from `/kaggle/working/outputs/oldckpt_real_v1/`.

---

## 5. Step 4: Local Ingestion & Pipeline Execution
Once downloaded, extract files into `results/predictions/`:
```powershell
Expand-Archive -Path chexpert_predictions_oldckpt_real_v1.zip -DestinationPath results/predictions/ -Force
```
Run provenance test and master pipeline:
```powershell
python -m pytest tests/test_provenance_stamping.py -v
python scripts/run_task_b_pipeline.py --run_id oldckpt_real_v1 --output_dir results/tables/oldckpt_real_v1
```
