# Data Acquisition and Directory Setup Guide (docs/DATA_GUIDE.md)

This guide documents how to acquire, configure, and verify the datasets used in **PneumoShift**:
1. **NIH ChestX-ray14**
2. **Stanford CheXpert**
3. **Synthetic Smoke Test Data** (generated locally for dry runs and continuous integration)

---

## 1. Expected Directory Layout

Set up a root data folder (e.g. `data/` or an external drive) adhering to the structure below:

```text
data/
├── nih_cxr/
│   ├── images/
│   │   ├── 00000001_000.png
│   │   ├── 00000001_001.png
│   │   └── ...
│   ├── Data_Entry_2017_v2020.csv
│   ├── train_val_list.txt
│   ├── test_list.txt
│   └── BBox_List_2017.csv
│
├── chexpert/
│   ├── CheXpert-v1.0-small/  (or full resolution CheXpert-v1.0/)
│   │   ├── train.csv
│   │   ├── valid.csv
│   │   ├── train/
│   │   │   └── patient00001/
│   │   │       └── study1/
│   │   │           └── view1_frontal.jpg
│   │   └── valid/
│   │       └── patient64541/
│   │           └── study1/
│   │               └── view1_frontal.jpg
│
└── synthetic/
    ├── images/
    │   ├── synth_0001.png
    │   └── ...
    └── metadata.csv
```

---

## 2. Dataset Acquisition Details

### NIH ChestX-ray14
- **Source**: NIH Clinical Center / Box repository.
- **Access**: Publicly available without credentialed registration.
- **Files required**:
  - `images/`: The 12 image batch archives extracted into a single folder.
  - `Data_Entry_2017_v2020.csv`: Image index, finding labels, patient ID, age, gender, view position.
  - `train_val_list.txt` & `test_list.txt`: Official NIH partition lists.
  - `BBox_List_2017.csv`: Ground-truth bounding boxes for disease localization evaluation.

### Stanford CheXpert
- **Source**: Stanford AIMI (Artificial Intelligence in Medicine and Imaging).
- **Access**: Registration required on the Stanford ML Group / AIMI portal.
- **License**: Strictly respect Stanford's data use agreement. Do not attempt automated scraping or redistribution.
- **Files required**:
  - `train.csv` and `valid.csv`.
  - Image directory tree (`train/`, `valid/`).

---

## 3. Synthetic Smoke Test Data

For testing pipelines without downloading gigabytes of medical imagery:
```bash
python scripts/prepare_splits.py --smoke_test --output_dir data/synthetic
python scripts/train.py --config configs/smoke_test.yaml
```
This generates tiny $64 \times 64$ or $224 \times 224$ synthetic chest X-ray radiographs with mock patient IDs and label vectors to verify data loaders, training loops, checkpointing, and evaluation metrics end-to-end.

