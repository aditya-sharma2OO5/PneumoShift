# Methodology and Experimental Protocol (docs/METHODS.md)

## 1. Overview
This document specifies the end-to-end experimental protocol for **PneumoShift**, investigating model robustness, calibration breakdown, threshold failure, and saliency degradation across chest X-ray distribution shifts.

---

## 2. Harmonized Shared Labels
Both NIH ChestX-ray14 and CheXpert datasets are mapped to 8 canonical labels:
1. `Atelectasis`
2. `Cardiomegaly`
3. `Consolidation`
4. `Edema`
5. `Pleural Effusion`
6. `Pneumonia` (Primary Headline Finding)
7. `Pneumothorax`
8. `No Finding`

---

## 3. Dataset Splitting Protocol
1. **Patient-level isolation**:
   - `patient_id` is parsed from file paths or metadata columns.
   - All images for a given patient are strictly assigned to exactly one partition (`train`, `val`, `test`, or `target_calibration`).
2. **NIH ChestX-ray14**:
   - Follows official `train_val_list.txt` and `test_list.txt`.
   - The official `train_val` cohort is split into 85% train and 15% validation at the patient level.
3. **Stanford CheXpert**:
   - Since CheXpert's official benchmark test set is held private by Stanford AIMI, `train.csv` is split into an 80/10/10 patient-level partition (`train`, `val`, `test`).
   - The official 200-study radiologist-annotated `valid.csv` is preserved as an external sanity check with reported confidence intervals.
4. **Target Calibration Holdout**:
   - A disjoint 10% slice of the target test set is carved out prior to final evaluation, designated as `target_cal`.

---

## 4. Evaluation Metrics & Statistical Testing
- **Discrimination**:
  - AUROC (Area under the Receiver Operating Characteristic curve)
  - AUPRC (Area under the Precision-Recall curve)
  - Sensitivity at fixed 90% specificity
  - Specificity at fixed 90% sensitivity
- **Calibration**:
  - Expected Calibration Error (ECE) with $M=15$ bins:
    $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
  - Maximum Calibration Error (MCE)
  - Brier score: $\frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$
  - Negative Log-Likelihood (NLL)
- **Statistical Significance**:
  - Non-parametric patient-clustered bootstrap with $B=1000$ iterations for all 95% confidence intervals.
  - DeLong test and paired bootstrap for comparing AUROC between in-domain and cross-domain models.
  - Benjamini-Hochberg FDR control $(\alpha = 0.05)$ across multiple comparison hypotheses.

---

## 5. Explainability Analysis Protocol
- **Grad-CAM Calculation**:
  Compute gradient of class score $y^c$ with respect to activation map $A^k$:
  $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
- **Faithfulness**:
  Progressively mask out top salient pixels in steps of 5% up to 50% and record pneumonia score degradation.
- **Sanity Checks**:
  Verify complete randomization of convolutional weights destroys attribution localization.
- **Plausibility & Shortcuts**:
  Measure attribution mass within lung segmentation masks vs exterior image margins/hardware markers.

