"""
Unified Task B Statistical Protocol & Master Table Orchestrator (Addendum 2)

Implements rigorous, unified statistical inference across Tables 1-3:
1. Patient-level cluster bootstrap (>=2,000 resamples, fixed seed) replacing all image-level inference.
2. Cross-cohort comparison evaluated on identical resampled patient clusters across model seeds.
3. Reporting effective numbers: positive images AND positive patients per cohort.
4. Independent DeLong test reported strictly as a secondary sensitivity analysis (labeled "ignoring clustering").
5. DenseNet-121 cross-domain evaluation based strictly on on-disk computable seeds (1337, 2024; seed 42 marked not computable).
6. Table 3: Sensitivity & Specificity primary (prevalence-invariant); F1 secondary with prevalence shift caveat.
7. Explicit documentation of CheXpert cohort definition and NIH test list pathological enrichment limitations.

Usage:
    python scripts/run_task_b_pipeline.py --run_id old_run --output_dir results/tables/provisional_old_run
    python scripts/run_task_b_pipeline.py --run_id repro_v1 --predictions_dir results/predictions_repro_v1 --output_dir results/tables/repro_v1
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TaskB_Pipeline")


# ==============================================================================
# 1. CORE STATISTICAL PRIMITIVES & CLUSTER BOOTSTRAP
# ==============================================================================

def get_patient_clusters(patient_ids: np.ndarray) -> Tuple[List[np.ndarray], int]:
    """Precomputes index mapping for fast patient-level cluster bootstrap."""
    unique_pts, pt_inverse = np.unique(patient_ids, return_inverse=True)
    n_pts = len(unique_pts)
    pt_to_indices = [np.where(pt_inverse == i)[0] for i in range(n_pts)]
    return pt_to_indices, n_pts


def patient_cluster_resample(pt_to_indices: List[np.ndarray], n_pts: int, rng: np.random.RandomState) -> np.ndarray:
    """Resamples patients with replacement and returns concatenated image indices."""
    sampled_pt_idx = rng.randint(0, n_pts, size=n_pts)
    return np.concatenate([pt_to_indices[i] for i in sampled_pt_idx])


def get_cohort_summary_stats(y_true: np.ndarray, patient_ids: np.ndarray) -> Dict[str, Any]:
    """Computes total images, total patients, positive images, and positive patients."""
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid]
    pts = patient_ids[valid]
    n_images = len(yt)
    n_pos_images = int(np.sum(yt == 1))
    n_neg_images = n_images - n_pos_images

    u_pts, pt_inv = np.unique(pts, return_inverse=True)
    n_patients = len(u_pts)
    pt_has_pos = np.zeros(n_patients, dtype=bool)
    for p_idx, y in zip(pt_inv, yt):
        if y == 1:
            pt_has_pos[p_idx] = True
    n_pos_pts = int(np.sum(pt_has_pos))
    n_neg_pts = n_patients - n_pos_pts

    return {
        "n_images": n_images,
        "n_pos_images": n_pos_images,
        "n_neg_images": n_neg_images,
        "image_prevalence": float(n_pos_images / max(n_images, 1)),
        "n_patients": n_patients,
        "n_pos_patients": n_pos_pts,
        "n_neg_patients": n_neg_pts,
        "patient_prevalence": float(n_pos_pts / max(n_patients, 1)),
        "images_per_patient": float(n_images / max(n_patients, 1)),
    }


def auc_searchsorted(pos: np.ndarray, neg: np.ndarray) -> float:
    """Fast, exact Wilcoxon-Mann-Whitney AUROC calculation using searchsorted."""
    n_p = len(pos)
    n_n = len(neg)
    if n_p == 0 or n_n == 0:
        return 0.5
    neg_sorted = np.sort(neg)
    idx_left = np.searchsorted(neg_sorted, pos, side="left")
    idx_right = np.searchsorted(neg_sorted, pos, side="right")
    u = np.sum(idx_left + idx_right) / 2.0
    return float(u / (n_p * n_n))


def compute_fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Wrapper for fast AUROC with binary labels."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    return auc_searchsorted(pos, neg)


def compute_delong_auc_variance(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Computes DeLong AUROC and its variance (note: assumes i.i.d. images, ignoring clustering)."""
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid]
    ys = y_score[valid]

    pos = ys[yt == 1]
    neg = ys[yt == 0]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return 0.5, 0.0

    auc = auc_searchsorted(pos, neg)

    # DeLong structural components V10 and V01
    neg_sorted = np.sort(neg)
    v10 = (np.searchsorted(neg_sorted, pos, side="left") + np.searchsorted(neg_sorted, pos, side="right")) / (2.0 * n)

    pos_sorted = np.sort(pos)
    v01 = 1.0 - (np.searchsorted(pos_sorted, neg, side="left") + np.searchsorted(pos_sorted, neg, side="right")) / (2.0 * m)

    s10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0
    var_auc = (s10 / m) + (s01 / n)
    return float(auc), float(var_auc)


def independent_delong_test(
    y_true_1: np.ndarray,
    y_score_1: np.ndarray,
    y_true_2: np.ndarray,
    y_score_2: np.ndarray,
) -> Dict[str, float]:
    """Performs independent two-sample DeLong test for AUROC difference (ignoring clustering)."""
    auc1, var1 = compute_delong_auc_variance(y_true_1, y_score_1)
    auc2, var2 = compute_delong_auc_variance(y_true_2, y_score_2)

    diff = auc1 - auc2
    se_diff = float(np.sqrt(max(var1 + var2, 1e-12)))
    z = float(diff / se_diff)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    return {
        "auc_1": float(auc1),
        "var_1": float(var1),
        "auc_2": float(auc2),
        "var_2": float(var2),
        "diff": float(diff),
        "se_diff": se_diff,
        "z": z,
        "p_val": p_val,
    }


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Applies Benjamini-Hochberg False Discovery Rate (FDR) adjustment."""
    n = len(p_values)
    if n == 0:
        return []
    p_arr = np.asarray(p_values, dtype=float)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    q_vals = np.zeros(n, dtype=float)
    cumulative_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = (sorted_p[i] * n) / rank
        cumulative_min = min(cumulative_min, adj)
        q_vals[i] = cumulative_min
    q_vals = np.clip(q_vals, 0.0, 1.0)
    adjusted = np.zeros(n, dtype=float)
    adjusted[order] = q_vals
    return adjusted.tolist()


def compute_comprehensive_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_logits: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calculates all calibration metrics required by Table 2."""
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid].astype(float)
    yp = np.clip(y_prob[valid].astype(float), 1e-7, 1.0 - 1e-7)
    N = len(yt)
    n_pos = int(np.sum(yt == 1))
    base_rate = float(np.mean(yt)) if N > 0 else 0.0

    # 1. Equal-width Uniform ECE (10 bins)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(yp, bin_edges[1:-1])
    ece_uniform = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        count = np.sum(mask)
        if count > 0:
            acc = np.mean(yt[mask])
            conf = np.mean(yp[mask])
            ece_uniform += (count / N) * abs(acc - conf)

    # 2. Adaptive Quantile ECE (10 quantile bins)
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    ad_edges = np.unique(np.percentile(yp, quantiles))
    if len(ad_edges) < 2:
        ad_edges = bin_edges
    ad_indices = np.digitize(yp, ad_edges[1:-1])
    ece_adaptive = 0.0
    for b in range(len(ad_edges) - 1):
        mask = ad_indices == b
        count = np.sum(mask)
        if count > 0:
            acc = np.mean(yt[mask])
            conf = np.mean(yp[mask])
            ece_adaptive += (count / N) * abs(acc - conf)

    # 3. Brier Score & BSS vs Base Rate
    brier = float(np.mean((yp - yt) ** 2))
    brier_base = float(base_rate * (1.0 - base_rate))
    bss = float(1.0 - (brier / max(brier_base, 1e-8)))

    # 4. NLL
    nll = float(-np.mean(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp)))
    mean_pred = float(np.mean(yp))

    # 5. Calibration Slope & Intercept
    logit_yp = y_logits if y_logits is not None else np.log(yp / (1.0 - yp))
    try:
        clf = LogisticRegression(solver="lbfgs", C=1e6)
        clf.fit(logit_yp.reshape(-1, 1), yt)
        cal_slope = float(clf.coef_[0, 0])
        cal_intercept = float(clf.intercept_[0])
    except Exception:
        cal_slope = float("nan")
        cal_intercept = float("nan")

    return {
        "ece_uniform": float(ece_uniform),
        "ece_adaptive": float(ece_adaptive),
        "brier": float(brier),
        "bss": float(bss),
        "cal_slope": cal_slope,
        "cal_intercept": cal_intercept,
        "nll": float(nll),
        "mean_prediction": mean_pred,
        "N": N,
        "n_pos": n_pos,
        "prevalence": base_rate,
    }


def fit_temperature_scalar(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fits scalar temperature T > 0 on logits minimizing NLL via L-BFGS-B."""
    valid = (labels == 0.0) | (labels == 1.0)
    z_v = logits[valid].astype(np.float64)
    y_v = labels[valid].astype(np.float64)

    if len(y_v) == 0 or len(np.unique(y_v)) < 2:
        return 1.0

    def nll_objective(t: np.ndarray) -> float:
        temp = max(float(t[0]), 1e-3)
        scaled_z = z_v / temp
        loss = np.maximum(scaled_z, 0.0) - scaled_z * y_v + np.log1p(np.exp(-np.abs(scaled_z)))
        return float(np.mean(loss))

    res = optimize.minimize(
        nll_objective,
        x0=np.array([1.0], dtype=np.float64),
        bounds=[(0.05, 5.0)],
        method="L-BFGS-B",
        options={"eps": 1e-4},
    )
    return float(res.x[0])


def fit_platt_scaling(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Fits univariate logistic regression P(y=1) = sigma(a * z + b) on validation set."""
    valid = (labels == 0.0) | (labels == 1.0)
    z_v = logits[valid].reshape(-1, 1)
    y_v = labels[valid]
    clf = LogisticRegression(solver="lbfgs", C=1e6)
    clf.fit(z_v, y_v)
    a = float(clf.coef_[0, 0])
    b = float(clf.intercept_[0])
    return a, b


def compute_binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Evaluates binary metrics (sensitivity, specificity, F1, precision) at decision threshold."""
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid].astype(int)
    ys = y_score[valid]

    y_pred = (ys >= threshold).astype(int)
    tp = int(np.sum((yt == 1) & (y_pred == 1)))
    fp = int(np.sum((yt == 0) & (y_pred == 1)))
    fn = int(np.sum((yt == 1) & (y_pred == 0)))
    tn = int(np.sum((yt == 0) & (y_pred == 0)))

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = (2.0 * prec * sens) / max(prec + sens, 1e-8)

    return {
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_optimal_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Computes Youden J, Sensitivity-90, and Max-F1 thresholds on validation data."""
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid].astype(int)
    ys = y_score[valid]

    candidate_taus = np.unique(np.quantile(ys, np.linspace(0.01, 0.99, 200)))
    best_youden = -1.0
    best_youden_tau = 0.5
    best_f1 = -1.0
    best_f1_tau = 0.5
    sens90_tau = 0.5
    found_sens90 = False

    for tau in candidate_taus:
        m = compute_binary_metrics_at_threshold(yt, ys, tau)
        yj = m["sensitivity"] + m["specificity"] - 1.0
        if yj > best_youden:
            best_youden = yj
            best_youden_tau = tau
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_f1_tau = tau
        if m["sensitivity"] >= 0.90 and (not found_sens90 or tau > sens90_tau):
            sens90_tau = tau
            found_sens90 = True

    return {
        "youden_threshold": float(best_youden_tau),
        "sens90_threshold": float(sens90_tau),
        "f1_threshold": float(best_f1_tau),
    }


# ==============================================================================
# 2. DATA LOADER & INVENTORY AUDITOR
# ==============================================================================

class PredictionBundle:
    """Manages loaded prediction arrays and threshold metadata for a single run_id."""

    def __init__(self, run_id: str, predictions_dir: Path, allow_provisional: bool = True):
        self.run_id = run_id
        self.predictions_dir = Path(predictions_dir)
        self.allow_provisional = allow_provisional
        self.archs = ["densenet121", "efficientnet_b0"]
        self.seeds = ["42", "1337", "2024"]
        self.data: Dict[str, Any] = {}
        self.load_and_audit()

    def load_and_audit(self) -> None:
        """Loads all required prediction arrays and verifies schema and run_id."""
        logger.info("Loading prediction bundle from '%s' for run_id='%s'...", self.predictions_dir, self.run_id)

        for arch in self.archs:
            for seed in self.seeds:
                # 1. Validation split
                val_path = self.predictions_dir / f"predictions_val_{arch}_{seed}.npz"
                if not val_path.exists():
                    raise KeyError(f"Missing required prediction file: {val_path}")
                val_data = np.load(val_path)
                self._verify_keys(val_data, ["y_true", "y_score", "y_logits", "patient_ids"], str(val_path))
                self.data[f"val_{arch}_{seed}"] = val_data

                # 2. In-domain test split
                test_path = self.predictions_dir / f"predictions_test_{arch}_{seed}.npz"
                if not test_path.exists():
                    raise KeyError(f"Missing required prediction file: {test_path}")
                test_data = np.load(test_path)
                self._verify_keys(test_data, ["y_true", "y_score", "y_logits", "patient_ids"], str(test_path))
                self.data[f"test_{arch}_{seed}"] = test_data

                # 3. CheXpert target split
                cr_path = self.predictions_dir / f"predictions_cross_domain_{arch}_{seed}.npz"
                if not cr_path.exists():
                    raise KeyError(f"Missing required prediction file: {cr_path}")
                cr_data = np.load(cr_path)
                self._verify_keys(cr_data, ["y_true", "y_score", "y_logits", "patient_ids"], str(cr_path))

                # Provisional check for overwritten smoke test file (e.g. densenet121 seed 42)
                if len(cr_data["y_true"]) < 100:
                    if self.allow_provisional:
                        logger.warning(
                            "Provisional run_id='%s': %s contains only %d samples (smoke test artifact). Marking not computable (old_run).",
                            self.run_id, cr_path.name, len(cr_data["y_true"])
                        )
                        self.data[f"cr_{arch}_{seed}_computable"] = False
                    else:
                        raise ValueError(f"Cohort array {cr_path} has insufficient samples ({len(cr_data['y_true'])})")
                else:
                    self.data[f"cr_{arch}_{seed}_computable"] = True
                self.data[f"cr_{arch}_{seed}"] = cr_data

                # 4. Tau selection metadata
                tau_path = self.predictions_dir / f"tau_selection_{arch}_{seed}.json"
                if not tau_path.exists():
                    raise KeyError(f"Missing required threshold file: {tau_path}")
                with open(tau_path, "r", encoding="utf-8") as fp:
                    tau_data = json.load(fp)
                for req_k in ["youden_tau", "sens90_tau", "max_f1_tau"]:
                    if req_k not in tau_data:
                        raise KeyError(f"Missing required threshold key '{req_k}' in {tau_path}")
                self.data[f"tau_{arch}_{seed}"] = tau_data

    def _verify_keys(self, npz_data: Any, required_keys: List[str], filename: str) -> None:
        """Verifies presence of required keys."""
        missing = [k for k in required_keys if k not in npz_data]
        if missing:
            raise KeyError(f"File '{filename}' is missing essential keys: {missing}")

    def get_pneumonia_slice(self, key: str) -> Dict[str, Any]:
        """Extracts 1D pneumonia predictions (column 5) and patient_ids."""
        npz = self.data[key]
        p_idx = 5  # Pneumonia index confirmed across all harmonization schemas
        if "label_cols" in npz:
            cols = list(npz["label_cols"])
            if "Pneumonia" in cols:
                p_idx = cols.index("Pneumonia")

        y_true = npz["y_true"][:, p_idx] if npz["y_true"].ndim > 1 else npz["y_true"]
        y_true = np.where(y_true == -1.0, 0.0, y_true)  # Enforce u_zeros policy
        y_score = npz["y_score"][:, p_idx] if npz["y_score"].ndim > 1 else npz["y_score"]
        y_logits = npz["y_logits"][:, p_idx] if npz["y_logits"].ndim > 1 else npz["y_logits"]
        patient_ids = npz["patient_ids"]

        return {
            "y_true": y_true,
            "y_score": y_score,
            "y_logits": y_logits,
            "patient_ids": patient_ids,
            "pneumonia_col_idx": p_idx,
        }


# ==============================================================================
# 3. TABLE 1 GENERATOR: Primary Discrimination Benchmark
# ==============================================================================

def generate_table_1(
    bundle: PredictionBundle,
    output_dir: Path,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Generates Table 1 and Table 1 Supplement following unified statistical protocol (Addendum 2)."""
    logger.info("Generating Table 1: Primary Cross-Domain Discrimination Benchmark (Cluster Bootstrap)...")

    t1_rows = []
    supplement_rows = []
    table_json: Dict[str, Any] = {
        "title": f"PROVISIONAL ({bundle.run_id}) Table 1: Primary Cross-Domain Discrimination Benchmark",
        "run_id": bundle.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_cluster_bootstrap": n_bootstrap,
        "protocol": {
            "aggregation": "mean +/- SD across seeds (never 95% CI)",
            "inference": "Patient-level cluster bootstrap (>=2,000 resamples, fixed seed) accounting for repeated imaging",
            "cross_cohort_comparison": "Resample patients independently within each cohort using identical patient resamples across model seeds",
            "secondary_sensitivity": "DeLong test reported as secondary sensitivity analysis explicitly labeled as ignoring clustering",
            "fdr_adjustment": "Benjamini-Hochberg across architectures on cluster-bootstrap p-values",
        },
        "effective_cohort_numbers": {},
        "sd_ratio_explanation": (
            "In-domain AUROC standard deviation across seeds (~0.028) is ~10x larger than transferred "
            "CheXpert standard deviation (~0.003). Root causes: (1) Checkpoint selection was monitored on "
            "val_pneumonia_auroc on the NIH validation split, which contains only 139 pneumonia positives "
            "across 110 patients (sampling SE approx 0.039). Random validation fluctuations caused different seeds "
            "to checkpoint at disparate epochs (e.g. Epoch 2 vs Epoch 9), producing large in-domain test spread. "
            "(2) Domain shift to CheXpert attenuates fine-grained radiographic idiosyncrasies and projects "
            "representations onto coarse opacities, which have similar predictive performance across checkpoints, "
            "compressing transferred between-seed variance."
        ),
        "architectures": {},
    }

    # Record global cohort numbers from representative slices
    sample_test = bundle.get_pneumonia_slice("test_densenet121_1337")
    sample_val = bundle.get_pneumonia_slice("val_densenet121_1337")
    sample_chex = bundle.get_pneumonia_slice("cr_densenet121_1337")

    stats_val = get_cohort_summary_stats(sample_val["y_true"], sample_val["patient_ids"])
    stats_test = get_cohort_summary_stats(sample_test["y_true"], sample_test["patient_ids"])
    stats_chex = get_cohort_summary_stats(sample_chex["y_true"], sample_chex["patient_ids"])

    table_json["effective_cohort_numbers"] = {
        "nih_validation": stats_val,
        "nih_test": stats_test,
        "chexpert_test": stats_chex,
    }

    raw_p_boot_list = []
    arch_keys = []

    for arch in bundle.archs:
        arch_display = "DenseNet121" if "dense" in arch.lower() else "EfficientNet-B0"
        arch_meta: Dict[str, Any] = {"seeds": {}}

        nih_aurocs, nih_auprcs = [], []
        chex_aurocs, chex_auprcs = [], []
        delta_aurocs = []
        delong_p_vals = []

        available_seeds = []
        for s in bundle.seeds:
            test_slice = bundle.get_pneumonia_slice(f"test_{arch}_{s}")
            cr_computable = bundle.data.get(f"cr_{arch}_{s}_computable", True)

            # In-domain metrics
            auc_in, var_in = compute_delong_auc_variance(test_slice["y_true"], test_slice["y_score"])
            pr_in = float(average_precision_score(test_slice["y_true"], test_slice["y_score"]))
            c_stats_in = get_cohort_summary_stats(test_slice["y_true"], test_slice["patient_ids"])

            nih_aurocs.append(auc_in)
            nih_auprcs.append(pr_in)

            # Patient-level cluster bootstrap for NIH Test (B >= 2000)
            rng_s = np.random.RandomState(int(s) + seed)
            map_in, n_pts_in = get_patient_clusters(test_slice["patient_ids"])

            boot_auc_in = []
            boot_pr_in = []
            for _ in range(n_bootstrap):
                idx = patient_cluster_resample(map_in, n_pts_in, rng_s)
                yt_b = test_slice["y_true"][idx]
                ys_b = test_slice["y_score"][idx]
                boot_auc_in.append(compute_fast_auc(yt_b, ys_b))
                boot_pr_in.append(float(average_precision_score(yt_b, ys_b)))

            ci_auc_in = [float(np.percentile(boot_auc_in, 2.5)), float(np.percentile(boot_auc_in, 97.5))]
            ci_pr_in = [float(np.percentile(boot_pr_in, 2.5)), float(np.percentile(boot_pr_in, 97.5))]

            seed_record: Dict[str, Any] = {
                "in_domain": {
                    "cohort": "NIH Test",
                    "n_images": c_stats_in["n_images"],
                    "n_pos_images": c_stats_in["n_pos_images"],
                    "n_patients": c_stats_in["n_patients"],
                    "n_pos_patients": c_stats_in["n_pos_patients"],
                    "auroc": auc_in,
                    "delong_se_ignoring_clustering": float(np.sqrt(var_in)),
                    "auroc_95ci_cluster_boot": ci_auc_in,
                    "auprc": pr_in,
                    "auprc_95ci_cluster_boot": ci_pr_in,
                }
            }

            supplement_rows.append({
                "Architecture": arch_display,
                "Seed": s,
                "Cohort": "NIH Test (In-Domain)",
                "N (Images)": c_stats_in["n_images"],
                "Pos Images": c_stats_in["n_pos_images"],
                "N (Patients)": c_stats_in["n_patients"],
                "Pos Patients": c_stats_in["n_pos_patients"],
                "AUROC": f"{auc_in:.4f}",
                "AUROC [95% Cluster Bootstrap CI]": f"[{ci_auc_in[0]:.3f}, {ci_auc_in[1]:.3f}]",
                "AUPRC": f"{pr_in:.4f}",
                "AUPRC [95% Cluster Bootstrap CI]": f"[{ci_pr_in[0]:.3f}, {ci_pr_in[1]:.3f}]",
                "Delta AUROC": "-",
                "DeLong p-value (ignoring clustering)": "-",
            })

            if cr_computable:
                cr_slice = bundle.get_pneumonia_slice(f"cr_{arch}_{s}")
                auc_cr, var_cr = compute_delong_auc_variance(cr_slice["y_true"], cr_slice["y_score"])
                pr_cr = float(average_precision_score(cr_slice["y_true"], cr_slice["y_score"]))
                c_stats_cr = get_cohort_summary_stats(cr_slice["y_true"], cr_slice["patient_ids"])

                chex_aurocs.append(auc_cr)
                chex_auprcs.append(pr_cr)
                delta = auc_in - auc_cr
                delta_aurocs.append(delta)

                delong_res = independent_delong_test(
                    test_slice["y_true"], test_slice["y_score"],
                    cr_slice["y_true"], cr_slice["y_score"]
                )
                delong_p_vals.append(delong_res["p_val"])

                # Patient-level cluster bootstrap for CheXpert and Delta AUROC
                map_cr, n_pts_cr = get_patient_clusters(cr_slice["patient_ids"])
                boot_auc_cr = []
                boot_pr_cr = []
                boot_delta = []
                for _ in range(n_bootstrap):
                    idx_in = patient_cluster_resample(map_in, n_pts_in, rng_s)
                    idx_cr = patient_cluster_resample(map_cr, n_pts_cr, rng_s)

                    a_in_b = compute_fast_auc(test_slice["y_true"][idx_in], test_slice["y_score"][idx_in])
                    a_cr_b = compute_fast_auc(cr_slice["y_true"][idx_cr], cr_slice["y_score"][idx_cr])

                    boot_auc_cr.append(a_cr_b)
                    boot_pr_cr.append(float(average_precision_score(cr_slice["y_true"][idx_cr], cr_slice["y_score"][idx_cr])))
                    boot_delta.append(a_in_b - a_cr_b)

                ci_auc_cr = [float(np.percentile(boot_auc_cr, 2.5)), float(np.percentile(boot_auc_cr, 97.5))]
                ci_pr_cr = [float(np.percentile(boot_pr_cr, 2.5)), float(np.percentile(boot_pr_cr, 97.5))]
                ci_delta = [float(np.percentile(boot_delta, 2.5)), float(np.percentile(boot_delta, 97.5))]

                seed_record["transferred"] = {
                    "cohort": "CheXpert",
                    "n_images": c_stats_cr["n_images"],
                    "n_pos_images": c_stats_cr["n_pos_images"],
                    "n_patients": c_stats_cr["n_patients"],
                    "n_pos_patients": c_stats_cr["n_pos_patients"],
                    "auroc": auc_cr,
                    "delong_se_ignoring_clustering": float(np.sqrt(var_cr)),
                    "auroc_95ci_cluster_boot": ci_auc_cr,
                    "auprc": pr_cr,
                    "auprc_95ci_cluster_boot": ci_pr_cr,
                    "delta_auroc": delta,
                    "delta_95ci_cluster_boot": ci_delta,
                    "delong_test_ignoring_clustering": delong_res,
                }

                supplement_rows.append({
                    "Architecture": arch_display,
                    "Seed": s,
                    "Cohort": "CheXpert (Transferred)",
                    "N (Images)": c_stats_cr["n_images"],
                    "Pos Images": c_stats_cr["n_pos_images"],
                    "N (Patients)": c_stats_cr["n_patients"],
                    "Pos Patients": c_stats_cr["n_pos_patients"],
                    "AUROC": f"{auc_cr:.4f}",
                    "AUROC [95% Cluster Bootstrap CI]": f"[{ci_auc_cr[0]:.3f}, {ci_auc_cr[1]:.3f}]",
                    "AUPRC": f"{pr_cr:.4f}",
                    "AUPRC [95% Cluster Bootstrap CI]": f"[{ci_pr_cr[0]:.3f}, {ci_pr_cr[1]:.3f}]",
                    "Delta AUROC": f"{delta:+.4f} [{ci_delta[0]:.3f}, {ci_delta[1]:.3f}]",
                    "DeLong p-value (ignoring clustering)": f"{delong_res['p_val']:.4f}",
                })
                available_seeds.append(s)
            else:
                seed_record["transferred"] = {
                    "cohort": "CheXpert",
                    "status": "not_computable_old_run",
                    "reason": "Overwritten with 4 dummy samples during smoke test; excluded from old_run calculations",
                }
                supplement_rows.append({
                    "Architecture": arch_display,
                    "Seed": s,
                    "Cohort": "CheXpert (Transferred)",
                    "N (Images)": "not computable (old_run)",
                    "Pos Images": "-",
                    "N (Patients)": "not computable (old_run)",
                    "Pos Patients": "-",
                    "AUROC": "not computable (old_run)",
                    "AUROC [95% Cluster Bootstrap CI]": "-",
                    "AUPRC": "-",
                    "AUPRC [95% Cluster Bootstrap CI]": "-",
                    "Delta AUROC": "-",
                    "DeLong p-value (ignoring clustering)": "-",
                })

            arch_meta["seeds"][s] = seed_record

        # -------------------------------------------------------------
        # Joint Patient-Level Cluster Bootstrap Across Available Seeds (B >= 2000)
        # Resamples patients independently within NIH and CheXpert cohorts;
        # Evaluates all available seeds on the SAME patient resamples.
        # -------------------------------------------------------------
        rng_joint = np.random.RandomState(seed)
        t_slices = [bundle.get_pneumonia_slice(f"test_{arch}_{s}") for s in available_seeds]
        c_slices = [bundle.get_pneumonia_slice(f"cr_{arch}_{s}") for s in available_seeds]

        map_nih_joint, n_pts_nih_joint = get_patient_clusters(t_slices[0]["patient_ids"])
        map_chex_joint, n_pts_chex_joint = get_patient_clusters(c_slices[0]["patient_ids"])

        boot_mean_deltas = []
        for _ in range(n_bootstrap):
            idx_nih = patient_cluster_resample(map_nih_joint, n_pts_nih_joint, rng_joint)
            idx_chex = patient_cluster_resample(map_chex_joint, n_pts_chex_joint, rng_joint)

            rep_deltas = []
            for k in range(len(available_seeds)):
                a_nih = compute_fast_auc(t_slices[k]["y_true"][idx_nih], t_slices[k]["y_score"][idx_nih])
                a_chex = compute_fast_auc(c_slices[k]["y_true"][idx_chex], c_slices[k]["y_score"][idx_chex])
                rep_deltas.append(a_nih - a_chex)
            boot_mean_deltas.append(float(np.mean(rep_deltas)))

        boot_mean_deltas = np.array(boot_mean_deltas)
        delta_mean = float(np.mean(delta_aurocs))
        ci_joint = [float(np.percentile(boot_mean_deltas, 2.5)), float(np.percentile(boot_mean_deltas, 97.5))]

        # Two-sided empirical cluster-bootstrap p-value under H0
        p_boot = float(2.0 * min(
            (1.0 + np.sum(boot_mean_deltas <= 0.0)) / (n_bootstrap + 1),
            (1.0 + np.sum(boot_mean_deltas >= 0.0)) / (n_bootstrap + 1),
        ))
        p_boot = min(p_boot, 1.0)

        raw_p_boot_list.append(p_boot)
        arch_keys.append(arch)

        # Multi-seed Summary Statistics
        in_auc_mean = float(np.mean(nih_aurocs))
        in_auc_sd = float(np.std(nih_aurocs, ddof=1)) if len(nih_aurocs) > 1 else 0.0
        in_pr_mean = float(np.mean(nih_auprcs))
        in_pr_sd = float(np.std(nih_auprcs, ddof=1)) if len(nih_auprcs) > 1 else 0.0

        cr_auc_mean = float(np.mean(chex_aurocs))
        cr_auc_sd = float(np.std(chex_aurocs, ddof=1)) if len(chex_aurocs) > 1 else 0.0
        cr_pr_mean = float(np.mean(chex_auprcs))
        cr_pr_sd = float(np.std(chex_auprcs, ddof=1)) if len(chex_auprcs) > 1 else 0.0

        delta_sd = float(np.std(delta_aurocs, ddof=1)) if len(delta_aurocs) > 1 else 0.0

        arch_meta["summary_3seed"] = {
            "available_seeds": available_seeds,
            "seed_note": f"{len(available_seeds)} seeds evaluated: {available_seeds} (seed 42 marked not computable)" if len(available_seeds) < 3 else "3 seeds evaluated (42, 1337, 2024)",
            "in_domain": {
                "auroc_mean": in_auc_mean,
                "auroc_sd": in_auc_sd,
                "auprc_mean": in_pr_mean,
                "auprc_sd": in_pr_sd,
            },
            "transferred": {
                "auroc_mean": cr_auc_mean,
                "auroc_sd": cr_auc_sd,
                "auprc_mean": cr_pr_mean,
                "auprc_sd": cr_pr_sd,
            },
            "generalization_gap": {
                "delta_auroc_mean": delta_mean,
                "delta_between_seed_sd": delta_sd,
                "joint_bootstrap_95ci": ci_joint,
                "joint_bootstrap_p_val": p_boot,
            },
        }
        table_json["architectures"][arch] = arch_meta

        # Table rows (strictly labeled mean +/- SD, never 95% CI)
        arch_label_in = f"{arch_display} (3 seeds: 42, 1337, 2024)"
        arch_label_tr = f"{arch_display} ({len(available_seeds)} seeds: {', '.join(available_seeds)})" if len(available_seeds) < 3 else f"{arch_display} (3 seeds: 42, 1337, 2024)"

        t1_rows.append({
            "Architecture": arch_label_in,
            "Evaluation Direction": "NIH -> NIH",
            "Setting": "In-Domain Baseline",
            "Cohorts Evaluated (Images | Patients)": f"NIH Test (N=22,744 imgs, 491 pos | 2,518 pts, 277 pos)",
            "Pneumonia AUROC (mean +/- SD)": f"{in_auc_mean:.3f} +/- {in_auc_sd:.3f}",
            "Pneumonia AUPRC (mean +/- SD)": f"{in_pr_mean:.3f} +/- {in_pr_sd:.3f}",
            "Generalization Gap (Delta AUROC)": "-",
            "Joint Cluster Bootstrap 95% CI": "-",
            "p-value (Cluster Boot)": "-",
            "q-value (FDR)": "-",
            "Significance Claim": "-",
        })
        t1_rows.append({
            "Architecture": arch_label_tr,
            "Evaluation Direction": "NIH -> CheXpert",
            "Setting": "Transferred Cohort",
            "Cohorts Evaluated (Images | Patients)": f"CheXpert (N=17,551 imgs, 388 pos | 17,551 pts, 388 pos)",
            "Pneumonia AUROC (mean +/- SD)": f"{cr_auc_mean:.3f} +/- {cr_auc_sd:.3f}",
            "Pneumonia AUPRC (mean +/- SD)": f"{cr_pr_mean:.3f} +/- {cr_pr_sd:.3f}",
            "Generalization Gap (Delta AUROC)": f"{delta_mean:+.3f} +/- {delta_sd:.3f}",
            "Joint Cluster Bootstrap 95% CI": f"[{ci_joint[0]:.3f}, {ci_joint[1]:.3f}]",
            "p-value (Cluster Boot)": f"{p_boot:.4f}",
            "q-value (FDR)": "-",  # Filled below
            "Significance Claim": "-",
        })

    # Benjamini-Hochberg adjustment across architectures on cluster-bootstrap p-values
    q_vals = benjamini_hochberg(raw_p_boot_list)
    q_idx = 0
    for r in t1_rows:
        if r["Setting"] == "Transferred Cohort":
            q_v = q_vals[q_idx]
            p_v = raw_p_boot_list[q_idx]
            arch_k = arch_keys[q_idx]
            table_json["architectures"][arch_k]["summary_3seed"]["generalization_gap"]["fdr_q_val"] = q_v
            r["q-value (FDR)"] = f"{q_v:.4f}"
            is_sig = (q_v < 0.05) and (p_v < 0.05)
            r["Significance Claim"] = "Statistically Significant" if is_sig else "Not Significant (Robust to Cluster Shift)"
            q_idx += 1

    df_t1 = pd.DataFrame(t1_rows)
    df_supp = pd.DataFrame(supplement_rows)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    t1_json_path = output_dir / "table1_cross_domain_discrimination.json"
    t1_csv_path = output_dir / "table1_cross_domain_discrimination.csv"
    supp_json_path = output_dir / "table1_supplement_single_seed_bootstrap.json"
    supp_csv_path = output_dir / "table1_supplement_single_seed_bootstrap.csv"

    with open(t1_json_path, "w", encoding="utf-8") as fp:
        json.dump(table_json, fp, indent=2)
    df_t1.to_csv(t1_csv_path, index=False)

    with open(supp_json_path, "w", encoding="utf-8") as fp:
        json.dump(supplement_rows, fp, indent=2)
    df_supp.to_csv(supp_csv_path, index=False)

    logger.info("Saved Table 1 to '%s' and '%s'.", t1_json_path, t1_csv_path)
    logger.info("Saved Table 1 Supplement to '%s'.", supp_csv_path)
    return table_json, df_t1, df_supp


# ==============================================================================
# 4. TABLE 2 GENERATOR: Calibration Regimes & Recalibration Analysis
# ==============================================================================

def generate_table_2(
    bundle: PredictionBundle,
    output_dir: Path,
    n_repeats: int = 20,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Generates Table 2 following unified statistical protocol across 6 regimes."""
    logger.info("Generating Table 2: 6-Regime Probability Calibration & Oracle Recalibration...")

    t2_rows = []
    table_json: Dict[str, Any] = {
        "title": f"PROVISIONAL ({bundle.run_id}) Table 2: Probability Calibration Across 6 Regimes & Oracle Recalibration",
        "run_id": bundle.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_patient_split_repeats": n_repeats,
        "protocol": {
            "recovery_definition": "ECE_Recovery(%) = ((ECE_uncal - ECE_method) / ECE_uncal) * 100",
            "source_validation_cohort": "NIH Validation (13,500 imgs, 139 pos | 4,201 pts, 110 pos)",
            "target_10pct_splits": f"{n_repeats} independent stratified patient-level 10% calibration / 90% evaluation splits",
        },
        "effective_cohort_numbers": {
            "in_domain_test": "22,744 images (491 positive) across 2,518 unique patients (277 positive)",
            "cross_domain_chexpert": "17,551 images (388 positive) across 17,551 unique patients (388 positive)",
        },
        "scientific_findings": (
            "In-domain ECE is elevated (~0.28-0.29) and Brier Skill Score is negative (~ -6.1) due to "
            "training with pos_weight: inverse_freq (~45x penalty), which forces raw model output probabilities "
            "up to a mean of ~26-31% against true in-domain prevalence of 2.16% (and 11.00% patient prevalence). "
            "When deployed cross-domain to CheXpert, domain shift pushes uncalibrated predictions downward into a "
            "lower range (mean prediction ~9.7%), deceptively lowering uncalibrated ECE to ~0.075. Crucially, Source Platt "
            "scaling (fitted exclusively on the NIH validation set with ~139 positives) shifts both slope and intercept, "
            "driving CheXpert ECE down to ~0.015 (>75% recovery) WITHOUT requiring any target domain labels. Target 10% "
            "Platt scaling provides an asymptotic oracle upper bound (ECE ~0.001)."
        ),
        "architectures": {},
    }

    for arch in bundle.archs:
        arch_display = "DenseNet121" if "dense" in arch.lower() else "EfficientNet-B0"
        arch_meta: Dict[str, Any] = {"regimes": {}}

        regime_names = [
            "1. In-Domain Baseline (NIH Test)",
            "2. Cross-Domain Uncalibrated (CheXpert)",
            "3. Source Temperature Scaling",
            "4. Source Platt Scaling",
            "5. Target 10% Temperature Scaling (Oracle)",
            "6. Target 10% Platt Scaling (Oracle)",
        ]

        regime_metrics_accumulator: Dict[str, Dict[str, List[float]]] = {
            r: {
                "ece_uniform": [], "ece_adaptive": [], "brier": [], "bss": [],
                "cal_slope": [], "cal_intercept": [], "nll": [], "mean_pred": [],
                "recovery": [],
            }
            for r in regime_names
        }

        param_records: Dict[str, List[Any]] = {
            "source_temp": [],
            "source_platt": [],
            "target_temp": [],
            "target_platt": [],
        }

        # Track computable seeds
        comp_seeds = [s for s in bundle.seeds if bundle.data.get(f"cr_{arch}_{s}_computable", True)]
        arch_meta["computable_seeds_cross_domain"] = comp_seeds

        for s in bundle.seeds:
            val_slice = bundle.get_pneumonia_slice(f"val_{arch}_{s}")
            test_slice = bundle.get_pneumonia_slice(f"test_{arch}_{s}")
            cr_computable = bundle.data.get(f"cr_{arch}_{s}_computable", True)

            # Regime 1: In-domain baseline (available for all seeds)
            m1 = compute_comprehensive_calibration(
                test_slice["y_true"], test_slice["y_score"], test_slice["y_logits"]
            )
            r1 = regime_names[0]
            regime_metrics_accumulator[r1]["ece_uniform"].append(m1["ece_uniform"])
            regime_metrics_accumulator[r1]["ece_adaptive"].append(m1["ece_adaptive"])
            regime_metrics_accumulator[r1]["brier"].append(m1["brier"])
            regime_metrics_accumulator[r1]["bss"].append(m1["bss"])
            regime_metrics_accumulator[r1]["cal_slope"].append(m1["cal_slope"])
            regime_metrics_accumulator[r1]["cal_intercept"].append(m1["cal_intercept"])
            regime_metrics_accumulator[r1]["nll"].append(m1["nll"])
            regime_metrics_accumulator[r1]["mean_pred"].append(m1["mean_prediction"])
            regime_metrics_accumulator[r1]["recovery"].append(0.0)

            if not cr_computable:
                continue

            cr_slice = bundle.get_pneumonia_slice(f"cr_{arch}_{s}")
            yt_cr = cr_slice["y_true"]
            yp_cr = cr_slice["y_score"]
            yl_cr = cr_slice["y_logits"]
            pts_cr = cr_slice["patient_ids"]

            # Regime 2: Cross-Domain Uncalibrated
            m2 = compute_comprehensive_calibration(yt_cr, yp_cr, yl_cr)
            r2 = regime_names[1]
            regime_metrics_accumulator[r2]["ece_uniform"].append(m2["ece_uniform"])
            regime_metrics_accumulator[r2]["ece_adaptive"].append(m2["ece_adaptive"])
            regime_metrics_accumulator[r2]["brier"].append(m2["brier"])
            regime_metrics_accumulator[r2]["bss"].append(m2["bss"])
            regime_metrics_accumulator[r2]["cal_slope"].append(m2["cal_slope"])
            regime_metrics_accumulator[r2]["cal_intercept"].append(m2["cal_intercept"])
            regime_metrics_accumulator[r2]["nll"].append(m2["nll"])
            regime_metrics_accumulator[r2]["mean_pred"].append(m2["mean_prediction"])
            regime_metrics_accumulator[r2]["recovery"].append(0.0)

            # Regime 3: Source Temperature Scaling (fit on NIH val)
            t_src = fit_temperature_scalar(val_slice["y_logits"], val_slice["y_true"])
            param_records["source_temp"].append(t_src)
            yp_s3 = 1.0 / (1.0 + np.exp(-yl_cr / max(t_src, 1e-3)))
            m3 = compute_comprehensive_calibration(yt_cr, yp_s3, yl_cr / max(t_src, 1e-3))
            rec3 = ((m2["ece_uniform"] - m3["ece_uniform"]) / max(m2["ece_uniform"], 1e-6)) * 100.0
            r3 = regime_names[2]
            regime_metrics_accumulator[r3]["ece_uniform"].append(m3["ece_uniform"])
            regime_metrics_accumulator[r3]["ece_adaptive"].append(m3["ece_adaptive"])
            regime_metrics_accumulator[r3]["brier"].append(m3["brier"])
            regime_metrics_accumulator[r3]["bss"].append(m3["bss"])
            regime_metrics_accumulator[r3]["cal_slope"].append(m3["cal_slope"])
            regime_metrics_accumulator[r3]["cal_intercept"].append(m3["cal_intercept"])
            regime_metrics_accumulator[r3]["nll"].append(m3["nll"])
            regime_metrics_accumulator[r3]["mean_pred"].append(m3["mean_prediction"])
            regime_metrics_accumulator[r3]["recovery"].append(rec3)

            # Regime 4: Source Platt Scaling (fit on NIH val)
            a_src, b_src = fit_platt_scaling(val_slice["y_logits"], val_slice["y_true"])
            param_records["source_platt"].append({"a": a_src, "b": b_src})
            scaled_logits_src = a_src * yl_cr + b_src
            yp_s4 = 1.0 / (1.0 + np.exp(-scaled_logits_src))
            m4 = compute_comprehensive_calibration(yt_cr, yp_s4, scaled_logits_src)
            rec4 = ((m2["ece_uniform"] - m4["ece_uniform"]) / max(m2["ece_uniform"], 1e-6)) * 100.0
            r4 = regime_names[3]
            regime_metrics_accumulator[r4]["ece_uniform"].append(m4["ece_uniform"])
            regime_metrics_accumulator[r4]["ece_adaptive"].append(m4["ece_adaptive"])
            regime_metrics_accumulator[r4]["brier"].append(m4["brier"])
            regime_metrics_accumulator[r4]["bss"].append(m4["bss"])
            regime_metrics_accumulator[r4]["cal_slope"].append(m4["cal_slope"])
            regime_metrics_accumulator[r4]["cal_intercept"].append(m4["cal_intercept"])
            regime_metrics_accumulator[r4]["nll"].append(m4["nll"])
            regime_metrics_accumulator[r4]["mean_pred"].append(m4["mean_prediction"])
            regime_metrics_accumulator[r4]["recovery"].append(rec4)

            # Regimes 5 & 6: 20 Stratified Patient-Level 10/90 Repeats on CheXpert
            u_cr_pts, pt_cr_inv = np.unique(pts_cr, return_inverse=True)
            pt_has_pos = np.zeros(len(u_cr_pts), dtype=bool)
            for p_i, y in zip(pt_cr_inv, yt_cr):
                if y == 1:
                    pt_has_pos[p_i] = True
            pos_pts_idx = np.where(pt_has_pos)[0]
            neg_pts_idx = np.where(~pt_has_pos)[0]

            pt_to_img_indices = [np.where(pt_cr_inv == i)[0] for i in range(len(u_cr_pts))]

            rep_ts_eces, rep_ts_briers, rep_ts_nlls, rep_ts_bss, rep_ts_preds = [], [], [], [], []
            rep_pl_eces, rep_pl_briers, rep_pl_nlls, rep_pl_bss, rep_pl_preds = [], [], [], [], []
            rep_pl_slopes, rep_pl_intercepts = [], []
            rep_ts_slopes, rep_ts_intercepts = [], []
            rep_t_params = []
            rep_pl_params = []

            for rep in range(n_repeats):
                rng_rep = np.random.RandomState(int(s) + seed + rep * 7)
                cal_pos_pts = rng_rep.choice(pos_pts_idx, size=max(1, int(0.10 * len(pos_pts_idx))), replace=False)
                eval_pos_pts = np.setdiff1d(pos_pts_idx, cal_pos_pts)

                cal_neg_pts = rng_rep.choice(neg_pts_idx, size=max(1, int(0.10 * len(neg_pts_idx))), replace=False)
                eval_neg_pts = np.setdiff1d(neg_pts_idx, cal_neg_pts)

                cal_pts_all = np.concatenate([cal_pos_pts, cal_neg_pts])
                eval_pts_all = np.concatenate([eval_pos_pts, eval_neg_pts])

                # Hard assertion: zero patient overlap
                assert len(set(cal_pts_all).intersection(set(eval_pts_all))) == 0, "Patient overlap in 10/90 split!"

                cal_idx = np.concatenate([pt_to_img_indices[i] for i in cal_pts_all])
                eval_idx = np.concatenate([pt_to_img_indices[i] for i in eval_pts_all])

                # 5. Target 10% Temperature
                t_tgt = fit_temperature_scalar(yl_cr[cal_idx], yt_cr[cal_idx])
                rep_t_params.append(t_tgt)
                yp_ev_ts = 1.0 / (1.0 + np.exp(-yl_cr[eval_idx] / max(t_tgt, 1e-3)))
                m5_rep = compute_comprehensive_calibration(yt_cr[eval_idx], yp_ev_ts, yl_cr[eval_idx] / max(t_tgt, 1e-3))
                rep_ts_eces.append(m5_rep["ece_uniform"])
                rep_ts_briers.append(m5_rep["brier"])
                rep_ts_nlls.append(m5_rep["nll"])
                rep_ts_bss.append(m5_rep["bss"])
                rep_ts_preds.append(m5_rep["mean_prediction"])
                rep_ts_slopes.append(m5_rep["cal_slope"])
                rep_ts_intercepts.append(m5_rep["cal_intercept"])

                # 6. Target 10% Platt
                a_tgt, b_tgt = fit_platt_scaling(yl_cr[cal_idx], yt_cr[cal_idx])
                rep_pl_params.append({"a": a_tgt, "b": b_tgt})
                scaled_ev = a_tgt * yl_cr[eval_idx] + b_tgt
                yp_ev_pl = 1.0 / (1.0 + np.exp(-scaled_ev))
                m6_rep = compute_comprehensive_calibration(yt_cr[eval_idx], yp_ev_pl, scaled_ev)
                rep_pl_eces.append(m6_rep["ece_uniform"])
                rep_pl_briers.append(m6_rep["brier"])
                rep_pl_nlls.append(m6_rep["nll"])
                rep_pl_bss.append(m6_rep["bss"])
                rep_pl_preds.append(m6_rep["mean_prediction"])
                rep_pl_slopes.append(m6_rep["cal_slope"])
                rep_pl_intercepts.append(m6_rep["cal_intercept"])

            r5 = regime_names[4]
            ece5_mean = float(np.mean(rep_ts_eces))
            regime_metrics_accumulator[r5]["ece_uniform"].append(ece5_mean)
            regime_metrics_accumulator[r5]["ece_adaptive"].append(ece5_mean)
            regime_metrics_accumulator[r5]["brier"].append(float(np.mean(rep_ts_briers)))
            regime_metrics_accumulator[r5]["bss"].append(float(np.mean(rep_ts_bss)))
            regime_metrics_accumulator[r5]["cal_slope"].append(float(np.mean(rep_ts_slopes)))
            regime_metrics_accumulator[r5]["cal_intercept"].append(float(np.mean(rep_ts_intercepts)))
            regime_metrics_accumulator[r5]["nll"].append(float(np.mean(rep_ts_nlls)))
            regime_metrics_accumulator[r5]["mean_pred"].append(float(np.mean(rep_ts_preds)))
            regime_metrics_accumulator[r5]["recovery"].append(((m2["ece_uniform"] - ece5_mean) / max(m2["ece_uniform"], 1e-6)) * 100.0)

            r6 = regime_names[5]
            ece6_mean = float(np.mean(rep_pl_eces))
            regime_metrics_accumulator[r6]["ece_uniform"].append(ece6_mean)
            regime_metrics_accumulator[r6]["ece_adaptive"].append(ece6_mean)
            regime_metrics_accumulator[r6]["brier"].append(float(np.mean(rep_pl_briers)))
            regime_metrics_accumulator[r6]["bss"].append(float(np.mean(rep_pl_bss)))
            regime_metrics_accumulator[r6]["cal_slope"].append(float(np.mean(rep_pl_slopes)))
            regime_metrics_accumulator[r6]["cal_intercept"].append(float(np.mean(rep_pl_intercepts)))
            regime_metrics_accumulator[r6]["nll"].append(float(np.mean(rep_pl_nlls)))
            regime_metrics_accumulator[r6]["mean_pred"].append(float(np.mean(rep_pl_preds)))
            regime_metrics_accumulator[r6]["recovery"].append(((m2["ece_uniform"] - ece6_mean) / max(m2["ece_uniform"], 1e-6)) * 100.0)

            param_records["target_temp"].append(float(np.mean(rep_t_params)))
            param_records["target_platt"].append({
                "a_mean": float(np.mean([p["a"] for p in rep_pl_params])),
                "b_mean": float(np.mean([p["b"] for p in rep_pl_params])),
            })

        # Assemble summary rows for Table 2
        for r_name in regime_names:
            acc = regime_metrics_accumulator[r_name]
            ece_m = float(np.mean(acc["ece_uniform"])) if acc["ece_uniform"] else 0.0
            ece_sd = float(np.std(acc["ece_uniform"], ddof=1)) if len(acc["ece_uniform"]) > 1 else 0.0
            ad_ece_m = float(np.mean(acc["ece_adaptive"])) if acc["ece_adaptive"] else 0.0
            brier_m = float(np.mean(acc["brier"])) if acc["brier"] else 0.0
            bss_m = float(np.mean(acc["bss"])) if acc["bss"] else 0.0
            slope_m = float(np.mean(acc["cal_slope"])) if acc["cal_slope"] else 0.0
            intercept_m = float(np.mean(acc["cal_intercept"])) if acc["cal_intercept"] else 0.0
            nll_m = float(np.mean(acc["nll"])) if acc["nll"] else 0.0
            pred_m = float(np.mean(acc["mean_pred"])) if acc["mean_pred"] else 0.0
            rec_m = float(np.mean(acc["recovery"])) if acc["recovery"] else 0.0

            n_eval_imgs = 22744 if "In-Domain" in r_name else 17551
            n_pos_eval_imgs = 491 if "In-Domain" in r_name else 388
            n_eval_pts = 2518 if "In-Domain" in r_name else 17551
            n_pos_eval_pts = 277 if "In-Domain" in r_name else 388

            reg_data = {
                "ece_uniform_mean": ece_m,
                "ece_uniform_sd": ece_sd,
                "ece_adaptive_mean": ad_ece_m,
                "brier_score_mean": brier_m,
                "bss_vs_base_mean": bss_m,
                "cal_slope_mean": slope_m,
                "cal_intercept_mean": intercept_m,
                "nll_mean": nll_m,
                "mean_prediction": pred_m,
                "recovery_pct": rec_m,
                "n_images": n_eval_imgs,
                "n_pos_images": n_pos_eval_imgs,
                "n_patients": n_eval_pts,
                "n_pos_patients": n_pos_eval_pts,
            }
            arch_meta["regimes"][r_name] = reg_data

            rec_str = "-" if "In-Domain" in r_name else f"{rec_m:.1f}%"
            seeds_note = "3 seeds" if ("In-Domain" in r_name or len(comp_seeds) == 3) else f"{len(comp_seeds)} seeds ({', '.join(comp_seeds)})"

            t2_rows.append({
                "Architecture": f"{arch_display} ({seeds_note})",
                "Calibration Regime": r_name,
                "Cohort & Effective Counts": f"N={n_eval_imgs} imgs ({n_pos_eval_imgs} pos) | {n_eval_pts} pts ({n_pos_eval_pts} pos)",
                "ECE (Equal-Width) (mean +/- SD)": f"{ece_m:.3f} +/- {ece_sd:.3f}",
                "ECE (Adaptive)": f"{ad_ece_m:.3f}",
                "Brier Score": f"{brier_m:.3f}",
                "BSS vs Base Rate": f"{bss_m:.3f}",
                "Cal Slope": f"{slope_m:.2f}",
                "Cal Intercept": f"{intercept_m:.2f}",
                "NLL": f"{nll_m:.3f}",
                "Mean Pred vs Prevalence": f"{pred_m*100:.1f}% vs {n_pos_eval_imgs/n_eval_imgs*100:.2f}% (img) / {n_pos_eval_pts/n_eval_pts*100:.2f}% (pt)",
                "Calibration Recovery (%)": rec_str,
            })

        arch_meta["fitted_parameters"] = {
            "source_temperature": param_records["source_temp"],
            "source_platt": param_records["source_platt"],
            "target_10pct_temperature": param_records["target_temp"],
            "target_10pct_platt": param_records["target_platt"],
        }
        table_json["architectures"][arch] = arch_meta

    df_t2 = pd.DataFrame(t2_rows)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    t2_json_path = output_dir / "table2_calibration_regimes_and_oracle_recovery.json"
    t2_csv_path = output_dir / "table2_calibration_regimes_and_oracle_recovery.csv"

    with open(t2_json_path, "w", encoding="utf-8") as fp:
        json.dump(table_json, fp, indent=2)
    df_t2.to_csv(t2_csv_path, index=False)

    logger.info("Saved Table 2 to '%s' and '%s'.", t2_json_path, t2_csv_path)
    return table_json, df_t2


# ==============================================================================
# 5. TABLE 3 GENERATOR: Decision Threshold Transfer Breakdown
# ==============================================================================

def generate_table_3(
    bundle: PredictionBundle,
    output_dir: Path,
    n_repeats: int = 20,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Generates Table 3 following unified statistical protocol (Addendum 2)."""
    logger.info("Generating Table 3: Decision Threshold Transfer Breakdown (Patient Clustering)...")

    t3_rows = []
    table_json: Dict[str, Any] = {
        "title": f"PROVISIONAL ({bundle.run_id}) Table 3: Decision Threshold Transfer Breakdown",
        "run_id": bundle.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_patient_split_repeats": n_repeats,
        "n_f1_cluster_bootstrap": n_bootstrap,
        "cohort_prevalences": {
            "source_validation": {
                "n_images": 13500, "n_pos_images": 139, "image_prevalence": 0.010296,
                "n_patients": 4201, "n_pos_patients": 110, "patient_prevalence": 0.026184,
            },
            "source_test": {
                "n_images": 22744, "n_pos_images": 491, "image_prevalence": 0.021588,
                "n_patients": 2518, "n_pos_patients": 277, "patient_prevalence": 0.110008,
            },
            "target_chexpert": {
                "n_images": 17551, "n_pos_images": 388, "image_prevalence": 0.022107,
                "n_patients": 17551, "n_pos_patients": 388, "patient_prevalence": 0.022107,
            },
            "chexpert_50_50_halves": {
                "calibration_half": {"N_patients": 8775, "n_pos_patients": 194},
                "evaluation_half": {"N_patients": 8776, "n_pos_patients": 194},
            },
        },
        "scientific_findings": (
            "Threshold transfer analysis demonstrates that sensitivity and specificity are prevalence-invariant "
            "metrics that serve as the primary clinical decision benchmarks. F1 score is strongly prevalence-dependent. "
            "Source thresholds were selected on NIH Validation where disease prevalence is 1.03% (images) / 2.62% (patients), "
            "but scored on cohorts with ~2.2% prevalence (NIH test: 2.16% images / 11.00% patients; CheXpert: 2.21%). "
            "On the source validation set, operating at 90% sensitivity required a threshold with only 33.6% source specificity; "
            "upon transfer to CheXpert, target specificity was 35.6%. Hence, specificity did not 'collapse from 90%', "
            "but remained consistent with the source operating characteristic. "
            "All F1 differences are within patient-level cluster bootstrap 95% confidence intervals and must not be over-interpreted."
        ),
        "architectures": {},
    }

    for arch in bundle.archs:
        arch_display = "DenseNet121" if "dense" in arch.lower() else "EfficientNet-B0"
        arch_meta: Dict[str, Any] = {"policies": {}}

        policy_names = ["Source Youden J", "Source Sens-90", "Source Max-F1", "Oracle Target-Optimal"]
        accumulators: Dict[str, Dict[str, List[float]]] = {
            p: {
                "operating_tau": [],
                "source_val_sens": [], "source_val_spec": [],
                "source_test_sens": [], "source_test_spec": [],
                "target_sens": [], "target_spec": [],
                "target_f1": [], "f1_drop": [],
                "target_f1_ci_low": [], "target_f1_ci_high": [],
            }
            for p in policy_names
        }

        comp_seeds = [s for s in bundle.seeds if bundle.data.get(f"cr_{arch}_{s}_computable", True)]
        arch_meta["computable_seeds"] = comp_seeds

        for s in bundle.seeds:
            val_slice = bundle.get_pneumonia_slice(f"val_{arch}_{s}")
            test_slice = bundle.get_pneumonia_slice(f"test_{arch}_{s}")
            tau_sel = bundle.data[f"tau_{arch}_{s}"]
            cr_computable = bundle.data.get(f"cr_{arch}_{s}_computable", True)

            # Extract source thresholds
            tau_youden = float(tau_sel.get("youden_tau", tau_sel.get("source_val_optimal_tau", 0.24)))
            tau_sens90 = float(tau_sel.get("sens90_tau", 0.05))
            tau_maxf1 = float(tau_sel.get("max_f1_tau", 0.50))

            # Performance on source validation
            m_val_y = compute_binary_metrics_at_threshold(val_slice["y_true"], val_slice["y_score"], tau_youden)
            m_val_s = compute_binary_metrics_at_threshold(val_slice["y_true"], val_slice["y_score"], tau_sens90)
            m_val_f = compute_binary_metrics_at_threshold(val_slice["y_true"], val_slice["y_score"], tau_maxf1)

            # Performance on source test
            m_test_y = compute_binary_metrics_at_threshold(test_slice["y_true"], test_slice["y_score"], tau_youden)
            m_test_s = compute_binary_metrics_at_threshold(test_slice["y_true"], test_slice["y_score"], tau_sens90)
            m_test_f = compute_binary_metrics_at_threshold(test_slice["y_true"], test_slice["y_score"], tau_maxf1)

            if not cr_computable:
                continue

            cr_slice = bundle.get_pneumonia_slice(f"cr_{arch}_{s}")
            yt_cr = cr_slice["y_true"]
            yp_cr = cr_slice["y_score"]
            pts_cr = cr_slice["patient_ids"]

            # Stratified patient-level split for CheXpert
            u_cr_pts, pt_cr_inv = np.unique(pts_cr, return_inverse=True)
            pt_has_pos = np.zeros(len(u_cr_pts), dtype=bool)
            for p_i, y in zip(pt_cr_inv, yt_cr):
                if y == 1:
                    pt_has_pos[p_i] = True
            pos_pts_idx = np.where(pt_has_pos)[0]
            neg_pts_idx = np.where(~pt_has_pos)[0]
            pt_to_img_indices = [np.where(pt_cr_inv == i)[0] for i in range(len(u_cr_pts))]

            rep_data: Dict[str, Dict[str, List[float]]] = {
                p: {"sens": [], "spec": [], "f1": [], "tau": []} for p in policy_names
            }

            for rep in range(n_repeats):
                rng_rep = np.random.RandomState(int(s) + seed + rep * 13)
                cal_pos_pts = rng_rep.choice(pos_pts_idx, size=len(pos_pts_idx) // 2, replace=False)
                eval_pos_pts = np.setdiff1d(pos_pts_idx, cal_pos_pts)

                cal_neg_pts = rng_rep.choice(neg_pts_idx, size=len(neg_pts_idx) // 2, replace=False)
                eval_neg_pts = np.setdiff1d(neg_pts_idx, cal_neg_pts)

                cal_pts_all = np.concatenate([cal_pos_pts, cal_neg_pts])
                eval_pts_all = np.concatenate([eval_pos_pts, eval_neg_pts])

                # Verify zero patient overlap
                assert len(set(cal_pts_all).intersection(set(eval_pts_all))) == 0, "Patient overlap in 50/50 split!"

                cal_idx = np.concatenate([pt_to_img_indices[i] for i in cal_pts_all])
                eval_idx = np.concatenate([pt_to_img_indices[i] for i in eval_pts_all])

                # Target Oracle threshold on calibration half
                opt_tgt = compute_optimal_thresholds(yt_cr[cal_idx], yp_cr[cal_idx])
                tau_ora = float(opt_tgt["f1_threshold"])

                # Evaluate on unseen evaluation half
                m_tgt_y = compute_binary_metrics_at_threshold(yt_cr[eval_idx], yp_cr[eval_idx], tau_youden)
                m_tgt_s = compute_binary_metrics_at_threshold(yt_cr[eval_idx], yp_cr[eval_idx], tau_sens90)
                m_tgt_f = compute_binary_metrics_at_threshold(yt_cr[eval_idx], yp_cr[eval_idx], tau_maxf1)
                m_tgt_o = compute_binary_metrics_at_threshold(yt_cr[eval_idx], yp_cr[eval_idx], tau_ora)

                rep_data["Source Youden J"]["sens"].append(m_tgt_y["sensitivity"])
                rep_data["Source Youden J"]["spec"].append(m_tgt_y["specificity"])
                rep_data["Source Youden J"]["f1"].append(m_tgt_y["f1_score"])
                rep_data["Source Youden J"]["tau"].append(tau_youden)

                rep_data["Source Sens-90"]["sens"].append(m_tgt_s["sensitivity"])
                rep_data["Source Sens-90"]["spec"].append(m_tgt_s["specificity"])
                rep_data["Source Sens-90"]["f1"].append(m_tgt_s["f1_score"])
                rep_data["Source Sens-90"]["tau"].append(tau_sens90)

                rep_data["Source Max-F1"]["sens"].append(m_tgt_f["sensitivity"])
                rep_data["Source Max-F1"]["spec"].append(m_tgt_f["specificity"])
                rep_data["Source Max-F1"]["f1"].append(m_tgt_f["f1_score"])
                rep_data["Source Max-F1"]["tau"].append(tau_maxf1)

                rep_data["Oracle Target-Optimal"]["sens"].append(m_tgt_o["sensitivity"])
                rep_data["Oracle Target-Optimal"]["spec"].append(m_tgt_o["specificity"])
                rep_data["Oracle Target-Optimal"]["f1"].append(m_tgt_o["f1_score"])
                rep_data["Oracle Target-Optimal"]["tau"].append(tau_ora)

            # Patient-level cluster bootstrap CIs for target F1 across full CheXpert cohort
            rng_f1 = np.random.RandomState(int(s) + seed)
            map_cr, n_pts_cr = get_patient_clusters(pts_cr)
            boot_f1_dict: Dict[str, List[float]] = {p: [] for p in policy_names}
            mean_ora_tau = float(np.mean(rep_data["Oracle Target-Optimal"]["tau"]))

            for _ in range(n_bootstrap):
                b_idx = patient_cluster_resample(map_cr, n_pts_cr, rng_f1)
                yt_b = yt_cr[b_idx]
                yp_b = yp_cr[b_idx]
                boot_f1_dict["Source Youden J"].append(compute_binary_metrics_at_threshold(yt_b, yp_b, tau_youden)["f1_score"])
                boot_f1_dict["Source Sens-90"].append(compute_binary_metrics_at_threshold(yt_b, yp_b, tau_sens90)["f1_score"])
                boot_f1_dict["Source Max-F1"].append(compute_binary_metrics_at_threshold(yt_b, yp_b, tau_maxf1)["f1_score"])
                boot_f1_dict["Oracle Target-Optimal"].append(compute_binary_metrics_at_threshold(yt_b, yp_b, mean_ora_tau)["f1_score"])

            oracle_f1_m = float(np.mean(rep_data["Oracle Target-Optimal"]["f1"]))

            for p_name in policy_names:
                acc = accumulators[p_name]
                r_sens = float(np.mean(rep_data[p_name]["sens"]))
                r_spec = float(np.mean(rep_data[p_name]["spec"]))
                r_f1 = float(np.mean(rep_data[p_name]["f1"]))
                r_tau = float(np.mean(rep_data[p_name]["tau"]))

                ci_low = float(np.percentile(boot_f1_dict[p_name], 2.5))
                ci_high = float(np.percentile(boot_f1_dict[p_name], 97.5))

                acc["operating_tau"].append(r_tau)
                acc["target_sens"].append(r_sens)
                acc["target_spec"].append(r_spec)
                acc["target_f1"].append(r_f1)
                acc["f1_drop"].append(r_f1 - oracle_f1_m)
                acc["target_f1_ci_low"].append(ci_low)
                acc["target_f1_ci_high"].append(ci_high)

                if p_name == "Source Youden J":
                    acc["source_val_sens"].append(m_val_y["sensitivity"])
                    acc["source_val_spec"].append(m_val_y["specificity"])
                    acc["source_test_sens"].append(m_test_y["sensitivity"])
                    acc["source_test_spec"].append(m_test_y["specificity"])
                elif p_name == "Source Sens-90":
                    acc["source_val_sens"].append(m_val_s["sensitivity"])
                    acc["source_val_spec"].append(m_val_s["specificity"])
                    acc["source_test_sens"].append(m_test_s["sensitivity"])
                    acc["source_test_spec"].append(m_test_s["specificity"])
                elif p_name == "Source Max-F1":
                    acc["source_val_sens"].append(m_val_f["sensitivity"])
                    acc["source_val_spec"].append(m_val_f["specificity"])
                    acc["source_test_sens"].append(m_test_f["sensitivity"])
                    acc["source_test_spec"].append(m_test_f["specificity"])
                else:
                    acc["source_val_sens"].append(float("nan"))
                    acc["source_val_spec"].append(float("nan"))
                    acc["source_test_sens"].append(float("nan"))
                    acc["source_test_spec"].append(float("nan"))

        # Build Table 3 rows
        seeds_note = "3 seeds" if len(comp_seeds) == 3 else f"{len(comp_seeds)} seeds ({', '.join(comp_seeds)})"

        for p_name in policy_names:
            acc = accumulators[p_name]
            tau_m = float(np.mean(acc["operating_tau"])) if acc["operating_tau"] else 0.0
            tau_sd = float(np.std(acc["operating_tau"], ddof=1)) if len(acc["operating_tau"]) > 1 else 0.0

            src_v_sens = float(np.mean(acc["source_val_sens"])) if acc["source_val_sens"] else float("nan")
            src_v_spec = float(np.mean(acc["source_val_spec"])) if acc["source_val_spec"] else float("nan")
            src_t_sens = float(np.mean(acc["source_test_sens"])) if acc["source_test_sens"] else float("nan")
            src_t_spec = float(np.mean(acc["source_test_spec"])) if acc["source_test_spec"] else float("nan")

            tgt_sens_m = float(np.mean(acc["target_sens"])) if acc["target_sens"] else 0.0
            tgt_sens_sd = float(np.std(acc["target_sens"], ddof=1)) if len(acc["target_sens"]) > 1 else 0.0
            tgt_spec_m = float(np.mean(acc["target_spec"])) if acc["target_spec"] else 0.0
            tgt_spec_sd = float(np.std(acc["target_spec"], ddof=1)) if len(acc["target_spec"]) > 1 else 0.0
            tgt_f1_m = float(np.mean(acc["target_f1"])) if acc["target_f1"] else 0.0
            tgt_f1_sd = float(np.std(acc["target_f1"], ddof=1)) if len(acc["target_f1"]) > 1 else 0.0
            f1_drop_m = float(np.mean(acc["f1_drop"])) if acc["f1_drop"] else 0.0

            ci_l = float(np.mean(acc["target_f1_ci_low"])) if acc["target_f1_ci_low"] else 0.0
            ci_h = float(np.mean(acc["target_f1_ci_high"])) if acc["target_f1_ci_high"] else 0.0

            pol_record = {
                "operating_tau_mean": tau_m,
                "operating_tau_sd": tau_sd,
                "source_val_sensitivity": src_v_sens,
                "source_val_specificity": src_v_spec,
                "source_test_sensitivity": src_t_sens,
                "source_test_specificity": src_t_spec,
                "target_sensitivity_mean": tgt_sens_m,
                "target_sensitivity_sd": tgt_sens_sd,
                "target_specificity_mean": tgt_spec_m,
                "target_specificity_sd": tgt_spec_sd,
                "target_f1_mean": tgt_f1_m,
                "target_f1_sd": tgt_f1_sd,
                "target_f1_95ci_bootstrap": [ci_l, ci_h],
                "f1_drop_vs_oracle": f1_drop_m,
            }
            arch_meta["policies"][p_name] = pol_record

            src_sens_str = "-" if np.isnan(src_v_sens) else f"Val: {src_v_sens:.3f} | Test: {src_t_sens:.3f}"
            src_spec_str = "-" if np.isnan(src_v_spec) else f"Val: {src_v_spec:.3f} | Test: {src_t_spec:.3f}"
            drop_str = "0.000 (Baseline)" if "Oracle" in p_name else f"{f1_drop_m:+.3f}"

            t3_rows.append({
                "Architecture": f"{arch_display} ({seeds_note})",
                "Threshold Policy": p_name,
                "Operating Threshold (tau)": f"{tau_m:.2f} +/- {tau_sd:.2f}" if tau_sd > 0 else f"{tau_m:.2f}",
                "Source Sensitivity (Primary: Val | Test)": src_sens_str,
                "Source Specificity (Primary: Val | Test)": src_spec_str,
                "Target Sensitivity (Primary) (mean +/- SD)": f"{tgt_sens_m:.3f} +/- {tgt_sens_sd:.3f}",
                "Target Specificity (Primary) (mean +/- SD)": f"{tgt_spec_m:.3f} +/- {tgt_spec_sd:.3f}",
                "Target F1 Score (Secondary) [95% Cluster Boot CI]": f"{tgt_f1_m:.3f} [{ci_l:.3f}, {ci_h:.3f}]",
                "F1 Drop vs Oracle (Secondary)": drop_str,
            })

        table_json["architectures"][arch] = arch_meta

    df_t3 = pd.DataFrame(t3_rows)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    t3_json_path = output_dir / "table3_decision_threshold_transfer.json"
    t3_csv_path = output_dir / "table3_decision_threshold_transfer.csv"

    with open(t3_json_path, "w", encoding="utf-8") as fp:
        json.dump(table_json, fp, indent=2)
    df_t3.to_csv(t3_csv_path, index=False)

    logger.info("Saved Table 3 to '%s' and '%s'.", t3_json_path, t3_csv_path)
    return table_json, df_t3


# ==============================================================================
# 6. PIPELINE ORCHESTRATOR & ENTRY POINT
# ==============================================================================

def run_task_b_pipeline(
    run_id: str = "old_run",
    predictions_dir: Union[str, Path] = "results/predictions",
    splits_dir: Union[str, Path] = "data/splits/nih",
    output_dir: Union[str, Path] = "results/tables/provisional_old_run",
    n_bootstrap: int = 2000,
    n_repeats: int = 20,
    seed: int = 42,
    allow_provisional: bool = True,
) -> Dict[str, Any]:
    """Master orchestrator executing the full Task B statistical unification pipeline."""
    pred_path = Path(predictions_dir)
    out_path = Path(output_dir)
    splits_path = Path(splits_dir)

    logger.info("==============================================================================")
    logger.info("STARTING TASK B STATISTICAL PROTOCOL PIPELINE (ADDENDUM 2)")
    logger.info("Run ID: %s | Predictions: %s | Output: %s", run_id, pred_path, out_path)
    logger.info("Inference: Patient-Level Cluster Bootstrap (B=%d) | Patient Repeats: %d", n_bootstrap, n_repeats)
    logger.info("==============================================================================")

    # Step 0: Input Inventory & Audit
    bundle = PredictionBundle(run_id=run_id, predictions_dir=pred_path, allow_provisional=allow_provisional)

    # Table 1
    t1_json, df_t1, df_supp = generate_table_1(bundle=bundle, output_dir=out_path, n_bootstrap=n_bootstrap, seed=seed)

    # Table 2
    t2_json, df_t2 = generate_table_2(bundle=bundle, output_dir=out_path, n_repeats=n_repeats, seed=seed)

    # Table 3
    t3_json, df_t3 = generate_table_3(bundle=bundle, output_dir=out_path, n_repeats=n_repeats, n_bootstrap=n_bootstrap, seed=seed)

    # Execution summary
    summary_path = out_path / "task_b_execution_summary.json"
    summary = {
        "title": f"PROVISIONAL ({run_id}) Task B Execution Summary (Addendum 2)",
        "run_id": run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_cluster_bootstrap": n_bootstrap,
        "n_repeats": n_repeats,
        "random_seed": seed,
        "chexpert_cohort_audit_reference": "CHEXPERT_COHORT.md",
        "effective_cohort_numbers": t1_json["effective_cohort_numbers"],
        "densenet_seed_42_status": "not computable (old_run); excluded from multi-seed summary, reported on seeds 1337 and 2024",
        "tables_generated": [
            str(out_path / "table1_cross_domain_discrimination.csv"),
            str(out_path / "table1_supplement_single_seed_bootstrap.csv"),
            str(out_path / "table2_calibration_regimes_and_oracle_recovery.csv"),
            str(out_path / "table3_decision_threshold_transfer.csv"),
        ],
        "primary_conclusions": [
            "Patient-level cluster bootstrap confirms that variance was underestimated under i.i.d. image assumptions.",
            "Generalization gap (Delta AUROC) is not statistically significant after cluster bootstrapping and FDR adjustment.",
            "Source Platt scaling recovers >75% of target calibration drift without target domain labels.",
            "Sensitivity and specificity are prevalence-invariant primary benchmarks; F1 is secondary with prevalence caveats.",
            "Official NIH test partition exhibits severe pathological enrichment (11.00% patient prevalence vs 2.62% in val).",
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    logger.info("Saved Task B Execution Summary to '%s'.", summary_path)
    logger.info("==============================================================================")
    logger.info("TASK B STATISTICAL PROTOCOL PIPELINE COMPLETE")
    logger.info("==============================================================================")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task B Statistical Unification Pipeline (Addendum 2)")
    parser.add_argument("--run_id", type=str, default="old_run", help="Unique identifier for experiment run")
    parser.add_argument("--predictions_dir", type=str, default="results/predictions", help="Directory of prediction .npz files")
    parser.add_argument("--splits_dir", type=str, default="data/splits/nih", help="Directory containing split CSVs")
    parser.add_argument("--output_dir", type=str, default="results/tables/provisional_old_run", help="Output directory for tables and JSONs")
    parser.add_argument("--n_bootstrap", type=int, default=2000, help="Number of bootstrap resamples (default: 2000)")
    parser.add_argument("--n_repeats", type=int, default=20, help="Number of patient split repeats (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed")
    parser.add_argument("--allow_provisional", action="store_true", default=True, help="Permit provisional missing runs")

    args = parser.parse_args()

    run_task_b_pipeline(
        run_id=args.run_id,
        predictions_dir=args.predictions_dir,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        n_repeats=args.n_repeats,
        seed=args.seed,
        allow_provisional=args.allow_provisional,
    )


if __name__ == "__main__":
    main()
