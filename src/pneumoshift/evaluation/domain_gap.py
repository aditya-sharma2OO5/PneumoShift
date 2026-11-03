"""Domain gap quantification: dataset probe, embedding MMD, and subgroup confounder analyses.

Quantifies penultimate representation divergence and demographic/procedural confounding
(AP vs PA view position, patient sex, patient age) when evaluating across domain boundaries.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.metrics import compute_discrimination_metrics
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.domain_gap")


def train_domain_classifier_probe(
    source_features: np.ndarray,
    target_features: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Trains a logistic regression probe on frozen penultimate features to predict dataset origin.

    An AUROC close to 1.0 (or > 0.95) confirms severe domain separability / penultimate feature shift,
    demonstrating that the encoder retains strong hospital/domain-identifying signatures.

    Uses Stratified K-Fold cross-validation to provide unbiased out-of-fold generalization metrics.

    Args:
        source_features: Penultimate representations from source domain (N_src, D).
        target_features: Penultimate representations from target domain (N_tgt, D).
        n_splits: Number of cross-validation folds (default: 5).
        seed: Random seed for fold splitting and solver.

    Returns:
        Structured dict with cross-validated probe AUROC, accuracy, balanced accuracy,
        fold AUROCs (mean and std), and sample counts.
    """
    n_source = len(source_features)
    n_target = len(target_features)

    if n_source == 0 or n_target == 0:
        return {
            "domain_probe_auroc": float("nan"),
            "probe_auroc": float("nan"),
            "probe_accuracy": float("nan"),
            "probe_balanced_accuracy": float("nan"),
            "fold_aurocs": [],
            "fold_auroc_mean": float("nan"),
            "fold_auroc_std": float("nan"),
            "n_source": n_source,
            "n_target": n_target,
            "n_samples": n_source + n_target,
        }

    X = np.vstack([source_features, target_features]).astype(np.float32)
    y = np.concatenate([np.zeros(n_source, dtype=int), np.ones(n_target, dtype=int)])

    # Determine safe number of folds
    min_class_samples = min(n_source, n_target)
    effective_splits = max(2, min(n_splits, min_class_samples))

    if min_class_samples < 2:
        # Not enough samples for cross-validation; return trivial result
        return {
            "domain_probe_auroc": 0.5,
            "probe_auroc": 0.5,
            "probe_accuracy": 0.5,
            "probe_balanced_accuracy": 0.5,
            "fold_aurocs": [0.5],
            "fold_auroc_mean": 0.5,
            "fold_auroc_std": 0.0,
            "n_source": n_source,
            "n_target": n_target,
            "n_samples": len(X),
        }

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)

    oof_probs = np.zeros(len(X), dtype=np.float32)
    oof_preds = np.zeros(len(X), dtype=int)
    fold_aurocs: List[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(X_tr_scaled, y_tr)

        val_probs = clf.predict_proba(X_val_scaled)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds

        if len(np.unique(y_val)) > 1:
            fold_auc = float(roc_auc_score(y_val, val_probs))
            fold_aurocs.append(fold_auc)

    total_auroc = float(roc_auc_score(y, oof_probs))
    total_acc = float(accuracy_score(y, oof_preds))
    total_bacc = float(balanced_accuracy_score(y, oof_preds))
    mean_fold_auc = float(np.mean(fold_aurocs)) if fold_aurocs else total_auroc
    std_fold_auc = float(np.std(fold_aurocs)) if fold_aurocs else 0.0

    return {
        "domain_probe_auroc": total_auroc,  # Backwards compatibility
        "probe_auroc": total_auroc,
        "probe_accuracy": total_acc,
        "probe_balanced_accuracy": total_bacc,
        "fold_aurocs": fold_aurocs,
        "fold_auroc_mean": mean_fold_auc,
        "fold_auroc_std": std_fold_auc,
        "n_source": n_source,
        "n_target": n_target,
        "n_samples": len(X),
    }


def compute_feature_mmd(
    features_x: np.ndarray,
    features_y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """Computes Maximum Mean Discrepancy (MMD) with RBF kernel between two feature distributions.

    Preserves exact function signature and float return type for backwards compatibility with ablation.py.
    """
    if len(features_x) == 0 or len(features_y) == 0:
        return 0.0

    if gamma is None:
        gamma = 1.0 / max(features_x.shape[1], 1)

    max_samples = 1000
    fx = features_x[:max_samples] if len(features_x) > max_samples else features_x
    fy = features_y[:max_samples] if len(features_y) > max_samples else features_y

    def rbf_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sq_dist = np.sum(a**2, axis=1, keepdims=True) + np.sum(b**2, axis=1, keepdims=True).T - 2 * np.dot(a, b.T)
        return np.exp(-gamma * np.maximum(sq_dist, 0.0))

    k_xx = rbf_kernel(fx, fx)
    k_yy = rbf_kernel(fy, fy)
    k_xy = rbf_kernel(fx, fy)

    mmd_sq = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return float(np.sqrt(max(mmd_sq, 0.0)))


def compute_feature_space_divergence(
    source_features: np.ndarray,
    target_features: np.ndarray,
    max_samples: int = 1000,
    n_permutations: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes multi-kernel RBF MMD, permutation significance test, and centroid distances.

    Uses a mixture of RBF kernels centered around the median heuristic bandwidth
    following standard domain adaptation literature (Gretton et al., Long et al.).

    Args:
        source_features: Penultimate features from source domain (N_src, D).
        target_features: Penultimate features from target domain (N_tgt, D).
        max_samples: Maximum samples per domain to prevent memory exhaustion.
        n_permutations: Number of permutation resamples for empirical p-value.
        seed: Random seed for subsampling and permutation.

    Returns:
        Structured dict with MMD, MMD^2, permutation p-value, centroid Euclidean distance,
        and centroid cosine similarity.
    """
    n_src = len(source_features)
    n_tgt = len(target_features)

    if n_src == 0 or n_tgt == 0:
        return {
            "mmd": 0.0,
            "mmd_squared": 0.0,
            "p_value": 1.0,
            "centroid_euclidean_distance": 0.0,
            "centroid_cosine_similarity": 1.0,
            "centroid_cosine_distance": 0.0,
            "bandwidth_heuristic": 1.0,
        }

    rng = np.random.RandomState(seed)

    # Subsample if necessary
    if n_src > max_samples:
        src_idx = rng.choice(n_src, size=max_samples, replace=False)
        src_feats = source_features[src_idx].astype(np.float64)
    else:
        src_feats = source_features.astype(np.float64)

    if n_tgt > max_samples:
        tgt_idx = rng.choice(n_tgt, size=max_samples, replace=False)
        tgt_feats = target_features[tgt_idx].astype(np.float64)
    else:
        tgt_feats = target_features.astype(np.float64)

    n_x = len(src_feats)
    n_y = len(tgt_feats)

    # Centroid distance metrics
    mean_src = np.mean(src_feats, axis=0)
    mean_tgt = np.mean(tgt_feats, axis=0)
    centroid_euclidean = float(np.linalg.norm(mean_src - mean_tgt))

    norm_src = np.linalg.norm(mean_src)
    norm_tgt = np.linalg.norm(mean_tgt)
    if norm_src > 0 and norm_tgt > 0:
        centroid_cos_sim = float(np.dot(mean_src, mean_tgt) / (norm_src * norm_tgt))
    else:
        centroid_cos_sim = 1.0
    centroid_cos_dist = float(1.0 - centroid_cos_sim)

    # Pairwise squared distances
    pooled = np.vstack([src_feats, tgt_feats])
    pooled_sq = np.sum(pooled**2, axis=1, keepdims=True)
    pair_sq_dist = pooled_sq + pooled_sq.T - 2 * np.dot(pooled, pooled.T)
    pair_sq_dist = np.maximum(pair_sq_dist, 0.0)

    # Median heuristic for base bandwidth
    non_diag = pair_sq_dist[np.triu_indices_from(pair_sq_dist, k=1)]
    median_dist_sq = float(np.median(non_diag)) if len(non_diag) > 0 else 1.0
    base_gamma = 1.0 / max(median_dist_sq, 1e-6)

    # Multi-kernel bandwidth multipliers (0.1, 0.5, 1.0, 2.0, 5.0)
    multipliers = [0.1, 0.5, 1.0, 2.0, 5.0]
    gammas = [base_gamma / m for m in multipliers]

    def compute_unbiased_mmd2(k_matrix: np.ndarray, nx: int, ny: int) -> float:
        k_xx = k_matrix[:nx, :nx]
        k_yy = k_matrix[nx:, nx:]
        k_xy = k_matrix[:nx, nx:]

        # Unbiased estimator: exclude diagonals from within-distribution terms
        sum_xx = np.sum(k_xx) - np.trace(k_xx)
        sum_yy = np.sum(k_yy) - np.trace(k_yy)
        sum_xy = np.sum(k_xy)

        term_xx = sum_xx / max(nx * (nx - 1), 1)
        term_yy = sum_yy / max(ny * (ny - 1), 1)
        term_xy = 2.0 * sum_xy / max(nx * ny, 1)

        return float(term_xx + term_yy - term_xy)

    # Construct composite multi-kernel matrix
    comp_k = np.zeros_like(pair_sq_dist)
    for g in gammas:
        comp_k += np.exp(-g * pair_sq_dist)
    comp_k /= len(gammas)

    obs_mmd2 = compute_unbiased_mmd2(comp_k, n_x, n_y)
    obs_mmd = float(np.sqrt(max(obs_mmd2, 0.0)))

    # Permutation test for empirical p-value
    perm_count = 0
    if n_permutations > 0:
        total_n = n_x + n_y
        for _ in range(n_permutations):
            perm_idx = rng.permutation(total_n)
            perm_k = comp_k[perm_idx][:, perm_idx]
            perm_mmd2 = compute_unbiased_mmd2(perm_k, n_x, n_y)
            if perm_mmd2 >= obs_mmd2:
                perm_count += 1
        p_value = float((perm_count + 1) / (n_permutations + 1))
    else:
        p_value = float("nan")

    return {
        "mmd": obs_mmd,
        "mmd_squared": obs_mmd2,
        "p_value": p_value,
        "centroid_euclidean_distance": centroid_euclidean,
        "centroid_cosine_similarity": centroid_cos_sim,
        "centroid_cosine_distance": centroid_cos_dist,
        "bandwidth_heuristic": median_dist_sq,
        "n_source_samples": n_x,
        "n_target_samples": n_y,
    }


def prepare_subgroup_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes view position, sex, and age columns into clean categorical subgroup features.

    Recognizes standard column variations across NIH ChestX-ray14 and Stanford CheXpert.
    """
    res = df.copy()

    # 1. Standardize View Position
    view_col = None
    for cand in ["view_position", "View Position", "AP/PA", "projection", "Frontal/Lateral"]:
        if cand in res.columns:
            view_col = cand
            break

    if view_col is not None:
        res["subgroup_view"] = (
            res[view_col].fillna("Unknown").astype(str).str.upper().str.strip()
        )
        # Normalize AP/PA values
        res["subgroup_view"] = res["subgroup_view"].replace({
            "PA": "PA",
            "AP": "AP",
            "LL": "LATERAL",
            "LATERAL": "LATERAL",
        })
    else:
        res["subgroup_view"] = "Unknown"

    # 2. Standardize Sex
    sex_col = None
    for cand in ["sex", "Sex", "Patient Gender", "gender"]:
        if cand in res.columns:
            sex_col = cand
            break

    if sex_col is not None:
        res["subgroup_sex"] = (
            res[sex_col].fillna("Unknown").astype(str).str.upper().str.strip()
        )
        res["subgroup_sex"] = res["subgroup_sex"].replace({"M": "M", "F": "F"})
    else:
        res["subgroup_sex"] = "Unknown"

    # 3. Standardize Age and Age Brackets (<40, 40-60, >60)
    age_col = None
    for cand in ["age", "Age", "Patient Age"]:
        if cand in res.columns:
            age_col = cand
            break

    if age_col is not None:
        res["subgroup_age_num"] = pd.to_numeric(res[age_col], errors="coerce").fillna(-1.0)
    else:
        res["subgroup_age_num"] = -1.0

    def categorize_age(age_val: float) -> str:
        if age_val < 0.0 or np.isnan(age_val):
            return "Unknown"
        if age_val < 40.0:
            return "<40"
        if age_val <= 60.0:
            return "40-60"
        return ">60"

    res["subgroup_age"] = res["subgroup_age_num"].apply(categorize_age)
    return res


def evaluate_subgroups(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_col: str,
    findings: Optional[List[str]] = None,
    min_samples: int = 5,
) -> Dict[str, Any]:
    """Evaluates discrimination metrics across demographic or procedural subgroups.

    Supports both 1D single-class and 2D multi-label predictions.

    Args:
        df: Metadata DataFrame containing the subgroup column.
        y_true: Binary ground truth labels (N,) or (N, C).
        y_pred: Predicted probabilities (N,) or (N, C).
        subgroup_col: Column name in df defining the subgroup partitions.
        findings: Optional list of class names if y_true is 2D.
        min_samples: Minimum samples required to evaluate a subgroup.

    Returns:
        Dict mapping subgroup value to discrimination metrics (or per-class metrics).
    """
    subgroup_results: Dict[str, Any] = {}

    if subgroup_col not in df.columns:
        return subgroup_results

    groups = sorted([str(g) for g in df[subgroup_col].dropna().unique() if str(g) != "Unknown"])
    is_multilabel = y_true.ndim > 1 and y_true.shape[1] > 1
    classes = findings or (SHARED_CLASSES if is_multilabel else ["Finding"])

    for g in groups:
        mask = (df[subgroup_col].astype(str) == g).values
        n_group = int(np.sum(mask))
        if n_group < min_samples:
            continue

        g_yt = y_true[mask]
        g_yp = y_pred[mask]

        if not is_multilabel:
            # 1D single-label
            yt_flat = g_yt.flatten()
            yp_flat = g_yp.flatten()
            subgroup_results[g] = compute_discrimination_metrics(yt_flat, yp_flat)
            subgroup_results[g]["n_samples"] = n_group
        else:
            # 2D multi-label
            group_class_metrics: Dict[str, Any] = {}
            aurocs: List[float] = []
            auprcs: List[float] = []

            for idx, c_name in enumerate(classes):
                c_yt = g_yt[:, idx]
                c_yp = g_yp[:, idx]
                c_m = compute_discrimination_metrics(c_yt, c_yp)
                c_m["n_samples"] = n_group
                group_class_metrics[c_name] = c_m

                if not np.isnan(c_m["auroc"]):
                    aurocs.append(c_m["auroc"])
                if not np.isnan(c_m["auprc"]):
                    auprcs.append(c_m["auprc"])

            group_class_metrics["macro_average"] = {
                "auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
                "auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
                "n_samples": n_group,
            }
            subgroup_results[g] = group_class_metrics

    return subgroup_results


def evaluate_subgroup_disparities(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    findings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Decomposes target cohort performance across View Position, Sex, and Age brackets.

    Calculates explicit subgroup disparity gaps:
      - View Disparity: AUROC(PA) - AUROC(AP)
      - Sex Disparity: AUROC(M) - AUROC(F)
      - Age Disparity: max(AUROC) - min(AUROC) across [<40, 40-60, >60]

    Args:
        df: Metadata DataFrame with patient demographic/procedural fields.
        y_true: Ground truth binary labels (N, C).
        y_pred: Predicted probabilities (N, C).
        findings: List of pathology finding names (default: SHARED_CLASSES).

    Returns:
        Structured dict with view, sex, and age breakdowns and disparity deltas.
    """
    classes = findings or SHARED_CLASSES
    std_df = prepare_subgroup_metadata(df)

    # 1. View Position Breakdown (AP vs PA)
    view_results = evaluate_subgroups(std_df, y_true, y_pred, "subgroup_view", findings=classes)

    # 2. Sex Breakdown (M vs F)
    sex_results = evaluate_subgroups(std_df, y_true, y_pred, "subgroup_sex", findings=classes)

    # 3. Age Breakdown (<40, 40-60, >60)
    age_results = evaluate_subgroups(std_df, y_true, y_pred, "subgroup_age", findings=classes)

    def extract_auc(results_dict: Dict[str, Any], group_key: str, finding: str) -> float:
        grp = results_dict.get(group_key, {})
        if finding in grp:
            return float(grp[finding].get("auroc", float("nan")))
        return float("nan")

    per_class_disparities: Dict[str, Any] = {}
    macro_view_pa: List[float] = []
    macro_view_ap: List[float] = []
    macro_sex_m: List[float] = []
    macro_sex_f: List[float] = []
    macro_age_young: List[float] = []
    macro_age_mid: List[float] = []
    macro_age_old: List[float] = []

    for f in classes:
        pa_auc = extract_auc(view_results, "PA", f)
        ap_auc = extract_auc(view_results, "AP", f)
        m_auc = extract_auc(sex_results, "M", f)
        f_auc = extract_auc(sex_results, "F", f)
        y_auc = extract_auc(age_results, "<40", f)
        mid_auc = extract_auc(age_results, "40-60", f)
        old_auc = extract_auc(age_results, ">60", f)

        # Disparity gaps
        delta_view = (pa_auc - ap_auc) if (not np.isnan(pa_auc) and not np.isnan(ap_auc)) else float("nan")
        delta_sex = (m_auc - f_auc) if (not np.isnan(m_auc) and not np.isnan(f_auc)) else float("nan")

        valid_age_aucs = [a for a in [y_auc, mid_auc, old_auc] if not np.isnan(a)]
        max_age_disparity = (max(valid_age_aucs) - min(valid_age_aucs)) if len(valid_age_aucs) > 1 else float("nan")

        if not np.isnan(pa_auc):
            macro_view_pa.append(pa_auc)
        if not np.isnan(ap_auc):
            macro_view_ap.append(ap_auc)
        if not np.isnan(m_auc):
            macro_sex_m.append(m_auc)
        if not np.isnan(f_auc):
            macro_sex_f.append(f_auc)
        if not np.isnan(y_auc):
            macro_age_young.append(y_auc)
        if not np.isnan(mid_auc):
            macro_age_mid.append(mid_auc)
        if not np.isnan(old_auc):
            macro_age_old.append(old_auc)

        per_class_disparities[f] = {
            "view": {"PA_auroc": pa_auc, "AP_auroc": ap_auc, "delta_view_pa_minus_ap": delta_view},
            "sex": {"M_auroc": m_auc, "F_auroc": f_auc, "delta_sex_m_minus_f": delta_sex},
            "age": {
                "<40_auroc": y_auc,
                "40-60_auroc": mid_auc,
                ">60_auroc": old_auc,
                "max_age_disparity": max_age_disparity,
            },
        }

    macro_summary = {
        "view": {
            "mean_PA_auroc": float(np.mean(macro_view_pa)) if macro_view_pa else float("nan"),
            "mean_AP_auroc": float(np.mean(macro_view_ap)) if macro_view_ap else float("nan"),
            "mean_delta_view": (
                float(np.mean(macro_view_pa) - np.mean(macro_view_ap))
                if (macro_view_pa and macro_view_ap)
                else float("nan")
            ),
        },
        "sex": {
            "mean_M_auroc": float(np.mean(macro_sex_m)) if macro_sex_m else float("nan"),
            "mean_F_auroc": float(np.mean(macro_sex_f)) if macro_sex_f else float("nan"),
            "mean_delta_sex": (
                float(np.mean(macro_sex_m) - np.mean(macro_sex_f))
                if (macro_sex_m and macro_sex_f)
                else float("nan")
            ),
        },
        "age": {
            "mean_young_auroc": float(np.mean(macro_age_young)) if macro_age_young else float("nan"),
            "mean_mid_auroc": float(np.mean(macro_age_mid)) if macro_age_mid else float("nan"),
            "mean_old_auroc": float(np.mean(macro_age_old)) if macro_age_old else float("nan"),
        },
    }

    return {
        "raw_subgroups": {
            "view": view_results,
            "sex": sex_results,
            "age": age_results,
        },
        "per_class_disparities": per_class_disparities,
        "headline_pneumonia": per_class_disparities.get(HEADLINE_CLASS, {}),
        "macro_summary": macro_summary,
    }


def evaluate_domain_gap_suite(
    source_features: np.ndarray,
    target_features: np.ndarray,
    target_metadata: pd.DataFrame,
    target_labels: np.ndarray,
    target_probs: np.ndarray,
    findings: Optional[List[str]] = None,
    shift_direction: str = "nih_to_chexpert",
    seed: int = 42,
) -> Dict[str, Any]:
    """Orchestrates comprehensive feature space probing and subgroup confounder analysis.

    Executes:
      1. Linear Domain Classifier Probe (Logistic Regression 5-Fold CV on frozen penultimate embeddings).
      2. Multi-kernel RBF MMD divergence with permutation test and centroid distance.
      3. Target Cohort Subgroup Confounder Analysis (AP vs PA view position, M vs F sex, and age brackets).
      4. Generates publication-ready Markdown tables.

    Args:
        source_features: Penultimate layer embeddings from source cohort (N_src, 1024).
        target_features: Penultimate layer embeddings from target cohort (N_tgt, 1024).
        target_metadata: Metadata DataFrame for target evaluation cohort.
        target_labels: Binary ground truth labels for target cohort (N_tgt, C).
        target_probs: Model prediction probabilities for target cohort (N_tgt, C).
        findings: List of pathology finding names (default: SHARED_CLASSES).
        shift_direction: Direction string ('nih_to_chexpert' or 'chexpert_to_nih').
        seed: Random seed for reproducibility.

    Returns:
        Structured evaluation dict with probe metrics, MMD, subgroup disparities, and Markdown tables.
    """
    classes = findings or SHARED_CLASSES

    # 1. Linear Domain Classifier Probe
    logger.info("Training linear domain classifier probe (%s)...", shift_direction)
    probe_results = train_domain_classifier_probe(
        source_features=source_features,
        target_features=target_features,
        n_splits=5,
        seed=seed,
    )

    # 2. Penultimate Feature Space Divergence (Multi-Kernel MMD & Centroid Distance)
    logger.info("Computing multi-kernel MMD feature divergence (%s)...", shift_direction)
    feature_div = compute_feature_space_divergence(
        source_features=source_features,
        target_features=target_features,
        max_samples=1000,
        n_permutations=100,
        seed=seed,
    )

    # 3. Subgroup Confounder Disparities
    logger.info("Analyzing demographic & procedural subgroup disparities (%s)...", shift_direction)
    subgroups = evaluate_subgroup_disparities(
        df=target_metadata,
        y_true=target_labels,
        y_pred=target_probs,
        findings=classes,
    )

    # 4. Generate Publication Markdown Tables
    probe_auc = probe_results.get("probe_auroc", float("nan"))
    probe_bacc = probe_results.get("probe_balanced_accuracy", float("nan"))
    probe_interp = (
        "Severe feature distribution shift (separable hospital signatures)"
        if probe_auc > 0.90
        else "Moderate feature distribution overlap"
    )

    table1_rows = [
        f"### Penultimate Feature Space Domain Gap & Linear Separability ({shift_direction.upper()})",
        "",
        "| Metric | Value | Interpretation / Significance |",
        "| :--- | :---: | :--- |",
        f"| **Domain Probe AUROC (5-Fold CV)** | {probe_auc:.4f} (±{probe_results.get('fold_auroc_std', 0.0):.4f}) | {probe_interp} |",
        f"| **Domain Probe Balanced Accuracy** | {probe_bacc:.4f} | Out-of-fold linear classifier accuracy |",
        f"| **Multi-Kernel RBF MMD** | {feature_div['mmd']:.4f} | Maximum Mean Discrepancy (5 RBF bandwidths) |",
        f"| **MMD Permutation Test p-value** | {feature_div['p_value']:.4e} | Null hypothesis: P(Source) == P(Target) |",
        f"| **Centroid Euclidean Distance** | {feature_div['centroid_euclidean_distance']:.4f} | Distance between class-agnostic feature means |",
        f"| **Centroid Cosine Similarity** | {feature_div['centroid_cosine_similarity']:.4f} | Angle between mean representation vectors |",
    ]

    table2_rows = [
        f"### Subgroup Confounder Disparities on Target Domain ({shift_direction.upper()})",
        "",
        "| Finding | View PA AUROC | View AP AUROC | Δ View (PA - AP) | Sex M AUROC | Sex F AUROC | Δ Sex (M - F) | Age <40 | Age 40-60 | Age >60 | Max Δ Age |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    p_disp = subgroups["per_class_disparities"]
    for f in classes:
        f_data = p_disp.get(f, {})
        v = f_data.get("view", {})
        s = f_data.get("sex", {})
        a = f_data.get("age", {})

        def fmt(val: float) -> str:
            return f"{val:.3f}" if not np.isnan(val) else "—"

        def fmt_d(val: float) -> str:
            return f"{val:+.3f}" if not np.isnan(val) else "—"

        table2_rows.append(
            f"| **{f}** | {fmt(v.get('PA_auroc', float('nan')))} | {fmt(v.get('AP_auroc', float('nan')))} | "
            f"{fmt_d(v.get('delta_view_pa_minus_ap', float('nan')))} | "
            f"{fmt(s.get('M_auroc', float('nan')))} | {fmt(s.get('F_auroc', float('nan')))} | "
            f"{fmt_d(s.get('delta_sex_m_minus_f', float('nan')))} | "
            f"{fmt(a.get('<40_auroc', float('nan')))} | {fmt(a.get('40-60_auroc', float('nan')))} | "
            f"{fmt(a.get('>60_auroc', float('nan')))} | {fmt(a.get('max_age_disparity', float('nan')))} |"
        )

    # Macro averages row
    m_sum = subgroups["macro_summary"]
    mv = m_sum["view"]
    ms = m_sum["sex"]
    ma = m_sum["age"]

    def fmt_m(val: float) -> str:
        return f"{val:.3f}" if not np.isnan(val) else "—"

    def fmt_md(val: float) -> str:
        return f"{val:+.3f}" if not np.isnan(val) else "—"

    table2_rows.append(
        f"| **Macro Average** | {fmt_m(mv.get('mean_PA_auroc', float('nan')))} | "
        f"{fmt_m(mv.get('mean_AP_auroc', float('nan')))} | "
        f"{fmt_md(mv.get('mean_delta_view', float('nan')))} | "
        f"{fmt_m(ms.get('mean_M_auroc', float('nan')))} | "
        f"{fmt_m(ms.get('mean_F_auroc', float('nan')))} | "
        f"{fmt_md(ms.get('mean_delta_sex', float('nan')))} | "
        f"{fmt_m(ma.get('mean_young_auroc', float('nan')))} | "
        f"{fmt_m(ma.get('mean_mid_auroc', float('nan')))} | "
        f"{fmt_m(ma.get('mean_old_auroc', float('nan')))} | — |"
    )

    return {
        "shift_direction": shift_direction,
        "domain_classifier_probe": probe_results,
        "feature_space_divergence": feature_div,
        "subgroup_disparities": subgroups,
        "headline_pneumonia": subgroups["headline_pneumonia"],
        "markdown_tables": {
            "feature_space_domain_gap": "\n".join(table1_rows),
            "subgroup_confounder_disparities": "\n".join(table2_rows),
        },
    }
