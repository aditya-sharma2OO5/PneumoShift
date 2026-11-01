"""Domain gap quantification: dataset probe, embedding MMD, and subgroup analyses."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pneumoshift.evaluation.metrics import compute_discrimination_metrics


def train_domain_classifier_probe(
    source_features: np.ndarray,
    target_features: np.ndarray,
    seed: int = 42,
) -> Dict[str, float]:
    """Trains a logistic regression probe on frozen penultimate features to predict source vs target dataset.

    An AUROC close to 1.0 indicates substantial domain separability / severe feature shift.
    """
    n_source = len(source_features)
    n_target = len(target_features)

    X = np.vstack([source_features, target_features])
    y = np.concatenate([np.zeros(n_source), np.ones(n_target)])

    # Shuffle
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    probe_auroc = float(roc_auc_score(y_test, probs))

    return {
        "domain_probe_auroc": probe_auroc,
        "n_samples": len(X),
    }


def compute_feature_mmd(
    features_x: np.ndarray,
    features_y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """Computes Maximum Mean Discrepancy (MMD) with RBF kernel between two feature distributions."""
    if len(features_x) == 0 or len(features_y) == 0:
        return 0.0

    if gamma is None:
        gamma = 1.0 / features_x.shape[1]

    # Subsample if necessary to prevent OOM
    max_samples = 1000
    if len(features_x) > max_samples:
        features_x = features_x[:max_samples]
    if len(features_y) > max_samples:
        features_y = features_y[:max_samples]

    def rbf_kernel(a, b):
        sq_dist = np.sum(a**2, axis=1, keepdims=True) + np.sum(b**2, axis=1, keepdims=True).T - 2 * np.dot(a, b.T)
        return np.exp(-gamma * np.maximum(sq_dist, 0))

    k_xx = rbf_kernel(features_x, features_x)
    k_yy = rbf_kernel(features_y, features_y)
    k_xy = rbf_kernel(features_x, features_y)

    mmd_sq = np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)
    return float(np.sqrt(max(mmd_sq, 0.0)))


def evaluate_subgroups(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_col: str,
) -> Dict[str, Dict[str, float]]:
    """Evaluates performance across demographic or procedural subgroups (e.g., AP vs PA, Sex, Age bins)."""
    subgroup_results: Dict[str, Dict[str, float]] = {}

    if subgroup_col not in df.columns:
        return subgroup_results

    groups = df[subgroup_col].dropna().unique()
    for g in groups:
        mask = (df[subgroup_col] == g).values
        if np.sum(mask) < 5:
            continue
        g_yt = y_true[mask]
        g_yp = y_pred[mask]
        subgroup_results[str(g)] = compute_discrimination_metrics(g_yt, g_yp)

    return subgroup_results

