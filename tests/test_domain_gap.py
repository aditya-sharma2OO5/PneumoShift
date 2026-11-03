"""Tests for penultimate feature space domain probing, multi-kernel MMD, and subgroup confounder analyses."""

import numpy as np
import pandas as pd
import pytest

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.domain_gap import (
    compute_feature_mmd,
    compute_feature_space_divergence,
    evaluate_domain_gap_suite,
    evaluate_subgroup_disparities,
    evaluate_subgroups,
    prepare_subgroup_metadata,
    train_domain_classifier_probe,
)


def test_train_domain_classifier_probe_well_separated():
    np.random.seed(42)
    n_source = 100
    n_target = 100
    dim = 64

    # Clearly separated domains (e.g. hospital-specific feature shift)
    source_features = np.random.normal(loc=1.5, scale=0.5, size=(n_source, dim))
    target_features = np.random.normal(loc=-1.5, scale=0.5, size=(n_target, dim))

    res = train_domain_classifier_probe(source_features, target_features, n_splits=5, seed=42)

    assert "probe_auroc" in res
    assert "domain_probe_auroc" in res  # Backwards compatibility
    assert "probe_accuracy" in res
    assert "probe_balanced_accuracy" in res
    assert "fold_aurocs" in res
    assert len(res["fold_aurocs"]) == 5
    assert res["probe_auroc"] >= 0.95
    assert res["probe_balanced_accuracy"] >= 0.90
    assert res["n_source"] == n_source
    assert res["n_target"] == n_target


def test_train_domain_classifier_probe_identical_distributions():
    np.random.seed(42)
    n = 100
    dim = 32

    # Identical feature distributions
    source_features = np.random.normal(loc=0.0, scale=1.0, size=(n, dim))
    target_features = np.random.normal(loc=0.0, scale=1.0, size=(n, dim))

    res = train_domain_classifier_probe(source_features, target_features, n_splits=5, seed=42)
    # When distributions are identical, linear probe cannot separate them: AUROC ~ 0.50
    assert 0.35 <= res["probe_auroc"] <= 0.65


def test_train_domain_classifier_probe_small_sample_fallback():
    source_features = np.random.randn(3, 16)
    target_features = np.random.randn(3, 16)

    res = train_domain_classifier_probe(source_features, target_features, n_splits=5, seed=42)
    assert res["n_samples"] == 6
    assert not np.isnan(res["probe_auroc"])


def test_compute_feature_mmd_backwards_compatible():
    np.random.seed(42)
    fx = np.random.normal(loc=0.0, scale=1.0, size=(50, 32))
    fy = np.random.normal(loc=2.0, scale=1.0, size=(50, 32))

    mmd_val = compute_feature_mmd(fx, fy)
    assert isinstance(mmd_val, float)
    assert mmd_val > 0.0

    # Identical features should yield near-zero MMD
    mmd_ident = compute_feature_mmd(fx, fx)
    assert mmd_ident < 1e-4


def test_compute_feature_space_divergence_multi_kernel():
    np.random.seed(42)
    fx = np.random.normal(loc=0.0, scale=1.0, size=(80, 32))
    fy = np.random.normal(loc=1.5, scale=1.0, size=(80, 32))

    div = compute_feature_space_divergence(
        source_features=fx,
        target_features=fy,
        max_samples=500,
        n_permutations=50,
        seed=42,
    )

    assert "mmd" in div
    assert "mmd_squared" in div
    assert "p_value" in div
    assert "centroid_euclidean_distance" in div
    assert "centroid_cosine_similarity" in div
    assert div["mmd"] > 0.0
    assert div["centroid_euclidean_distance"] > 0.0
    assert div["p_value"] < 0.05  # Strongly rejects null hypothesis of equal distributions


def test_prepare_subgroup_metadata():
    raw_df = pd.DataFrame({
        "AP/PA": ["PA", "ap", "LL", None],
        "Patient Gender": ["M", "F", "f", None],
        "Patient Age": [25, 45, 75, "invalid"],
    })

    std_df = prepare_subgroup_metadata(raw_df)
    assert "subgroup_view" in std_df.columns
    assert "subgroup_sex" in std_df.columns
    assert "subgroup_age" in std_df.columns

    assert std_df["subgroup_view"].tolist() == ["PA", "AP", "LATERAL", "UNKNOWN"]
    assert std_df["subgroup_sex"].tolist() == ["M", "F", "F", "UNKNOWN"]
    assert std_df["subgroup_age"].tolist() == ["<40", "40-60", ">60", "Unknown"]


def test_evaluate_subgroups_1d_and_2d():
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "view": ["PA"] * 60 + ["AP"] * 60,
    })

    # 1D single-class evaluation
    y_true_1d = (np.random.rand(n) > 0.7).astype(np.float32)
    y_pred_1d = np.random.uniform(0.1, 0.9, n)

    res_1d = evaluate_subgroups(df, y_true_1d, y_pred_1d, subgroup_col="view")
    assert "PA" in res_1d
    assert "AP" in res_1d
    assert "auroc" in res_1d["PA"]
    assert res_1d["PA"]["n_samples"] == 60

    # 2D multi-label evaluation
    classes = ["Pneumonia", "Effusion"]
    y_true_2d = (np.random.rand(n, len(classes)) > 0.7).astype(np.float32)
    y_pred_2d = np.random.uniform(0.1, 0.9, (n, len(classes)))

    res_2d = evaluate_subgroups(df, y_true_2d, y_pred_2d, subgroup_col="view", findings=classes)
    assert "PA" in res_2d
    assert "Pneumonia" in res_2d["PA"]
    assert "macro_average" in res_2d["PA"]
    assert "auroc" in res_2d["PA"]["macro_average"]


def test_evaluate_subgroup_disparities():
    np.random.seed(42)
    n = 150
    classes = ["Pneumonia", "Atelectasis"]

    df = pd.DataFrame({
        "view_position": ["PA"] * 75 + ["AP"] * 75,
        "sex": ["M"] * 75 + ["F"] * 75,
        "age": [30] * 50 + [50] * 50 + [70] * 50,
    })

    # Model performs systematically better on PA views than AP bedside views
    y_true = (np.random.rand(n, len(classes)) > 0.8).astype(np.float32)
    y_pred = np.zeros((n, len(classes)))

    # Perfect separation on PA (first 75 samples)
    y_pred[:75] = np.where(y_true[:75] == 1, 0.85, 0.15)
    # Degraded/noisy predictions on AP bedside views (last 75 samples)
    y_pred[75:] = np.random.uniform(0.2, 0.8, size=(75, len(classes)))

    disparities = evaluate_subgroup_disparities(df, y_true, y_pred, findings=classes)

    assert "per_class_disparities" in disparities
    assert "headline_pneumonia" in disparities
    assert "macro_summary" in disparities

    pneu_view = disparities["per_class_disparities"]["Pneumonia"]["view"]
    assert pneu_view["PA_auroc"] > pneu_view["AP_auroc"]
    assert pneu_view["delta_view_pa_minus_ap"] > 0.0  # PA better than AP


def test_evaluate_domain_gap_suite_full():
    np.random.seed(42)
    n_src = 80
    n_tgt = 120
    dim = 64
    classes = ["Pneumonia", "Effusion", "Infiltration"]

    source_feats = np.random.normal(loc=1.0, scale=0.5, size=(n_src, dim))
    target_feats = np.random.normal(loc=-1.0, scale=0.5, size=(n_tgt, dim))

    target_df = pd.DataFrame({
        "view_position": ["PA"] * 60 + ["AP"] * 60,
        "sex": ["M"] * 60 + ["F"] * 60,
        "age": [35] * 40 + [52] * 40 + [68] * 40,
    })

    target_labels = (np.random.rand(n_tgt, len(classes)) > 0.8).astype(np.float32)
    target_probs = np.random.uniform(0.1, 0.9, (n_tgt, len(classes)))

    suite = evaluate_domain_gap_suite(
        source_features=source_feats,
        target_features=target_feats,
        target_metadata=target_df,
        target_labels=target_labels,
        target_probs=target_probs,
        findings=classes,
        shift_direction="nih_to_chexpert",
        seed=42,
    )

    assert suite["shift_direction"] == "nih_to_chexpert"
    assert "domain_classifier_probe" in suite
    assert "feature_space_divergence" in suite
    assert "subgroup_disparities" in suite
    assert "headline_pneumonia" in suite
    assert "markdown_tables" in suite

    # Check Markdown tables
    tables = suite["markdown_tables"]
    assert "feature_space_domain_gap" in tables
    assert "subgroup_confounder_disparities" in tables
    assert "Penultimate Feature Space Domain Gap & Linear Separability" in tables["feature_space_domain_gap"]
    assert "Subgroup Confounder Disparities on Target Domain" in tables["subgroup_confounder_disparities"]
    assert "**Pneumonia**" in tables["subgroup_confounder_disparities"]
    assert "**Macro Average**" in tables["subgroup_confounder_disparities"]

