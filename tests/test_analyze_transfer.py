"""Comprehensive unit & integration tests for transfer evaluation (ANALYSIS_PLAN_TRANSFER.md v2.2.3)."""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import scipy.special
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pneumoshift.analysis.bootstrap import (
    ClusterBootstrapPlan,
    benjamini_hochberg,
    compute_paired_architecture_bootstrap,
    two_sided_bootstrap_pvalue,
)
from pneumoshift.analysis.metrics import (
    apply_label_policy,
    compute_calibration_suite,
    compute_ece_adaptive,
    compute_ece_uniform,
    compute_operating_metrics,
    delong_two_sample,
    fast_auroc,
    fit_binomial_glm_logits,
    select_thresholds,
    stable_nll_from_logits,
    target_10pct_platt,
)
from pneumoshift.analysis.pipeline import run_transfer_analysis
from pneumoshift.analysis.provenance import load_real_run_bundle, sha256_file


def test_clustered_bootstrap_coverage_on_simulated_data():
    """Verifies ~95% empirical coverage for patient-clustered CI under strong intracluster correlation."""
    rng = np.random.default_rng(2026)
    n_clusters = 80
    cluster_size = 5
    covered = 0
    n_trials = 150

    pids = np.repeat(np.arange(n_clusters), cluster_size)
    _, ga = np.unique(pids, return_inverse=True)

    # Strong intra-cluster correlation: patient-level latent binary state + 5% within-patient flip
    # P(Y_A=1) = 0.50*0.95 + 0.50*0.05 = 0.50; P(Y_B=1) = 0.20*0.95 + 0.80*0.05 = 0.23 -> diff = 0.27
    true_diff = 0.27
    for trial in range(n_trials):
        y_a = np.repeat(rng.binomial(1, 0.50, size=n_clusters), cluster_size).astype(np.float64)
        flip_a = rng.random(len(y_a)) < 0.05
        y_a[flip_a] = 1.0 - y_a[flip_a]
        y_b = np.repeat(rng.binomial(1, 0.20, size=n_clusters), cluster_size).astype(np.float64)
        flip_b = rng.random(len(y_b)) < 0.05
        y_b[flip_b] = 1.0 - y_b[flip_b]

        plan = ClusterBootstrapPlan(pids, pids, n_resamples=300, seed=1000 + trial)
        diffs = []
        for cwa, cwb in plan.iter_cluster_weights():
            wa = cwa[ga].astype(np.float64)
            wb = cwb[ga].astype(np.float64)
            diffs.append(np.dot(wa, y_a) / wa.sum() - np.dot(wb, y_b) / wb.sum())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        if lo <= true_diff <= hi:
            covered += 1

    cov_rate = covered / n_trials
    assert 0.89 <= cov_rate <= 0.99, f"Unexpected clustered bootstrap coverage: {cov_rate:.3f}"


def test_two_sided_bootstrap_pvalue_formula_edge_cases():
    """Verifies §5.4 pseudocount continuity-corrected two-sided bootstrap p-value formula."""
    b = 2000
    # 1. All replicates strictly positive -> #(<=0)=0, #(>=0)=2000 -> p = 2 * (1 / 2001)
    all_pos = np.ones(b, dtype=np.float64)
    p_pos = two_sided_bootstrap_pvalue(all_pos)
    assert p_pos == pytest.approx(2.0 / (b + 1.0), rel=1e-12)

    # 2. All replicates strictly negative -> symmetric
    all_neg = -np.ones(b, dtype=np.float64)
    p_neg = two_sided_bootstrap_pvalue(all_neg)
    assert p_neg == pytest.approx(2.0 / (b + 1.0), rel=1e-12)

    # 3. All replicates exact zero -> #(<=0)=2000, #(>=0)=2000 -> capped at 1.0
    all_zero = np.zeros(b, dtype=np.float64)
    assert two_sided_bootstrap_pvalue(all_zero) == 1.0

    # 4. Exact 50/50 split with zero overlap -> capped at 1.0 (not > 1.0)
    half = np.concatenate([np.ones(1000), -np.ones(1000)])
    assert two_sided_bootstrap_pvalue(half) == 1.0

    # 5. Exact non-trivial proportion: 50 negatives and 1950 positives -> p = 2 * (51 / 2001)
    skewed = np.concatenate([-np.ones(50), np.ones(1950)])
    assert two_sided_bootstrap_pvalue(skewed) == pytest.approx(2.0 * 51.0 / 2001.0, rel=1e-12)


def test_benjamini_hochberg_matches_scipy_and_manual():
    """Verifies BH step-up adjusted q-values, monotonicity clamping, and significance flags."""
    p_vals = [0.001, 0.04, 0.039, 0.25, 0.015, 0.80]
    q_vals, sig = benjamini_hochberg(p_vals, alpha=0.05)

    # Compare against scipy.stats.false_discovery_control
    expected_q = scipy.stats.false_discovery_control(p_vals, method="bh")
    np.testing.assert_allclose(q_vals, expected_q, atol=1e-12)

    # Manual check: sorted p = [0.001 (r1), 0.015 (r2), 0.039 (r3), 0.04 (r4), 0.25 (r5), 0.80 (r6)]
    # raw = [0.006, 0.045, 0.078, 0.060, 0.30, 0.80] -> cummin from right sets r3 from 0.078 to 0.060!
    assert q_vals[2] == pytest.approx(0.060, rel=1e-12)
    assert q_vals[1] == pytest.approx(0.060, rel=1e-12)
    assert sig == [True, False, False, False, True, False]


def test_three_seed_mean_pairing_and_between_seed_sd():
    """Verifies paired identical resamples across seeds reduce variance and between-seed SD uses ddof=1."""
    rng = np.random.default_rng(7)
    n_pat_a, n_pat_b = 50, 60
    pids_a = np.repeat(np.arange(n_pat_a), 4)
    pids_b = np.repeat(np.arange(n_pat_b), 4)
    y_a = rng.binomial(1, 0.2, size=len(pids_a)).astype(np.float64)
    y_b = rng.binomial(1, 0.2, size=len(pids_b)).astype(np.float64)

    # Three seeds with deterministic separation so seed offsets are clear
    base_a = 1.5 * y_a - 0.5 + rng.normal(0, 0.4, size=len(pids_a))
    base_b = 1.0 * y_b - 0.5 + rng.normal(0, 0.4, size=len(pids_b))

    seed_inputs = {}
    for s_idx, seed in enumerate((42, 1337, 2024)):
        za = base_a + 0.05 * s_idx * rng.normal(size=len(pids_a))
        zb = base_b + 0.05 * s_idx * rng.normal(size=len(pids_b))
        th = select_thresholds(y_a, za)
        seed_inputs[seed] = {"logits_a": za, "logits_b": zb, "thresholds": th}

    plan = ClusterBootstrapPlan(pids_a, pids_b, n_resamples=100, seed=42)
    out = compute_paired_architecture_bootstrap(plan, y_a, y_b, seed_inputs)

    # Verify 3-seed mean & ddof=1 SD match exact per-seed point estimates
    per_seed_d_auc = np.array([out["per_seed"][s]["contrast"]["auroc"]["estimate"] for s in (42, 1337, 2024)])
    joint_auc = out["joint"]["contrast"]["auroc"]
    assert joint_auc["estimate"] == pytest.approx(float(np.mean(per_seed_d_auc)), rel=1e-12)
    assert joint_auc["between_seed_sd_ddof1"] == pytest.approx(float(np.std(per_seed_d_auc, ddof=1)), rel=1e-12)

    # Verify pairing invariant: identical constant signal across seeds gives exact 0 variance across seeds
    # (confirming all 3 seeds consumed identical cluster weights on every iteration b)
    same_inputs = {
        seed: {"logits_a": base_a, "logits_b": base_b, "thresholds": select_thresholds(y_a, base_a)}
        for seed in (42, 1337, 2024)
    }
    out_same = compute_paired_architecture_bootstrap(plan, y_a, y_b, same_inputs)
    assert out_same["joint"]["contrast"]["auroc"]["ci_lower"] == pytest.approx(out_same["per_seed"][42]["contrast"]["auroc"]["ci_lower"], rel=1e-12)
    assert out_same["joint"]["contrast"]["auroc"]["ci_upper"] == pytest.approx(out_same["per_seed"][42]["contrast"]["auroc"]["ci_upper"], rel=1e-12)
    assert out_same["joint"]["contrast"]["auroc"]["between_seed_sd_ddof1"] == 0.0


def test_threshold_selection_highest_threshold_tie_break_and_sens_rules():
    """Verifies §9.1-9.3: Youden tie selects highest threshold, Sens>=0.90/0.95 selects largest tau."""
    # Construct deterministic fixture with exact tie in Youden's J across two cutoffs:
    # Positives (P=10) at probs: 8 @ 0.90, 1 @ 0.50, 1 @ 0.20
    # Negatives (N=10) at probs: 1 @ 0.95, 1 @ 0.60, 8 @ 0.30
    # At tau=0.90: TP=8 (Sens=0.8), FP=1 (Spec=0.9) -> J = 0.70
    # At tau=0.50: TP=9 (Sens=0.9), FP=2 (Spec=0.8) -> J = 0.70 (EXACT TIE!)
    # Highest-threshold tie-break MUST pick tau_Youden = 0.90 (not 0.50)!
    # For tau_sens90: Sens >= 0.90 first occurs at tau = 0.50 (TP=9/10=0.90).
    # For tau_sens95: Sens >= 0.95 occurs only at tau = 0.20 (TP=10/10=1.00).
    p_vals = np.array([0.90] * 8 + [0.50, 0.20] + [0.95, 0.60] + [0.30] * 8, dtype=np.float64)
    y = np.array([1.0] * 10 + [0.0] * 10, dtype=np.float64)
    logits = scipy.special.logit(p_vals)

    th = select_thresholds(y, logits)
    assert th["tau_youden"] == pytest.approx(0.90, abs=1e-12)
    assert th["tau_sens90"] == pytest.approx(0.50, abs=1e-12)
    assert th["tau_sens95"] == pytest.approx(0.20, abs=1e-12)

    # Verify operating decision rule p >= tau matches exactly
    m90 = compute_operating_metrics(y, logits, th["tau_sens90"])
    assert m90["sensitivity"] == pytest.approx(0.90, abs=1e-12)
    assert m90["specificity"] == pytest.approx(0.80, abs=1e-12)


def test_slope_estimation_and_extreme_logit_stability():
    """Verifies unpenalized logistic regression slope vs sklearn and PRE-E-001 -123.38 logit stability."""
    rng = np.random.default_rng(99)
    n = 2000
    z = rng.normal(0.0, 1.5, size=n)
    y = rng.binomial(1, scipy.special.expit(-2.2 + 0.85 * z)).astype(np.float64)

    # Inject PRE-E-001 style extreme finite logit (with both negative and positive label tests)
    z_out = z.copy()
    z_out[0] = -123.38
    y_out = y.copy()
    y_out[0] = 1.0  # worst-case where clipped y_score would severely distort NLL!

    # Compare damped Newton GLM against sklearn unpenalized LogisticRegression(C=np.inf)
    clf = LogisticRegression(C=np.inf, solver="lbfgs", tol=1e-12, max_iter=1000)
    clf.fit(z_out.reshape(-1, 1), y_out)
    a_glm, b_glm = fit_binomial_glm_logits(z_out, y_out)

    assert a_glm == pytest.approx(float(clf.intercept_[0]), abs=1e-7)
    assert b_glm == pytest.approx(float(clf.coef_[0, 0]), abs=1e-7)

    # Verify NLL uses exact logaddexp (~123.38 / N extra) rather than clipped y_score (~16.1 / N)
    nll_exact = stable_nll_from_logits(z_out, y_out)
    nll_rest = stable_nll_from_logits(z_out[1:], y_out[1:])
    assert (n * nll_exact - (n - 1) * nll_rest) == pytest.approx(123.38, abs=1e-5)

    cal = compute_calibration_suite(z_out, y_out, n_bins=15)
    assert np.isfinite(cal["nll"]) and np.isfinite(cal["brier"]) and np.isfinite(cal["bss"])
    assert cal["slope"] == pytest.approx(b_glm, abs=1e-12)


def test_label_policy_counts_match_analysis_plan_exact_totals():
    """Verifies §3.1-3.4 label policy logic reproduces exact plan counts (191027; 4675; 20656; 175046; 6550)."""
    # Exact counts from ANALYSIS_PLAN_TRANSFER.md §3:
    # Positive (1.0): 4,675; Uncertain (-1.0): 15,981; Negative (0.0): 1,875; Blank (NaN): 168,496
    raw = np.concatenate([
        np.full(4675, 1.0),
        np.full(15981, -1.0),
        np.full(1875, 0.0),
        np.full(168496, np.nan),
    ])
    # Assign deterministic patient clusters matching ~64,534 patients
    pids = np.array([f"patient{i % 64534:05d}" for i in range(len(raw))], dtype=object)

    uz = apply_label_policy(raw, "u_zeros", pids)
    uo = apply_label_policy(raw, "u_ones", pids)
    ue = apply_label_policy(raw, "u_excluded", pids)
    ex = apply_label_policy(raw, "explicitly_labeled_only", pids)

    assert uz["n_images"] == 191027 and uz["n_pos_images"] == 4675 and uz["n_patients"] == 64534
    assert uo["n_images"] == 191027 and uo["n_pos_images"] == 20656
    assert ue["n_images"] == 175046 and ue["n_pos_images"] == 4675
    assert ex["n_images"] == 6550 and ex["n_pos_images"] == 4675


def test_delong_and_ece_and_target_10pct_platt_exactness():
    """Verifies O(N log N) DeLong vs brute force, 15-bin ECE, and 20-split patient-disjoint Platt."""
    rng = np.random.default_rng(11)
    za = np.round(rng.normal(size=120), 2)
    ya = rng.binomial(1, scipy.special.expit(za)).astype(np.float64)
    zb = np.round(rng.normal(size=150), 2)
    yb = rng.binomial(1, scipy.special.expit(0.5 * zb - 0.3)).astype(np.float64)

    # Brute-force DeLong structural matrix check
    def brute_auc_var(z, y):
        pos = z[y == 1.0]
        neg = z[y == 0.0]
        v10 = np.array([np.mean((pi > neg) + 0.5 * (pi == neg)) for pi in pos])
        v01 = np.array([np.mean((pos > nj) + 0.5 * (pos == nj)) for nj in neg])
        return float(v10.mean()), float(np.var(v10, ddof=1) / len(pos) + np.var(v01, ddof=1) / len(neg))

    auc_a, va = brute_auc_var(za, ya)
    auc_b, vb = brute_auc_var(zb, yb)
    dl = delong_two_sample(za, ya, zb, yb)
    assert fast_auroc(za, ya) == pytest.approx(roc_auc_score(ya, za), abs=1e-12)
    assert dl["delta_auroc"] == pytest.approx(auc_a - auc_b, abs=1e-12)
    assert dl["se"] == pytest.approx(np.sqrt(va + vb), abs=1e-12)

    # 15-bin ECE checks
    p_const = np.full(150, 0.2)
    y_bin = np.concatenate([np.ones(30), np.zeros(120)])
    assert compute_ece_uniform(y_bin, p_const, n_bins=15) == pytest.approx(0.0, abs=1e-12)
    assert compute_ece_adaptive(y_bin, np.linspace(0.01, 0.99, 150), n_bins=15) >= 0.0

    # Target-10% Platt: 20 splits, strict patient disjointness
    pids = np.repeat(np.arange(100), 6)
    z_all = rng.normal(size=600)
    y_all = rng.binomial(1, scipy.special.expit(0.7 * z_all - 1.0)).astype(np.float64)
    res_p = target_10pct_platt(z_all, y_all, pids, n_splits=20, base_seed=100)
    assert res_p["n_splits"] == 20
    assert len(res_p["splits"]) == 20
    assert all(sp["n_fit_patients"] == 10 and sp["n_eval_patients"] == 90 for sp in res_p["splits"])
    assert np.isfinite(res_p["nll_mean"]) and res_p["nll_sd"] >= 0.0


def test_provenance_rejection_and_synthetic_pipeline_with_pre_e_001(tmp_path: Path):
    """Tests strict provenance refusal rules plus end-to-end synthetic run with PRE-E-001 rows."""
    # Setup minimal valid directory mirroring oldckpt_real_v1 schema
    plan_p = REPO_ROOT / "ANALYSIS_PLAN_TRANSFER.md"
    man_p = REPO_ROOT / "configs" / "checkpoint_manifest.json"
    ref_p = REPO_ROOT / "configs" / "nih_test_reference_logits.npz"

    plan_sha = sha256_file(plan_p, lf_normalize=True)
    man_sha = sha256_file(man_p, lf_normalize=True)
    ref_sha = sha256_file(ref_p, lf_normalize=False)
    man_data = json.loads(man_p.read_text(encoding="utf-8"))

    run_dir = tmp_path / "run_ok"
    run_dir.mkdir()

    gate_payload = {
        "status": "PASSED",
        "smoke": False,
        "run_id": "oldckpt_real_v1",
        "plan_sha256": plan_sha,
        "manifest_sha256": man_sha,
        "reference_bundle_sha256": ref_sha,
        "git_commit": "d79bc756ba9a60d628674f2af1a683bbc25e2e6e",
        "gpu_name": "Tesla T4",
        "torch_version": "2.11.0+cu130",
    }
    (run_dir / "NIH_GATE_PASSED.json").write_text(json.dumps(gate_payload), encoding="utf-8")
    (run_dir / "EXCLUSIONS.json").write_text(json.dumps({"n_excluded": 0, "exclusions": []}), encoding="utf-8")

    rng = np.random.default_rng(55)
    n_val, n_test, n_cx = 120, 120, 150
    pids_val = np.array([f"nih_{i // 3}" for i in range(n_val)])
    pids_test = np.array([f"nih_t_{i // 3}" for i in range(n_test)])
    pids_cx = np.array([f"patient{i // 3:05d}" for i in range(n_cx)])
    y_val = rng.binomial(1, 0.25, size=n_val).astype(np.float32)
    y_test = rng.binomial(1, 0.25, size=n_test).astype(np.float32)
    raw_cx = rng.choice([1.0, 0.0, -1.0, np.nan], size=n_cx, p=[0.15, 0.20, 0.15, 0.50]).astype(np.float32)
    label_cols = np.array(["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax", "No Finding"])

    for mkey, spec in man_data["checkpoints"].items():
        arch = spec["architecture"]
        seed = int(spec["seed"])
        csha = spec["sha256"]
        for split, nn, yy, pids in (("val", n_val, y_val, pids_val), ("test", n_test, y_test, pids_test)):
            lg = np.zeros((nn, 8), dtype=np.float32)
            lg[:, 5] = (1.6 * yy - 1.2 + rng.normal(0, 0.5, size=nn)).astype(np.float32)
            np.savez(
                run_dir / f"predictions_nih_{split}_{arch}_{seed}.npz",
                y_logits=lg,
                y_true=yy,
                patient_ids=pids,
                pneumonia_col_idx=np.int64(5),
                label_cols=label_cols,
                checkpoint_sha256=np.array(csha),
                run_id=np.array("oldckpt_real_v1"),
                data_source=np.array("real_inference"),
                analysis_plan_sha256=np.array(plan_sha),
                gpu_name=np.array("Tesla T4"),
                torch_version=np.array("2.11.0+cu130"),
            )

        lg_cx = np.zeros((n_cx, 8), dtype=np.float32)
        y_uz = np.where(raw_cx == 1.0, 1.0, 0.0).astype(np.float32)
        lg_cx[:, 5] = (1.2 * y_uz - 2.0 + rng.normal(0, 0.6, size=n_cx)).astype(np.float32)
        if mkey == "efficientnet_b0_2024":
            lg_cx[10, 5] = -123.38  # simulate extreme outlier
        np.savez(
            run_dir / f"predictions_chexpert_frontal_{arch}_{seed}.npz",
            y_logits=lg_cx,
            raw_pneumonia=raw_cx,
            y_true_u_zeros=y_uz,
            y_true_u_ones=np.where((raw_cx == 1.0) | (raw_cx == -1.0), 1.0, 0.0).astype(np.float32),
            patient_ids=pids_cx,
            metadata_path=np.array([f"CheXpert/patient{i // 3:05d}/study1/view1.jpg" for i in range(n_cx)]),
            metadata_ap_pa=np.array(["AP" if i % 2 == 0 else "PA" for i in range(n_cx)]),
            metadata_sex=np.array(["Female" if i % 3 == 0 else "Male" for i in range(n_cx)]),
            metadata_age=np.array([30.0 + (i % 50) for i in range(n_cx)], dtype=np.float32),
            pneumonia_col_idx=np.int64(5),
            label_cols=label_cols,
            row_order_hash=np.array("deadbeef" * 8),
            checkpoint_sha256=np.array(csha),
            run_id=np.array("oldckpt_real_v1"),
            data_source=np.array("real_inference"),
            analysis_plan_sha256=np.array(plan_sha),
            gpu_name=np.array("Tesla T4"),
            torch_version=np.array("2.11.0+cu130"),
        )

    bundle = load_real_run_bundle(run_dir, plan_p, man_p, ref_p)
    bundle["sensitivity_exclude_rows"] = [10]
    res = run_transfer_analysis(
        bundle=bundle,
        out_dir=tmp_path / "out_ok",
        script_path=REPO_ROOT / "scripts" / "analyze_transfer.py",
        n_bootstrap=30,
        bootstrap_seed=42,
        n_target_platt_splits=4,
    )
    assert len(res["primary_endpoints_6test_bh_family"]) == 6
    assert (tmp_path / "out_ok" / "results.json").is_file()
    assert (tmp_path / "out_ok" / "outlier_sensitivity_pre_e_001.csv").is_file()

    # Refusal tests: mutate GPU mismatch, bad data_source, bad plan hash
    bad_gate = dict(gate_payload, plan_sha256="0" * 64)
    (run_dir / "NIH_GATE_PASSED.json").write_text(json.dumps(bad_gate), encoding="utf-8")
    with pytest.raises(ValueError, match="plan_sha256 mismatch"):
        load_real_run_bundle(run_dir, plan_p, man_p, ref_p)

    # Restore gate, break gpu_name in one CheXpert file
    (run_dir / "NIH_GATE_PASSED.json").write_text(json.dumps(gate_payload), encoding="utf-8")
    p_cx_bad = run_dir / "predictions_chexpert_frontal_densenet121_42.npz"
    d_bad = dict(np.load(p_cx_bad, allow_pickle=False))
    d_bad["gpu_name"] = np.array("NVIDIA A100")
    np.savez(p_cx_bad, **d_bad)
    with pytest.raises(ValueError, match="gpu_name"):
        load_real_run_bundle(run_dir, plan_p, man_p, ref_p)

    # Break data_source
    d_bad["gpu_name"] = np.array("Tesla T4")
    d_bad["data_source"] = np.array("synthetic")
    np.savez(p_cx_bad, **d_bad)
    with pytest.raises(ValueError, match="real_inference"):
        load_real_run_bundle(run_dir, plan_p, man_p, ref_p)
