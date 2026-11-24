"""Patient-clustered paired multi-seed bootstrap engine and BH correction (§4-§5)."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from pneumoshift.analysis.metrics import (
    _auroc_tiers,
    compute_operating_metrics,
    fast_auroc,
    fit_binomial_glm_logits,
    logit_to_prob,
)


def two_sided_bootstrap_pvalue(boot_diffs: np.ndarray) -> float:
    """Computes two-sided pseudocount continuity-corrected bootstrap p-value (§5.4).

    p = min(1, 2 * min((1 + #{theta* <= 0}) / (B + 1), (1 + #{theta* >= 0}) / (B + 1)))
    """
    arr = np.asarray(boot_diffs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    b = len(arr)
    if b == 0:
        return float("nan")
    p_le = (1.0 + float(np.sum(arr <= 0.0))) / (b + 1.0)
    p_ge = (1.0 + float(np.sum(arr >= 0.0))) / (b + 1.0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """Applies Benjamini-Hochberg FDR control across a finite hypothesis family (§4.2)."""
    m = len(p_values)
    if m == 0:
        return [], []
    p_arr = np.clip(np.asarray(p_values, dtype=np.float64), 0.0, 1.0)
    order = np.argsort(p_arr, kind="mergesort")
    sorted_p = p_arr[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    raw_q = sorted_p * (m / ranks)
    adj_sorted = np.clip(np.minimum.accumulate(raw_q[::-1])[::-1], 0.0, 1.0)
    adj = np.empty(m, dtype=np.float64)
    adj[order] = adj_sorted
    sig = [bool(q <= alpha) for q in adj]
    return adj.tolist(), sig


class ClusterBootstrapPlan:
    """Generates and replays identical patient-cluster resamples across all models/seeds."""

    def __init__(
        self,
        patient_ids_a: np.ndarray,
        patient_ids_b: np.ndarray,
        n_resamples: int = 2000,
        seed: int = 42,
    ) -> None:
        self.n_resamples = int(n_resamples)
        self.seed = int(seed)

        _, self.group_a = np.unique(np.asarray(patient_ids_a, dtype=object), return_inverse=True)
        _, self.group_b = np.unique(np.asarray(patient_ids_b, dtype=object), return_inverse=True)
        self.n_clusters_a = int(self.group_a.max()) + 1 if len(self.group_a) > 0 else 0
        self.n_clusters_b = int(self.group_b.max()) + 1 if len(self.group_b) > 0 else 0

        # Generate exact seed streams deterministically so every architecture & seed uses
        # identical patient cluster resamples for Cohort A (NIH) and Cohort B (CheXpert).
        master_ss = np.random.SeedSequence(self.seed)
        self._ss_a, self._ss_b = master_ss.spawn(2)

    def iter_cluster_weights(self):
        """Yields (weights_a, weights_b) integer cluster frequency vectors for b=1..B."""
        rng_a = np.random.default_rng(self._ss_a)
        rng_b = np.random.default_rng(self._ss_b)
        ga = self.n_clusters_a
        gb = self.n_clusters_b
        for _ in range(self.n_resamples):
            cw_a = np.bincount(rng_a.integers(0, ga, size=ga), minlength=ga)
            cw_b = np.bincount(rng_b.integers(0, gb, size=gb), minlength=gb)
            yield cw_a, cw_b


class _CohortSeedPrecomp:
    """Precomputed arrays enabling O(N) exact weighted AUROC, GLM slope, and threshold stats."""

    def __init__(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        group: np.ndarray,
        n_clusters: int,
        thresholds: Dict[str, float],
    ) -> None:
        self.z = np.asarray(logits, dtype=np.float64)
        self.y = np.asarray(y_true, dtype=np.float64)
        self.group = group
        self.n_clusters = n_clusters

        # Exact AUROC tier structures
        self.m_tiers, self.inv, pos_mask, neg_mask = _auroc_tiers(self.z, self.y)
        self.inv_pos = self.inv[pos_mask]
        self.inv_neg = self.inv[neg_mask]
        self.g_pos = group[pos_mask]
        self.g_neg = group[neg_mask]

        # Full-sample GLM init for 2-3 step convergence inside bootstrap
        self.a0, self.b0 = fit_binomial_glm_logits(self.z, self.y)

        # Patient-aggregated confusion counts per fixed threshold (makes 2000 reps ~0.05s total!)
        p = logit_to_prob(self.z)
        self.thresh_cluster_counts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        is_pos = self.y == 1.0
        is_neg = self.y == 0.0
        for tname, tau in thresholds.items():
            pred = p >= tau
            tp_c = np.bincount(group[pred & is_pos], minlength=n_clusters).astype(np.float64)
            fp_c = np.bincount(group[pred & is_neg], minlength=n_clusters).astype(np.float64)
            tn_c = np.bincount(group[(~pred) & is_neg], minlength=n_clusters).astype(np.float64)
            fn_c = np.bincount(group[(~pred) & is_pos], minlength=n_clusters).astype(np.float64)
            self.thresh_cluster_counts[tname] = (tp_c, fp_c, tn_c, fn_c)

    def auroc_weighted(self, cw: np.ndarray) -> float:
        p_k = np.bincount(self.inv_pos, weights=cw[self.g_pos], minlength=self.m_tiers)
        n_k = np.bincount(self.inv_neg, weights=cw[self.g_neg], minlength=self.m_tiers)
        p_tot = float(p_k.sum())
        n_tot = float(n_k.sum())
        if p_tot <= 0.0 or n_tot <= 0.0:
            return float("nan")
        n_before = np.empty(self.m_tiers, dtype=np.float64)
        n_before[0] = 0.0
        if self.m_tiers > 1:
            np.cumsum(n_k[:-1], out=n_before[1:])
        return float(np.dot(p_k, n_before + 0.5 * n_k) / (p_tot * n_tot))

    def slope_intercept_weighted(self, cw: np.ndarray) -> Tuple[float, float]:
        w = cw[self.group].astype(np.float64)
        return fit_binomial_glm_logits(self.z, self.y, weights=w, init=(self.a0, self.b0))

    def op_metrics_weighted(self, cw: np.ndarray, tname: str) -> Dict[str, float]:
        tp_c, fp_c, tn_c, fn_c = self.thresh_cluster_counts[tname]
        cw_f = cw.astype(np.float64)
        tp = float(np.dot(cw_f, tp_c))
        fp = float(np.dot(cw_f, fp_c))
        tn = float(np.dot(cw_f, tn_c))
        fn = float(np.dot(cw_f, fn_c))
        return {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0.0 else float("nan"),
            "specificity": tn / (tn + fp) if (tn + fp) > 0.0 else float("nan"),
            "ppv": tp / (tp + fp) if (tp + fp) > 0.0 else float("nan"),
            "npv": tn / (tn + fn) if (tn + fn) > 0.0 else float("nan"),
        }


def _summarize_boot(point: float, reps: np.ndarray) -> Dict[str, float]:
    """Computes percentile 95% CI and §5.4 two-sided bootstrap p-value."""
    valid = reps[np.isfinite(reps)]
    if len(valid) == 0:
        return {"estimate": float(point), "ci_lower": float("nan"), "ci_upper": float("nan"), "p_value": float("nan"), "n_valid": 0}
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {
        "estimate": float(point),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "p_value": two_sided_bootstrap_pvalue(valid),
        "n_valid": int(len(valid)),
    }


def compute_paired_architecture_bootstrap(
    plan: ClusterBootstrapPlan,
    y_a: np.ndarray,
    y_b: np.ndarray,
    seed_inputs: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes paired 3-seed patient-clustered bootstrap contrasts for one architecture.

    Args:
        plan: Pre-seeded ClusterBootstrapPlan (Cohort A = NIH test, Cohort B = CheXpert).
        y_a: Binary ground truth for Cohort A.
        y_b: Binary ground truth for Cohort B.
        seed_inputs: Mapping seed -> {
            'logits_a': 1D array, 'logits_b': 1D array,
            'thresholds': {'tau_youden': float, 'tau_sens90': float, 'tau_sens95': float}
        }
    """
    seeds = sorted(seed_inputs.keys())
    k_seeds = len(seeds)
    b_iters = plan.n_resamples

    thresh_names = ("tau_sens90", "tau_youden", "tau_sens95")
    scalar_metrics = [
        "auroc",
        "slope",
        "intercept",
        "sens_tau_sens90",
        "spec_tau_sens90",
        "ppv_tau_sens90",
        "npv_tau_sens90",
        "sens_tau_youden",
        "spec_tau_youden",
        "ppv_tau_youden",
        "npv_tau_youden",
        "sens_tau_sens95",
        "spec_tau_sens95",
        "ppv_tau_sens95",
        "npv_tau_sens95",
    ]

    point_a: Dict[int, Dict[str, float]] = {s: {} for s in seeds}
    point_b: Dict[int, Dict[str, float]] = {s: {} for s in seeds}
    point_d: Dict[int, Dict[str, float]] = {s: {} for s in seeds}

    pre_a: Dict[int, _CohortSeedPrecomp] = {}
    pre_b: Dict[int, _CohortSeedPrecomp] = {}

    for s in seeds:
        sin = seed_inputs[s]
        za = np.asarray(sin["logits_a"], dtype=np.float64)
        zb = np.asarray(sin["logits_b"], dtype=np.float64)
        th = sin["thresholds"]

        pa_obj = _CohortSeedPrecomp(za, y_a, plan.group_a, plan.n_clusters_a, th)
        pb_obj = _CohortSeedPrecomp(zb, y_b, plan.group_b, plan.n_clusters_b, th)
        pre_a[s] = pa_obj
        pre_b[s] = pb_obj

        point_a[s]["auroc"] = fast_auroc(za, y_a)
        point_b[s]["auroc"] = fast_auroc(zb, y_b)
        point_a[s]["intercept"], point_a[s]["slope"] = pa_obj.a0, pa_obj.b0
        point_b[s]["intercept"], point_b[s]["slope"] = pb_obj.a0, pb_obj.b0

        for tname in thresh_names:
            ma = compute_operating_metrics(y_a, za, th[tname])
            mb = compute_operating_metrics(y_b, zb, th[tname])
            for opk, pfx in (("sensitivity", "sens"), ("specificity", "spec"), ("ppv", "ppv"), ("npv", "npv")):
                mk = f"{pfx}_{tname}"
                point_a[s][mk] = ma[opk]
                point_b[s][mk] = mb[opk]

        for mk in scalar_metrics:
            point_d[s][mk] = point_a[s][mk] - point_b[s][mk]

    # Preallocate bootstrap replicate arrays: metric -> (k_seeds, B)
    reps_a = {mk: np.full((k_seeds, b_iters), np.nan, dtype=np.float64) for mk in scalar_metrics}
    reps_b = {mk: np.full((k_seeds, b_iters), np.nan, dtype=np.float64) for mk in scalar_metrics}
    reps_d = {mk: np.full((k_seeds, b_iters), np.nan, dtype=np.float64) for mk in scalar_metrics}

    for b_idx, (cw_a, cw_b) in enumerate(plan.iter_cluster_weights()):
        for s_idx, s in enumerate(seeds):
            pa_obj = pre_a[s]
            pb_obj = pre_b[s]

            auc_a = pa_obj.auroc_weighted(cw_a)
            auc_b = pb_obj.auroc_weighted(cw_b)
            ia, sa = pa_obj.slope_intercept_weighted(cw_a)
            ib, sb = pb_obj.slope_intercept_weighted(cw_b)

            reps_a["auroc"][s_idx, b_idx] = auc_a
            reps_b["auroc"][s_idx, b_idx] = auc_b
            reps_d["auroc"][s_idx, b_idx] = auc_a - auc_b

            reps_a["slope"][s_idx, b_idx] = sa
            reps_b["slope"][s_idx, b_idx] = sb
            reps_d["slope"][s_idx, b_idx] = sa - sb

            reps_a["intercept"][s_idx, b_idx] = ia
            reps_b["intercept"][s_idx, b_idx] = ib
            reps_d["intercept"][s_idx, b_idx] = ia - ib

            for tname in thresh_names:
                oma = pa_obj.op_metrics_weighted(cw_a, tname)
                omb = pb_obj.op_metrics_weighted(cw_b, tname)
                for opk, pfx in (("sensitivity", "sens"), ("specificity", "spec"), ("ppv", "ppv"), ("npv", "npv")):
                    mk = f"{pfx}_{tname}"
                    va, vb = oma[opk], omb[opk]
                    reps_a[mk][s_idx, b_idx] = va
                    reps_b[mk][s_idx, b_idx] = vb
                    reps_d[mk][s_idx, b_idx] = va - vb

    per_seed_summary: Dict[int, Dict[str, Any]] = {}
    for s_idx, s in enumerate(seeds):
        per_seed_summary[s] = {"cohort_a": {}, "cohort_b": {}, "contrast": {}}
        for mk in scalar_metrics:
            per_seed_summary[s]["cohort_a"][mk] = _summarize_boot(point_a[s][mk], reps_a[mk][s_idx])
            per_seed_summary[s]["cohort_b"][mk] = _summarize_boot(point_b[s][mk], reps_b[mk][s_idx])
            per_seed_summary[s]["contrast"][mk] = _summarize_boot(point_d[s][mk], reps_d[mk][s_idx])

    joint_summary: Dict[str, Any] = {"cohort_a": {}, "cohort_b": {}, "contrast": {}}
    for mk in scalar_metrics:
        for label, pdict, rdict in (
            ("cohort_a", point_a, reps_a),
            ("cohort_b", point_b, reps_b),
            ("contrast", point_d, reps_d),
        ):
            vals = np.array([pdict[s][mk] for s in seeds], dtype=np.float64)
            mean_pt = float(np.mean(vals))
            between_sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            mean_reps = np.mean(rdict[mk], axis=0)
            entry = _summarize_boot(mean_pt, mean_reps)
            entry["between_seed_sd_ddof1"] = between_sd
            joint_summary[label][mk] = entry

    return {
        "seeds": seeds,
        "n_resamples": b_iters,
        "bootstrap_seed": plan.seed,
        "per_seed": per_seed_summary,
        "joint": joint_summary,
    }
