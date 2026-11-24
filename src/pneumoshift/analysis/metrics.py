"""Mathematically exact, logit-stable metrics, thresholds, and recalibration estimators."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
from sklearn.metrics import average_precision_score


def logit_to_prob(logits: np.ndarray) -> np.ndarray:
    """Computes probabilities sigma(z) in float64 without artificial clipping."""
    return scipy.special.expit(np.asarray(logits, dtype=np.float64))


def _bce_logits_loss(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes log(1 + exp(u)) - y * u without overflow or inf-inf cancellation."""
    term = np.where(u > 0.0, (1.0 - y) * u, -y * u)
    return np.logaddexp(0.0, -np.abs(u)) + term


def stable_nll_from_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Computes binary cross-entropy NLL stably directly from logits via logaddexp.

    Never computes log(y_score), avoiding inf/clipping distortion on extreme logits (PRE-E-001).
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    losses = _bce_logits_loss(z, y)
    if weights is None:
        return float(np.mean(losses))
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return float("nan")
    return float(np.dot(w, losses) / w_sum)


def fit_binomial_glm_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    weights: Optional[np.ndarray] = None,
    init: Tuple[float, float] = (0.0, 1.0),
    max_iter: int = 35,
) -> Tuple[float, float]:
    """Fits unpenalized logistic regression logit(P(Y=1)) = a + b * z via damped 2x2 Newton-Raphson.

    Matches statsmodels.api.GLM(..., family=Binomial()) and sklearn.LogisticRegression(C=np.inf)
    to machine precision (~1e-9) while remaining fast for bootstrap replicates and numerically
    stable on extreme finite logits (e.g., z = -123.38).
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        nz = w > 0.0
        z = z[nz]
        y = y[nz]
        w = w[nz]
    else:
        w = np.ones(len(z), dtype=np.float64)

    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    a, b = float(init[0]), float(init[1])
    z2 = z * z

    def obj(aa: float, bb: float) -> float:
        u = aa + bb * z
        if not np.isfinite(u).all():
            return float("inf")
        return float(np.dot(w, _bce_logits_loss(u, y)))

    cur_obj = obj(a, b)
    for _ in range(max_iter):
        u = a + b * z
        p = scipy.special.expit(u)
        r = w * (p - y)
        g0 = float(r.sum())
        g1 = float(np.dot(r, z))
        v = w * p * (1.0 - p)
        h00 = float(v.sum())
        h01 = float(np.dot(v, z))
        h11 = float(np.dot(v, z2))
        det = h00 * h11 - h01 * h01
        if det <= 1e-18:
            # Near-singular Hessian (e.g. complete separation in tiny subgroup) -> stop safely
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        if not (np.isfinite(da) and np.isfinite(db)):
            break

        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break

        # Cap Newton step magnitude to prevent overflow on quasi-separated tiny fixtures
        max_step_norm = max(abs(da), abs(db))
        scale = min(1.0, 10.0 / max_step_norm) if max_step_norm > 10.0 else 1.0
        da *= scale
        db *= scale

        step = 1.0
        accepted = False
        while step >= 1e-5:
            na = a - step * da
            nb = b - step * db
            new_obj = obj(na, nb)
            if np.isfinite(new_obj) and new_obj <= cur_obj + 1e-12 * max(1.0, abs(cur_obj)):
                a, b, cur_obj = na, nb, new_obj
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    return float(a), float(b)


def fit_temperature_scaling(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fits scalar T > 0 on validation logits by minimizing NLL: p = sigma(z / T)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 1.0

    def obj(log_t: float) -> float:
        t = float(np.exp(log_t))
        return stable_nll_from_logits(z / t, y)

    res = scipy.optimize.minimize_scalar(obj, bounds=(np.log(1e-3), np.log(100.0)), method="bounded")
    return float(np.exp(res.x))


def fit_platt_scaling(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """Fits unpenalized affine logistic regression (a, b): p = sigma(a * z + b)."""
    # fit_binomial_glm_logits solves a + b * z -> intercept b_platt, slope a_platt
    intercept, slope = fit_binomial_glm_logits(logits, y_true)
    return slope, intercept


def compute_ece_uniform(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Computes M=15 equal-width Expected Calibration Error on [0, 1]."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_prob, dtype=np.float64)
    n = len(yt)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign probabilities exactly at 1.0 to the last bin
    bin_idx = np.clip(np.digitize(yp, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bin_idx == b
        cnt = int(m.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(yt[m].mean()) - float(yp[m].mean()))
    return float(ece)


def compute_ece_adaptive(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Computes M=15 equal-mass (quantile) Expected Calibration Error."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_prob, dtype=np.float64)
    n = len(yt)
    if n == 0:
        return float("nan")
    order = np.argsort(yp, kind="mergesort")
    bins = np.array_split(order, n_bins)
    ece = 0.0
    for idx in bins:
        cnt = len(idx)
        if cnt > 0:
            ece += (cnt / n) * abs(float(yt[idx].mean()) - float(yp[idx].mean()))
    return float(ece)


def compute_calibration_suite(
    logits: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
    glm_init: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, float]:
    """Computes full pre-specified calibration suite directly from logits (§8.2)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    if len(y) == 0:
        return {k: float("nan") for k in (
            "ece_uniform", "ece_adaptive", "brier", "bss", "nll", "slope", "intercept",
            "prevalence", "mean_pred",
        )}
    p = logit_to_prob(z)
    prev = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    brier_ref = prev * (1.0 - prev)
    bss = float(1.0 - (brier / brier_ref)) if brier_ref > 0.0 else float("nan")
    nll = stable_nll_from_logits(z, y)
    intercept, slope = fit_binomial_glm_logits(z, y, init=glm_init)
    return {
        "ece_uniform": compute_ece_uniform(y, p, n_bins=n_bins),
        "ece_adaptive": compute_ece_adaptive(y, p, n_bins=n_bins),
        "brier": brier,
        "bss": bss,
        "nll": nll,
        "slope": slope,
        "intercept": intercept,
        "prevalence": prev,
        "mean_pred": float(p.mean()),
    }


def _auroc_tiers(logits: np.ndarray, y_true: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Precomputes ascending unique logit tiers for exact O(N log N) AUROC & DeLong."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    _, inv = np.unique(z, return_inverse=True)
    m = int(inv.max()) + 1 if len(inv) > 0 else 0
    return m, inv, (y == 1.0), (y == 0.0)


def fast_auroc(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Computes exact AUROC using midrank concordant pairs on ascending logit tiers."""
    y = np.asarray(y_true, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    m, inv, pos_mask, neg_mask = _auroc_tiers(logits, y)
    p_k = np.bincount(inv[pos_mask], minlength=m).astype(np.float64)
    n_k = np.bincount(inv[neg_mask], minlength=m).astype(np.float64)
    p_tot = float(p_k.sum())
    n_tot = float(n_k.sum())
    n_before = np.empty(m, dtype=np.float64)
    n_before[0] = 0.0
    if m > 1:
        np.cumsum(n_k[:-1], out=n_before[1:])
    concordant = float(np.dot(p_k, n_before + 0.5 * n_k))
    return float(concordant / (p_tot * n_tot))


def compute_auprc(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Computes exact AUPRC directly from logits (monotone-invariant rank ordering)."""
    y = np.asarray(y_true, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(logits, dtype=np.float64)))


def _delong_one_cohort(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, int, int]:
    """Computes AUROC and asymptotic DeLong variance (Sun & Xu 2014 exact midrank form)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    m, inv, pos_mask, neg_mask = _auroc_tiers(z, y)
    p_k = np.bincount(inv[pos_mask], minlength=m).astype(np.float64)
    n_k = np.bincount(inv[neg_mask], minlength=m).astype(np.float64)
    p_tot = int(p_k.sum())
    n_tot = int(n_k.sum())
    if p_tot < 2 or n_tot < 2:
        return float("nan"), float("nan"), p_tot, n_tot

    n_before = np.empty(m, dtype=np.float64)
    n_before[0] = 0.0
    if m > 1:
        np.cumsum(n_k[:-1], out=n_before[1:])

    p_after = np.empty(m, dtype=np.float64)
    p_after[-1] = 0.0
    if m > 1:
        np.cumsum(p_k[:0:-1], out=p_after[-2::-1])

    v10_tiers = (n_before + 0.5 * n_k) / float(n_tot)
    v01_tiers = (p_after + 0.5 * p_k) / float(p_tot)

    v10 = v10_tiers[inv[pos_mask]]
    v01 = v01_tiers[inv[neg_mask]]
    auc = float(v10.mean())
    var = float(np.var(v10, ddof=1) / p_tot + np.var(v01, ddof=1) / n_tot)
    return auc, var, p_tot, n_tot


def delong_two_sample(
    logits_a: np.ndarray,
    y_a: np.ndarray,
    logits_b: np.ndarray,
    y_b: np.ndarray,
) -> Dict[str, Any]:
    """Two-sample unpaired DeLong test for AUROC(A) - AUROC(B) (§5.6 sensitivity analysis)."""
    auc_a, var_a, pa, na = _delong_one_cohort(logits_a, y_a)
    auc_b, var_b, pb, nb = _delong_one_cohort(logits_b, y_b)
    diff = auc_a - auc_b
    se = float(np.sqrt(max(var_a + var_b, 0.0)))
    if se <= 0.0 or not np.isfinite(se):
        z_stat, p_val = float("nan"), float("nan")
        ci_low, ci_high = float("nan"), float("nan")
    else:
        z_stat = float(diff / se)
        p_val = float(2.0 * scipy.stats.norm.sf(abs(z_stat)))
        ci_low = float(diff - 1.959963984540054 * se)
        ci_high = float(diff + 1.959963984540054 * se)
    return {
        "auroc_a": auc_a,
        "auroc_b": auc_b,
        "delta_auroc": diff,
        "se": se,
        "z_stat": z_stat,
        "p_value": p_val,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_pos_a": pa,
        "n_neg_a": na,
        "n_pos_b": pb,
        "n_neg_b": nb,
        "caveat": "Assumes independent observations; ignores patient-level clustering (§5.6).",
    }


def select_thresholds(y_true: np.ndarray, logits_val: np.ndarray) -> Dict[str, float]:
    """Selects tau_Youden, tau_sens90, and tau_sens95 on source validation probabilities.

    Implements §9.1-9.3 exact rules on p = sigma(z) with decision rule (p >= tau):
      - tau_Youden: max { argmax_tau (Sens(tau) + Spec(tau) - 1) } (highest threshold on ties).
      - tau_sens90: max { tau : Sens(tau) >= 0.90 }.
      - tau_sens95: max { tau : Sens(tau) >= 0.95 }.
    Uses integer cross-products (tp * N_neg + tn * P_pos) to prevent float rounding from
    breaking exact ties arbitrarily.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = logit_to_prob(logits_val)
    p_tot = int((y == 1.0).sum())
    n_tot = int((y == 0.0).sum())
    if p_tot == 0 or n_tot == 0:
        raise ValueError("Cannot select thresholds on single-class validation labels.")

    # Sort descending by predicted probability so prefix sweeps p >= tau cleanly
    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]

    # Collapse identical probability tiers (all ties share exact threshold p_tier)
    change_idx = np.where(np.diff(p_sorted) != 0.0)[0]
    tier_ends = np.append(change_idx, len(p_sorted) - 1)
    tier_probs = p_sorted[tier_ends]

    cum_tp = np.cumsum(y_sorted)[tier_ends].astype(np.int64)
    cum_fp = ((tier_ends + 1) - cum_tp).astype(np.int64)
    cum_tn = n_tot - cum_fp

    # Exact integer numerator for (Sens + Spec - 1) * P_tot * N_tot = tp * N_tot + tn * P_tot - P_tot * N_tot
    j_num = cum_tp * n_tot + cum_tn * p_tot
    max_j = int(j_num.max())
    # Since max_J >= N_tot * P_tot corresponds to J >= 0 (every threshold below min achieves J=0,
    # and threshold > max(p) gives tp=0, tn=N -> J=0 with j_num = P_tot * N_tot), handle trivial J<=0 safely
    youden_candidates = tier_probs[j_num == max_j]
    tau_youden = float(np.max(youden_candidates))

    # Fixed sensitivity >= 0.90 (exact: 100 * tp >= 90 * P_tot -> 10 * tp >= 9 * P_tot)
    elig_90 = np.where(cum_tp * 10 >= 9 * p_tot)[0]
    tau_sens90 = float(np.max(tier_probs[elig_90])) if len(elig_90) > 0 else float(tier_probs[-1])

    # Fixed sensitivity >= 0.95 (exact: 20 * tp >= 19 * P_tot)
    elig_95 = np.where(cum_tp * 20 >= 19 * p_tot)[0]
    tau_sens95 = float(np.max(tier_probs[elig_95])) if len(elig_95) > 0 else float(tier_probs[-1])

    return {
        "tau_youden": tau_youden,
        "tau_sens90": tau_sens90,
        "tau_sens95": tau_sens95,
    }


def compute_operating_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    threshold_prob: float,
) -> Dict[str, float]:
    """Computes operating-point metrics under decision rule: positive if p >= tau (§9.1)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = logit_to_prob(logits)
    pred = p >= threshold_prob
    pos = y == 1.0
    neg = y == 0.0
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & neg))
    tn = int(np.sum((~pred) & neg))
    fn = int(np.sum((~pred) & pos))
    return {
        "threshold": float(threshold_prob),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "ppv": float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": int(len(y)),
    }


def apply_label_policy(
    raw_pneumonia: np.ndarray,
    policy: str,
    patient_ids: np.ndarray,
) -> Dict[str, Any]:
    """Applies one of the 4 CheXpert label mapping policies (§3.1-3.4) + patient counts."""
    raw = np.asarray(raw_pneumonia, dtype=np.float64)
    pids = np.asarray(patient_ids, dtype=object)

    if policy == "u_zeros":
        mask = np.ones(len(raw), dtype=bool)
        y = np.where(raw == 1.0, 1.0, 0.0)
    elif policy == "u_ones":
        mask = np.ones(len(raw), dtype=bool)
        y = np.where((raw == 1.0) | (raw == -1.0), 1.0, 0.0)
    elif policy == "u_excluded":
        mask = raw != -1.0
        y = np.where(raw[mask] == 1.0, 1.0, 0.0)
    elif policy == "explicitly_labeled_only":
        mask = (raw == 1.0) | (raw == 0.0)
        y = np.where(raw[mask] == 1.0, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown label policy: {policy!r}")

    sub_pids = pids[mask]
    n_images = int(mask.sum())
    n_pos_images = int((y == 1.0).sum())
    n_patients = int(len(np.unique(sub_pids))) if n_images > 0 else 0
    n_pos_patients = int(len(np.unique(sub_pids[y == 1.0]))) if n_pos_images > 0 else 0

    return {
        "policy": policy,
        "mask": mask,
        "y_true": y,
        "patient_ids": sub_pids,
        "n_images": n_images,
        "n_pos_images": n_pos_images,
        "image_prevalence": float(n_pos_images / n_images) if n_images > 0 else float("nan"),
        "n_patients": n_patients,
        "n_pos_patients": n_pos_patients,
        "patient_prevalence": float(n_pos_patients / n_patients) if n_patients > 0 else float("nan"),
    }


def target_10pct_platt(
    logits: np.ndarray,
    y_true: np.ndarray,
    patient_ids: np.ndarray,
    n_splits: int = 20,
    base_seed: int = 100,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Evaluates target-10% patient-level Platt recalibration across 20 random splits (§7.3)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    pids = np.asarray(patient_ids, dtype=object)

    unique_pats, pat_inv = np.unique(pids, return_inverse=True)
    n_pats = len(unique_pats)
    n_fit_pats = max(1, int(round(0.10 * n_pats)))

    split_records: List[Dict[str, Any]] = []
    metric_keys = ("ece_uniform", "ece_adaptive", "brier", "bss", "nll", "slope", "intercept")

    for i in range(n_splits):
        split_seed = base_seed + i
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(n_pats)
        fit_pats = perm[:n_fit_pats]
        is_fit_pat = np.zeros(n_pats, dtype=bool)
        is_fit_pat[fit_pats] = True
        fit_mask = is_fit_pat[pat_inv]
        eval_mask = ~fit_mask

        a_hat, b_hat = fit_platt_scaling(z[fit_mask], y[fit_mask])
        if np.isfinite(a_hat) and np.isfinite(b_hat):
            scaled_eval_z = a_hat * z[eval_mask] + b_hat
            cal = compute_calibration_suite(scaled_eval_z, y[eval_mask], n_bins=n_bins)
        else:
            cal = {mk: float("nan") for mk in metric_keys}

        rec = {
            "split_index": i,
            "seed": split_seed,
            "n_fit_patients": int(n_fit_pats),
            "n_eval_patients": int(n_pats - n_fit_pats),
            "n_fit_images": int(fit_mask.sum()),
            "n_eval_images": int(eval_mask.sum()),
            "platt_a": float(a_hat),
            "platt_b": float(b_hat),
        }
        for mk in metric_keys:
            rec[mk] = float(cal[mk])
        split_records.append(rec)

    summary: Dict[str, Any] = {
        "n_splits": n_splits,
        "seeds": [base_seed + i for i in range(n_splits)],
        "splits": split_records,
    }
    for key in ("platt_a", "platt_b") + metric_keys:
        vals = np.array([r[key] for r in split_records], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(vals))
        summary[f"{key}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary


def extract_subgroup_masks(
    view_position: Optional[np.ndarray],
    sex: Optional[np.ndarray],
    age: Optional[np.ndarray],
    n_rows: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Constructs mutually exclusive exploratory subgroup masks (§11)."""
    subgroups: Dict[str, Dict[str, np.ndarray]] = {}

    if view_position is not None and len(view_position) == n_rows:
        vp = np.char.upper(np.char.strip(view_position.astype(str)))
        subgroups["projection"] = {
            "AP": vp == "AP",
            "PA": vp == "PA",
        }

    if sex is not None and len(sex) == n_rows:
        sx = np.char.upper(np.char.strip(sex.astype(str)))
        subgroups["sex"] = {
            "Female": (sx == "FEMALE") | (sx == "F"),
            "Male": (sx == "MALE") | (sx == "M"),
        }

    if age is not None and len(age) == n_rows:
        ag = np.asarray(age, dtype=np.float64)
        valid_age = np.isfinite(ag) & (ag >= 0.0) & (ag <= 120.0)
        subgroups["age_band"] = {
            "<40": valid_age & (ag < 40.0),
            "40-64": valid_age & (ag >= 40.0) & (ag < 65.0),
            ">=65": valid_age & (ag >= 65.0),
        }

    return subgroups
