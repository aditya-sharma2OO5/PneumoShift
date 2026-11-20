"""Regenerate PROVENANCE_TABLE.csv ensuring complete alignment with Task B provisional outputs."""

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_provenance(
    tables_dir: Path = PROJECT_ROOT / "results" / "tables" / "provisional_old_run",
    output_csv: Path = PROJECT_ROOT / "PROVENANCE_TABLE.csv",
    run_id: str = "old_run",
):
    t1_json_p = tables_dir / "table1_cross_domain_discrimination.json"
    t2_json_p = tables_dir / "table2_calibration_regimes_and_oracle_recovery.json"
    t3_json_p = tables_dir / "table3_decision_threshold_transfer.json"

    for p in [t1_json_p, t2_json_p, t3_json_p]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required table JSON: {p}")

    with open(t1_json_p, "r", encoding="utf-8") as fp:
        t1_data = json.load(fp)
    with open(t2_json_p, "r", encoding="utf-8") as fp:
        t2_data = json.load(fp)
    with open(t3_json_p, "r", encoding="utf-8") as fp:
        t3_data = json.load(fp)

    # Assert single run_id
    for name, d in [("Table 1", t1_data), ("Table 2", t2_data), ("Table 3", t3_data)]:
        if d.get("run_id") != run_id:
            raise ValueError(f"{name} run_id mismatch: expected '{run_id}', found '{d.get('run_id')}'")

    rows = []

    # -------------------------------------------------------------
    # Table 1 Items
    # -------------------------------------------------------------
    for arch_key, arch_disp in [("densenet121", "DenseNet121"), ("efficientnet_b0", "EfficientNet-B0")]:
        arch_t1 = t1_data["architectures"][arch_key]["summary_3seed"]
        in_d = arch_t1["in_domain"]
        tr_d = arch_t1["transferred"]
        gap_d = arch_t1["generalization_gap"]

        rows.append({
            "item": f"Table 1: {arch_disp} (In-Domain) AUROC",
            "value": f"{in_d['auroc_mean']:.3f} +/- {in_d['auroc_sd']:.3f}",
            "source file": f"results/tables/provisional_{run_id}/table1_cross_domain_discrimination.json",
            "JSON key": f"architectures.{arch_key}.summary_3seed.in_domain.auroc_mean",
            "data_source": "real",
            "how computed (script + line)": f"scripts/run_task_b_pipeline.py:270-340 (mean +/- SD across seeds on NIH Test N=22,744)",
        })
        rows.append({
            "item": f"Table 1: {arch_disp} (In-Domain) AUPRC",
            "value": f"{in_d['auprc_mean']:.3f} +/- {in_d['auprc_sd']:.3f}",
            "source file": f"results/tables/provisional_{run_id}/table1_cross_domain_discrimination.json",
            "JSON key": f"architectures.{arch_key}.summary_3seed.in_domain.auprc_mean",
            "data_source": "real",
            "how computed (script + line)": f"scripts/run_task_b_pipeline.py:270-340 (mean +/- SD average_precision_score across seeds on NIH Test)",
        })
        rows.append({
            "item": f"Table 1: {arch_disp} (Transferred) AUROC",
            "value": f"{tr_d['auroc_mean']:.3f} +/- {tr_d['auroc_sd']:.3f}",
            "source file": f"results/tables/provisional_{run_id}/table1_cross_domain_discrimination.json",
            "JSON key": f"architectures.{arch_key}.summary_3seed.transferred.auroc_mean",
            "data_source": "real",
            "how computed (script + line)": f"scripts/run_task_b_pipeline.py:270-340 (mean +/- SD across seeds on CheXpert N=17,551)",
        })
        rows.append({
            "item": f"Table 1: {arch_disp} (Transferred) AUPRC",
            "value": f"{tr_d['auprc_mean']:.3f} +/- {tr_d['auprc_sd']:.3f}",
            "source file": f"results/tables/provisional_{run_id}/table1_cross_domain_discrimination.json",
            "JSON key": f"architectures.{arch_key}.summary_3seed.transferred.auprc_mean",
            "data_source": "real",
            "how computed (script + line)": f"scripts/run_task_b_pipeline.py:270-340 (mean +/- SD average_precision_score across seeds on CheXpert)",
        })
        rows.append({
            "item": f"Table 1: {arch_disp} Generalization Gap",
            "value": f"{gap_d['delta_auroc_mean']:+.3f} +/- {gap_d['delta_between_seed_sd']:.3f}",
            "source file": f"results/tables/provisional_{run_id}/table1_cross_domain_discrimination.json",
            "JSON key": f"architectures.{arch_key}.summary_3seed.generalization_gap.delta_auroc_mean",
            "data_source": "real",
            "how computed (script + line)": f"scripts/run_task_b_pipeline.py:305-330 (mean AUROC difference NIH - CheXpert)",
        })
        rows.append({
            "item": f"Table 1: {arch_disp} Joint Bootstrap 95% CI",
            "value": f"[{gap_d['joint_bootstrap_95ci'][0]:.3f}, {gap_d['joint_bootstrap_95ci'][1]:.3f}]",
            "source file": f"results/tables/provisional_{run_id}/table1_cross_domain_discrimination.json",
            "JSON key": f"architectures.{arch_key}.summary_3seed.generalization_gap.joint_bootstrap_95ci",
            "data_source": "real",
            "how computed (script + line)": f"scripts/run_task_b_pipeline.py:280-310 (2000 joint bootstrap resamples across seeds)",
        })
        rows.append({
            "item": f"Table 1: {arch_disp} Raw p-value",
            "value": f"{gap_d['joint_bootstrap_p_val']:.4f}",
            "source file": f"results/tables/provisional_{run_id}/table1_cross_domain_discrimination.json",
            "JSON key": f"architectures.{arch_key}.summary_3seed.generalization_gap.joint_bootstrap_p_val",
            "data_source": "real",
            "how computed (script + line)": f"scripts/run_task_b_pipeline.py:315-325 (two-sided joint bootstrap p-value under H0)",
        })
        rows.append({
            "item": f"Table 1: {arch_disp} FDR q-value",
            "value": f"{gap_d['fdr_q_val']:.4f}",
            "source file": f"results/tables/provisional_{run_id}/table1_cross_domain_discrimination.json",
            "JSON key": f"architectures.{arch_key}.summary_3seed.generalization_gap.fdr_q_val",
            "data_source": "real",
            "how computed (script + line)": f"scripts/run_task_b_pipeline.py:350-365 (Benjamini-Hochberg FDR correction across architectures)",
        })

    # -------------------------------------------------------------
    # Table 2 Items
    # -------------------------------------------------------------
    for arch_key, arch_disp in [("densenet121", "DenseNet121"), ("efficientnet_b0", "EfficientNet-B0")]:
        arch_t2 = t2_data["architectures"][arch_key]["regimes"]
        for reg_k, short_name in [
            ("1. In-Domain Baseline (NIH Test)", "In-Domain Baseline"),
            ("2. Cross-Domain Uncalibrated (CheXpert)", "Cross-Domain Uncalibrated"),
            ("3. Source Temperature Scaling", "Source Temperature Scaled"),
            ("4. Source Platt Scaling", "Source Platt Scaled"),
            ("5. Target 10% Temperature Scaling (Oracle)", "Target 10% Temp Scaled"),
            ("6. Target 10% Platt Scaling (Oracle)", "Target 10% Platt Scaled"),
        ]:
            reg_d = arch_t2[reg_k]
            rows.append({
                "item": f"Table 2: {arch_disp} ({short_name}) ECE",
                "value": f"{reg_d['ece_uniform_mean']:.3f} +/- {reg_d['ece_uniform_sd']:.3f}",
                "source file": f"results/tables/provisional_{run_id}/table2_calibration_regimes_and_oracle_recovery.json",
                "JSON key": f"architectures.{arch_key}.regimes['{reg_k}'].ece_uniform_mean",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:430-515 (10-bin equal-width ECE)",
            })
            rows.append({
                "item": f"Table 2: {arch_disp} ({short_name}) Brier Score",
                "value": f"{reg_d['brier_score_mean']:.3f}",
                "source file": f"results/tables/provisional_{run_id}/table2_calibration_regimes_and_oracle_recovery.json",
                "JSON key": f"architectures.{arch_key}.regimes['{reg_k}'].brier_score_mean",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:125-135 (mean squared error of predicted probabilities)",
            })
            rows.append({
                "item": f"Table 2: {arch_disp} ({short_name}) BSS vs Base Rate",
                "value": f"{reg_d['bss_vs_base_mean']:.3f}",
                "source file": f"results/tables/provisional_{run_id}/table2_calibration_regimes_and_oracle_recovery.json",
                "JSON key": f"architectures.{arch_key}.regimes['{reg_k}'].bss_vs_base_mean",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:128-132 (Brier Skill Score = 1 - Brier / (p * (1 - p)))",
            })
            rows.append({
                "item": f"Table 2: {arch_disp} ({short_name}) NLL",
                "value": f"{reg_d['nll_mean']:.3f}",
                "source file": f"results/tables/provisional_{run_id}/table2_calibration_regimes_and_oracle_recovery.json",
                "JSON key": f"architectures.{arch_key}.regimes['{reg_k}'].nll_mean",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:133-135 (binary cross entropy log loss)",
            })
            rows.append({
                "item": f"Table 2: {arch_disp} ({short_name}) Recovery",
                "value": f"{reg_d['recovery_pct']:.1f}%" if "In-Domain" not in reg_k else "-",
                "source file": f"results/tables/provisional_{run_id}/table2_calibration_regimes_and_oracle_recovery.json",
                "JSON key": f"architectures.{arch_key}.regimes['{reg_k}'].recovery_pct",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:460-520 ((ECE_uncal - ECE_method) / ECE_uncal * 100)",
            })

    # -------------------------------------------------------------
    # Table 3 Items
    # -------------------------------------------------------------
    for arch_key, arch_disp in [("densenet121", "DenseNet121"), ("efficientnet_b0", "EfficientNet-B0")]:
        arch_t3 = t3_data["architectures"][arch_key]["policies"]
        for pol_k, short_name in [
            ("Source Youden J", "Source Youden J"),
            ("Source Sens-90", "Source Sens-90"),
            ("Source Max-F1", "Source Max-F1"),
            ("Oracle Target-Optimal", "Oracle Target-Optimal"),
        ]:
            pol_d = arch_t3[pol_k]
            rows.append({
                "item": f"Table 3: {arch_disp} ({short_name}) tau",
                "value": f"{pol_d['operating_tau_mean']:.2f}",
                "source file": f"results/tables/provisional_{run_id}/table3_decision_threshold_transfer.json",
                "JSON key": f"architectures.{arch_key}.policies['{pol_k}'].operating_tau_mean",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:640-720 (threshold selected on source validation set or target cal half)",
            })
            rows.append({
                "item": f"Table 3: {arch_disp} ({short_name}) Target Sens",
                "value": f"{pol_d['target_sensitivity_mean']:.3f} +/- {pol_d['target_sensitivity_sd']:.3f}",
                "source file": f"results/tables/provisional_{run_id}/table3_decision_threshold_transfer.json",
                "JSON key": f"architectures.{arch_key}.policies['{pol_k}'].target_sensitivity_mean",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:650-720 (evaluated on CheXpert 50% evaluation half over 20 repeats)",
            })
            rows.append({
                "item": f"Table 3: {arch_disp} ({short_name}) Target Spec",
                "value": f"{pol_d['target_specificity_mean']:.3f} +/- {pol_d['target_specificity_sd']:.3f}",
                "source file": f"results/tables/provisional_{run_id}/table3_decision_threshold_transfer.json",
                "JSON key": f"architectures.{arch_key}.policies['{pol_k}'].target_specificity_mean",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:650-720 (evaluated on CheXpert 50% evaluation half over 20 repeats)",
            })
            rows.append({
                "item": f"Table 3: {arch_disp} ({short_name}) Target F1",
                "value": f"{pol_d['target_f1_mean']:.3f} [{pol_d['target_f1_95ci_bootstrap'][0]:.3f}, {pol_d['target_f1_95ci_bootstrap'][1]:.3f}]",
                "source file": f"results/tables/provisional_{run_id}/table3_decision_threshold_transfer.json",
                "JSON key": f"architectures.{arch_key}.policies['{pol_k}'].target_f1_mean",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:680-720 (mean F1 with 2000 bootstrap replicates)",
            })
            rows.append({
                "item": f"Table 3: {arch_disp} ({short_name}) F1 Drop",
                "value": f"{pol_d['f1_drop_vs_oracle']:+.3f}" if "Oracle" not in pol_k else "0.000 (Baseline)",
                "source file": f"results/tables/provisional_{run_id}/table3_decision_threshold_transfer.json",
                "JSON key": f"architectures.{arch_key}.policies['{pol_k}'].f1_drop_vs_oracle",
                "data_source": "real",
                "how computed (script + line)": f"scripts/run_task_b_pipeline.py:690-720 (policy F1 minus oracle F1)",
            })

    # Abstract provisional recovery tracking
    rec_d121_platt = t2_data["architectures"]["densenet121"]["regimes"]["4. Source Platt Scaling"]["recovery_pct"]
    rec_eff_platt = t2_data["architectures"]["efficientnet_b0"]["regimes"]["4. Source Platt Scaling"]["recovery_pct"]
    mean_platt_rec = (rec_d121_platt + rec_eff_platt) / 2.0

    rows.append({
        "item": "Provisional Abstract Reference: Source Platt ECE Recovery",
        "value": f"{mean_platt_rec:.1f}% (DenseNet: {rec_d121_platt:.1f}%, EfficientNet: {rec_eff_platt:.1f}%)",
        "source file": f"results/tables/provisional_{run_id}/table2_calibration_regimes_and_oracle_recovery.json",
        "JSON key": "architectures.*.regimes['4. Source Platt Scaling'].recovery_pct",
        "data_source": "real",
        "how computed (script + line)": "scripts/run_task_b_pipeline.py:460-520 (regenerated from JSON; manuscript text unedited)",
    })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Successfully generated {output_csv} with {len(df)} real provenance entries.")

if __name__ == "__main__":
    build_provenance()
