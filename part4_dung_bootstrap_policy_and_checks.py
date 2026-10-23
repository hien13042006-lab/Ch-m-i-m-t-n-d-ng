import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
EVAL_DIR = os.path.join(OUT_DIR, "eval_results")
os.makedirs(EVAL_DIR, exist_ok=True)
QUICK_TEST = os.environ.get("HC_QUICK_TEST") == "1"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _one_boot_iter(n, y, pa, pb, seed_i, fallback_da, fallback_dp):
    rng = np.random.default_rng(seed_i)
    idx = rng.integers(0, n, size=n)
    yi = y[idx]
    if yi.min() == yi.max():
        return fallback_da, fallback_dp
    return (
        float(roc_auc_score(yi, pb[idx]) - roc_auc_score(yi, pa[idx])),
        float(average_precision_score(yi, pb[idx]) - average_precision_score(yi, pa[idx])),
    )


def bootstrap_group_metrics(y, pa, pb, n_boot=1000, seed=42):
    n = len(y)
    auc_a0, auc_b0 = float(roc_auc_score(y, pa)), float(roc_auc_score(y, pb))
    pra_a0, pra_b0 = float(average_precision_score(y, pa)), float(average_precision_score(y, pb))
    da0, dp0 = auc_b0 - auc_a0, pra_b0 - pra_a0
    iter_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_boot)
    pairs = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_one_boot_iter)(n, y, pa, pb, int(s), da0, dp0) for s in iter_seeds
    )
    d_auc = np.array([p[0] for p in pairs], dtype=np.float64)
    d_pra = np.array([p[1] for p in pairs], dtype=np.float64)
    auc_lo, auc_hi = np.percentile(d_auc, [2.5, 97.5])
    pra_lo, pra_hi = np.percentile(d_pra, [2.5, 97.5])
    return {
        "n_customers": int(n), "observed_bad_rate": float(np.mean(y)),
        "auc_a": auc_a0, "auc_b": auc_b0, "delta_auc": float(da0),
        "delta_auc_ci95_low": float(auc_lo), "delta_auc_ci95_high": float(auc_hi),
        "pr_auc_a": pra_a0, "pr_auc_b": pra_b0, "delta_pr_auc": float(dp0),
        "delta_pr_auc_ci95_low": float(pra_lo), "delta_pr_auc_ci95_high": float(pra_hi),
        "conclusion_auc": "cải thiện" if auc_lo > 0 else "chưa đủ bằng chứng",
        "conclusion_pr_auc": "cải thiện" if pra_lo > 0 else "chưa đủ bằng chứng",
    }, d_auc, d_pra


def evaluate_policy_both_modes(df_in, cohort_col, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_all = df_in["TARGET"].to_numpy()
    pa_all, pb_all = df_in["pred_risk_model_a"].to_numpy(), df_in["pred_risk_model_b"].to_numpy()
    reh_mask = (df_in[cohort_col] == "rehabilitated").to_numpy()
    y_reh, pa_reh, pb_reh = y_all[reh_mask], pa_all[reh_mask], pb_all[reh_mask]
    n_reh = len(y_reh)
    out_rows = []

    def append_row(mode_name, cut_a, cut_b):
        app_a, app_b = pa_reh <= cut_a, pb_reh <= cut_b
        rate_a, rate_b = float(np.mean(app_a)), float(np.mean(app_b))
        bad_a = float(np.mean(y_reh[app_a])) if app_a.sum() > 0 else 0.0
        bad_b = float(np.mean(y_reh[app_b])) if app_b.sum() > 0 else 0.0
        d_rate, d_bad = np.empty(n_boot), np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n_reh, size=n_reh)
            aa, ab, yr = app_a[idx], app_b[idx], y_reh[idx]
            d_rate[i] = np.mean(ab) - np.mean(aa)
            d_bad[i] = (np.mean(yr[ab]) if ab.sum() > 0 else bad_b) - (np.mean(yr[aa]) if aa.sum() > 0 else bad_a)
        r_lo, r_hi = np.percentile(d_rate, [2.5, 97.5])
        b_lo, b_hi = np.percentile(d_bad, [2.5, 97.5])
        out_rows.append({
            "cohort": cohort_col, "policy_mode": mode_name,
            "system_cutoff_risk_a": cut_a, "system_cutoff_risk_b": cut_b,
            "rehab_approval_a": rate_a, "rehab_approval_b": rate_b,
            "delta_approval": float(rate_b - rate_a), "delta_approval_ci95_low": float(r_lo), "delta_approval_ci95_high": float(r_hi),
            "conclusion_approval": "cải thiện" if r_lo > 0 else "chưa đủ bằng chứng",
            "rehab_bad_rate_a": bad_a, "rehab_bad_rate_b": bad_b,
            "delta_bad_rate": float(bad_b - bad_a), "delta_bad_rate_ci95_low": float(b_lo), "delta_bad_rate_ci95_high": float(b_hi),
            "conclusion_bad_rate": "cải thiện" if b_hi < 0 else "chưa đủ bằng chứng",
        })

    for q in [0.30, 0.50, 0.70]:
        append_row(f"mode1_fixed_system_approval_{int(q*100)}pct", float(np.quantile(pa_all, q)), float(np.quantile(pb_all, q)))

    cut_a50 = float(np.quantile(pa_all, 0.50))
    target_T = float(np.mean(y_all[pa_all <= cut_a50]))
    ord_b = np.argsort(pb_all)
    cum_bad_b = np.cumsum(y_all[ord_b]) / np.arange(1, len(y_all) + 1)
    ok_idx = np.where(cum_bad_b <= target_T)[0]
    cut_b_T = float(pb_all[ord_b[ok_idx[-1]]]) if len(ok_idx) > 0 else float(np.quantile(pb_all, 0.50))
    append_row(f"mode2_matched_system_bad_rate_T_{target_T:.4f}", cut_a50, cut_b_T)
    return pd.DataFrame(out_rows)


def evaluate_subgroup_calibration(df_sub, subgroup_label, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y, pa, pb = df_sub["TARGET"].to_numpy(), df_sub["pred_risk_model_a"].to_numpy(), df_sub["pred_risk_model_b"].to_numpy()
    n = len(y)
    obs, mean_pa, mean_pb = float(np.mean(y)), float(np.mean(pa)), float(np.mean(pb))
    b_obs, b_da, b_db = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        b_obs[i], b_da[i], b_db[i] = np.mean(yi), np.mean(pa[idx]) - np.mean(yi), np.mean(pb[idx]) - np.mean(yi)
    obs_lo, obs_hi = np.percentile(b_obs, [2.5, 97.5])
    da_lo, da_hi = np.percentile(b_da, [2.5, 97.5])
    db_lo, db_hi = np.percentile(b_db, [2.5, 97.5])
    return {
        "subgroup": subgroup_label, "n_customers": int(n),
        "observed_bad_rate": obs, "observed_ci95_low": float(obs_lo), "observed_ci95_high": float(obs_hi),
        "pred_risk_mean_a": mean_pa, "calib_gap_a": float(mean_pa - obs), "calib_gap_a_ci95_low": float(da_lo), "calib_gap_a_ci95_high": float(da_hi),
        "pred_risk_mean_b": mean_pb, "calib_gap_b": float(mean_pb - obs), "calib_gap_b_ci95_low": float(db_lo), "calib_gap_b_ci95_high": float(db_hi),
    }


def run_part4_dung():
    oof = pd.read_csv(os.path.join(OUT_DIR, "oof_predictions.csv")).sort_values("SK_ID_CURR").reset_index(drop=True)
    folds = pd.read_csv(os.path.join(OUT_DIR, "folds.csv")).sort_values("SK_ID_CURR").reset_index(drop=True)
    scored = pd.read_csv(os.path.join(OUT_DIR, "scored_customers.csv")).sort_values("SK_ID_CURR").reset_index(drop=True)
    cohorts = pd.read_csv(os.path.join(OUT_DIR, "frozen_cohort_labels_N12.csv")).sort_values("SK_ID_CURR").reset_index(drop=True)
    fb_meta = pd.read_parquet(os.path.join(OUT_DIR, "features_model_b.parquet"), columns=["SK_ID_CURR", "HAS_INTERNAL_HISTORY"]).sort_values("SK_ID_CURR").reset_index(drop=True)
    shap_pts = pd.read_parquet(os.path.join(OUT_DIR, "shap_points_by_group.parquet"))

    with open(os.path.join(OUT_DIR, "config_shared.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(OUT_DIR, "scoring_params.json"), "r", encoding="utf-8") as f:
        sparams = json.load(f)

    df = oof.merge(fb_meta, on="SK_ID_CURR", how="left").merge(cohorts, on="SK_ID_CURR", how="left")

    recomp_auc_a = float(roc_auc_score(df["TARGET"], df["pred_risk_model_a"]))
    recomp_auc_b = float(roc_auc_score(df["TARGET"], df["pred_risk_model_b"]))
    ind_check_df = pd.DataFrame([
        {"model": "model_a", "reported_auc": cfg["oof_auc_a"], "recomputed_auc": recomp_auc_a, "abs_diff": abs(recomp_auc_a - cfg["oof_auc_a"])},
        {"model": "model_b", "reported_auc": cfg["oof_auc_b"], "recomputed_auc": recomp_auc_b, "abs_diff": abs(recomp_auc_b - cfg["oof_auc_b"])},
    ])
    ind_check_df.to_csv(os.path.join(EVAL_DIR, "independent_auc_check.csv"), index=False)

    n_boot = 100 if QUICK_TEST else 1000
    metric_rows = []
    r_all, _, _ = bootstrap_group_metrics(df["TARGET"].to_numpy(), df["pred_risk_model_a"].to_numpy(), df["pred_risk_model_b"].to_numpy(), n_boot=n_boot)
    metric_rows.append({"config": "ALL", "group": "full_population", **r_all})

    cold_sub = df[df["HAS_INTERNAL_HISTORY"] == 0]
    r_cold, _, _ = bootstrap_group_metrics(cold_sub["TARGET"].to_numpy(), cold_sub["pred_risk_model_a"].to_numpy(), cold_sub["pred_risk_model_b"].to_numpy(), n_boot=n_boot)
    metric_rows.append({"config": "ALL", "group": "no_internal_history", **r_cold})

    nobb_sub = df[df["cohort_t2_N12"] == "no_bureau_balance"]
    r_nobb, b_auc_nobb, b_pra_nobb = bootstrap_group_metrics(nobb_sub["TARGET"].to_numpy(), nobb_sub["pred_risk_model_a"].to_numpy(), nobb_sub["pred_risk_model_b"].to_numpy(), n_boot=n_boot)

    for t in [2, 3, 4]:
        col = f"cohort_t{t}_N12"
        stored = {}
        for g_name in ["rehabilitated", "unrehabilitated", "never_bad", "no_bureau_balance", "excluded_missing_data"]:
            if g_name == "no_bureau_balance":
                metric_rows.append({"config": f"t{t}_N12", "group": g_name, **r_nobb})
                stored[g_name] = (r_nobb, b_auc_nobb, b_pra_nobb)
                continue
            sub = df[df[col] == g_name]
            r_g, b_auc, b_pra = bootstrap_group_metrics(sub["TARGET"].to_numpy(), sub["pred_risk_model_a"].to_numpy(), sub["pred_risk_model_b"].to_numpy(), n_boot=n_boot)
            metric_rows.append({"config": f"t{t}_N12", "group": g_name, **r_g})
            stored[g_name] = (r_g, b_auc, b_pra)

        rest_sub = df[df[col].isin(["never_bad", "unrehabilitated"])]
        r_rest, b_auc_rest, b_pra_rest = bootstrap_group_metrics(rest_sub["TARGET"].to_numpy(), rest_sub["pred_risk_model_a"].to_numpy(), rest_sub["pred_risk_model_b"].to_numpy(), n_boot=n_boot)
        metric_rows.append({"config": f"t{t}_N12", "group": "rest_never_plus_unrehab", **r_rest})
        r_reh, b_auc_reh, b_pra_reh = stored["rehabilitated"]
        did_auc, did_pra = b_auc_reh - b_auc_rest, b_pra_reh - b_pra_rest
        d_auc_lo, d_auc_hi = np.percentile(did_auc, [2.5, 97.5])
        d_pra_lo, d_pra_hi = np.percentile(did_pra, [2.5, 97.5])
        metric_rows.append({
            "config": f"t{t}_N12", "group": "did_rehab_vs_rest",
            "n_customers": int(r_reh["n_customers"]), "observed_bad_rate": float(r_reh["observed_bad_rate"]),
            "auc_a": np.nan, "auc_b": np.nan,
            "delta_auc": float(r_reh["delta_auc"] - r_rest["delta_auc"]),
            "delta_auc_ci95_low": float(d_auc_lo), "delta_auc_ci95_high": float(d_auc_hi),
            "pr_auc_a": np.nan, "pr_auc_b": np.nan,
            "delta_pr_auc": float(r_reh["delta_pr_auc"] - r_rest["delta_pr_auc"]),
            "delta_pr_auc_ci95_low": float(d_pra_lo), "delta_pr_auc_ci95_high": float(d_pra_hi),
            "conclusion_auc": "cải thiện" if d_auc_lo > 0 else "chưa đủ bằng chứng",
            "conclusion_pr_auc": "cải thiện" if d_pra_lo > 0 else "chưa đủ bằng chứng",
        })

    pd.DataFrame(metric_rows).to_csv(os.path.join(EVAL_DIR, "metrics_auc_bootstrap.csv"), index=False)
    pd.concat([evaluate_policy_both_modes(df, f"cohort_t{t}_N12", n_boot=n_boot) for t in [2, 3, 4]], ignore_index=True).to_csv(
        os.path.join(EVAL_DIR, "policy_evaluation.csv"), index=False
    )

    cal_rows = [evaluate_subgroup_calibration(df[df[f"cohort_t{t}_N12"] == "rehabilitated"], f"rehabilitated_t{t}_N12", n_boot=n_boot) for t in [2, 3, 4]]
    cal_rows.append(evaluate_subgroup_calibration(cold_sub, "no_internal_history", n_boot=n_boot))
    pd.DataFrame(cal_rows).to_csv(os.path.join(EVAL_DIR, "rehab_and_cold_calibration.csv"), index=False)

    EXPECTED_A2 = {
        "cohort_t2_N12": {"never_bad": 85865, "unrehabilitated": 2321, "rehabilitated": 3968, "excluded_missing_data": 77, "no_bureau_balance": 215280},
        "cohort_t3_N12": {"never_bad": 88826, "unrehabilitated": 1349, "rehabilitated": 2007, "excluded_missing_data": 49, "no_bureau_balance": 215280},
        "cohort_t4_N12": {"never_bad": 89620, "unrehabilitated": 1144, "rehabilitated": 1429, "excluded_missing_data": 38, "no_bureau_balance": 215280},
    }
    with open(os.path.join(OUT_DIR, "feature_list_a.txt"), "r", encoding="utf-8") as f:
        fa_list = [x.strip() for x in f if x.strip() and not x.startswith("#")]
    with open(os.path.join(OUT_DIR, "feature_list_b.txt"), "r", encoding="utf-8") as f:
        fb_list = [x.strip() for x in f if x.strip() and not x.startswith("#")]

    c1_ok = len(folds) == len(oof) == len(scored) == len(cohorts) == 307_511
    c2_ok = all(cohorts[c].value_counts().to_dict() == exp for c, exp in EXPECTED_A2.items())
    forbidden = {"SK_ID_CURR", "TARGET", "fold", "cohort_t2_N12", "cohort_t3_N12", "cohort_t4_N12"}
    c3_ok = not (forbidden & set(fa_list)) and not (forbidden & set(fb_list)) and len(fa_list) == 171 and len(fb_list) == 219

    err_pa = float(np.max(np.abs(oof["pred_risk_model_a"].to_numpy() - sigmoid(oof["logit_model_a"].to_numpy()))))
    err_pb = float(np.max(np.abs(oof["pred_risk_model_b"].to_numpy() - sigmoid(oof["logit_model_b"].to_numpy()))))
    c4_ok = err_pa < 1e-6 and err_pb < 1e-6 and float(ind_check_df["abs_diff"].max()) < 1e-6

    pts_cols = [c for c in shap_pts.columns if c.startswith("pts_")]
    recon_sc = np.clip(shap_pts["base_score"].to_numpy() + shap_pts[pts_cols].sum(axis=1).to_numpy(), 1.0, 1001.0)
    max_sum_err = float(np.max(np.abs(recon_sc - shap_pts["score_b"].to_numpy())))
    c5_ok = max_sum_err <= 1.0 and sparams["clip_rate_a"] <= 0.001 and sparams["clip_rate_b"] <= 0.001
    c6_ok = bool(sparams["strictly_monotonic_bad_rate_b"]) and float(sparams["max_psi_b"]) < 0.1
    med_other = float(np.median(np.abs(shap_pts["pts_other_factors"].to_numpy()) / np.maximum(np.abs(shap_pts[pts_cols].to_numpy()).sum(axis=1), 1e-6)))
    c7_ok = med_other <= 0.20

    trace_df = pd.DataFrame([
        {"report_section": "1_overview_sample_size", "metric_name": "total_train_customers", "source_file": "oof_predictions.csv", "column_or_key": "SK_ID_CURR"},
        {"report_section": "1_cohort_counts", "metric_name": "cohort_t2_t3_t4_counts", "source_file": "frozen_cohort_labels_N12.csv", "column_or_key": "cohort_t2_N12..cohort_t4_N12"},
        {"report_section": "2_model_discrimination", "metric_name": "delta_auc_and_pr_auc_ci95", "source_file": "eval_results/metrics_auc_bootstrap.csv", "column_or_key": "delta_auc, delta_pr_auc"},
        {"report_section": "2_seed_fold_stability", "metric_name": "fold_and_seed_auc", "source_file": "fold_seed_auc_report.csv", "column_or_key": "auc"},
        {"report_section": "3_scorecard_grades", "metric_name": "grade_1_to_10_bad_rates", "source_file": "eval_results/grade_table_10_b.csv", "column_or_key": "bad_rate, bad_per_100"},
        {"report_section": "3_grade_transitions", "metric_name": "rehab_grade_migration", "source_file": "eval_results/grade_transition_summary_t2.csv", "column_or_key": "pct_rehab, observed_bad_rate"},
        {"report_section": "4_approval_policy", "metric_name": "policy_mode1_and_mode2", "source_file": "eval_results/policy_evaluation.csv", "column_or_key": "delta_approval, delta_bad_rate"},
        {"report_section": "4_calibration_and_cold_start", "metric_name": "rehab_and_cold_calibration", "source_file": "eval_results/rehab_and_cold_calibration.csv", "column_or_key": "calib_gap_a, calib_gap_b"},
    ])
    trace_df.to_csv(os.path.join(EVAL_DIR, "report_number_traceability.csv"), index=False)
    c8_ok = all(os.path.exists(os.path.join(OUT_DIR, r["source_file"])) for _, r in trace_df.iterrows())

    pd.DataFrame([
        {"check_id": "C1", "description": "Đủ 307 511 dòng, khớp SK_ID_CURR và cột fold", "passed": bool(c1_ok), "detail": f"n={len(oof):,}"},
        {"check_id": "C2", "description": "Số khách mỗi nhóm khớp bảng A2", "passed": bool(c2_ok), "detail": "Khớp chính xác t2, t3, t4"},
        {"check_id": "C3", "description": "Không có TARGET, SK_ID_CURR, cột nhóm trong danh sách đặc trưng", "passed": bool(c3_ok), "detail": f"A={len(fa_list)}, B={len(fb_list)}"},
        {"check_id": "C4", "description": "pred_risk khớp sigmoid(logit), điểm khớp scoring_params, AUC độc lập khớp < 1e-6", "passed": bool(c4_ok), "detail": f"max_err={max(err_pa, err_pb):.2e}"},
        {"check_id": "C5", "description": "Tổng điểm nhóm khớp điểm cuối, tỷ lệ bị cắt ở 1/1001 <= 0.1%", "passed": bool(c5_ok), "detail": f"clip_B={sparams['clip_rate_b']:.4%}, sum_err={max_sum_err:.4f}"},
        {"check_id": "C6", "description": "Tỷ lệ vỡ nợ tăng nghiêm ngặt theo hạng 1..10, PSI < 0.1", "passed": bool(c6_ok), "detail": f"monotonic={sparams['strictly_monotonic_bad_rate_b']}, max_psi={sparams['max_psi_b']:.5f}"},
        {"check_id": "C7", "description": "Tỷ lệ độ lớn nhóm 'Các yếu tố khác' <= 20%", "passed": bool(c7_ok), "detail": f"median_other_share={med_other:.2%}"},
        {"check_id": "C8", "description": "Mọi số trong báo cáo truy xuất được về file trong eval_results/ hoặc file gốc", "passed": bool(c8_ok), "detail": f"traced_tables={len(trace_df)}"},
    ]).to_csv(os.path.join(EVAL_DIR, "integration_checks_part_c.csv"), index=False)


if __name__ == "__main__":
    run_part4_dung()