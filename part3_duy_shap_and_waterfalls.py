import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
EVAL_DIR = os.path.join(OUT_DIR, "eval_results")
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(EVAL_DIR, exist_ok=True)


def assign_feature_group(col):
    if col.startswith("EXT_SOURCE"):
        return ("ext_source_scores", "Điểm tín dụng từ nguồn dữ liệu bên ngoài")
    if col.startswith("INS_"):
        return ("internal_installment_behavior", "Kỷ luật trả góp nội bộ (trả đúng hạn, trước hạn)")
    if col.startswith("POS_"):
        return ("internal_pos_cash_behavior", "Lịch sử vay trả góp POS và vay tiền mặt nội bộ")
    if col.startswith("CC_"):
        return ("internal_credit_card_behavior", "Mức sử dụng hạn mức và trả nợ thẻ tín dụng")
    if col.startswith("PREV_") or col == "HAS_INTERNAL_HISTORY":
        return ("internal_previous_app_history", "Lịch sử phê duyệt các đơn đăng ký vay trước đây")
    if col.startswith("B_BB_") or "OVERDUE" in col or "DPD" in col:
        return ("bureau_delinquency_history", "Lịch sử quá hạn tại các tổ chức tín dụng khác")
    if col.startswith("B_") or col in ("HAS_BUREAU_RECORD", "CROSS_TOTAL_DEBT_TO_INCOME"):
        return ("bureau_credit_exposure", "Quy mô dư nợ và khoản vay đang mở tại Bureau")
    if any(k in col for k in ("INCOME", "CREDIT", "ANNUITY", "GOODS")):
        return ("application_financial_ratios", "Tỷ lệ thu nhập, khoản vay và gánh nặng trả góp")
    if any(k in col for k in ("DAYS_", "HOUR_", "NAME_", "CODE_", "FLAG_", "CNT_", "OWN_", "OCCUPATION", "ORGANIZATION", "REG_", "LIVE_")):
        return ("applicant_demographics_stability", "Độ tuổi, thâm niên làm việc và thông tin cá nhân")
    return ("other_factors", "Các yếu tố bổ sung về nơi ở và hồ sơ")


def plot_customer_waterfall(card, out_png, out_csv):
    grp_items = sorted(card["group_points_b"].items(), key=lambda x: abs(x[1]), reverse=True)
    wf_df = pd.DataFrame(grp_items, columns=["group_code", "points"])
    wf_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    y_pos = np.arange(len(wf_df))
    pts = wf_df["points"].to_numpy()
    colors = ["#1F77B4" if v >= 0 else "#D62728" for v in pts]
    ax.barh(y_pos, pts, color=colors, height=0.6)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wf_df["group_code"])
    ax.invert_yaxis()
    for i, v in enumerate(pts):
        ax.text(v + (1.5 if v >= 0 else -1.5), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_xlabel("Điểm cộng (+) / Điểm trừ (-) trên thang 1 - 1 001")
    ax.set_title(
        f"Giải thích điểm SHAP: SK_ID_CURR = {card['sk_id_curr']} ({card['case_type']})\n"
        f"Điểm A = {card['score_a']} (Hạng {card['grade_a']}) -> Điểm B = {card['score_b']} (Hạng {card['grade_b']}, Khởi điểm = {card['base_score_b']})",
        fontsize=10.5,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def run_part3_duy():
    oof = pd.read_csv(os.path.join(OUT_DIR, "oof_predictions.csv")).sort_values("SK_ID_CURR").reset_index(drop=True)
    scored = pd.read_csv(os.path.join(OUT_DIR, "scored_customers.csv")).sort_values("SK_ID_CURR").reset_index(drop=True)
    cohorts = pd.read_csv(os.path.join(OUT_DIR, "frozen_cohort_labels_N12.csv")).sort_values("SK_ID_CURR").reset_index(drop=True)
    fb = pd.read_parquet(os.path.join(OUT_DIR, "features_model_b.parquet")).sort_values("SK_ID_CURR").reset_index(drop=True)

    with open(os.path.join(OUT_DIR, "feature_list_b.txt"), "r", encoding="utf-8") as f:
        feats_b = [x.strip() for x in f if x.strip() and not x.startswith("#")]
    with open(os.path.join(OUT_DIR, "scoring_params.json"), "r", encoding="utf-8") as f:
        sparams = json.load(f)

    factor, anchor_logit = float(sparams["factor"]), float(sparams["anchor_logit"])

    fmap_rows = [{"feature": c, "group_code": assign_feature_group(c)[0], "group_description_vi": assign_feature_group(c)[1]} for c in feats_b]
    fmap_df = pd.DataFrame(fmap_rows)
    fmap_df.to_csv(os.path.join(OUT_DIR, "feature_group_mapping.csv"), index=False)

    rehab_m = (cohorts["cohort_t2_N12"] == "rehabilitated").to_numpy()
    unrehab_m = (cohorts["cohort_t2_N12"] == "unrehabilitated").to_numpy()
    never_m = (cohorts["cohort_t2_N12"] == "never_bad").to_numpy()
    rng = np.random.default_rng(42)
    eval_idx = np.unique(np.concatenate([
        np.where(rehab_m)[0],
        rng.choice(np.where(unrehab_m)[0], size=min(1000, int(unrehab_m.sum())), replace=False),
        rng.choice(np.where(never_m)[0], size=min(2000, int(never_m.sum())), replace=False),
    ]))

    pd.DataFrame({"SK_ID_CURR": oof.loc[eval_idx, "SK_ID_CURR"].to_numpy()}).to_csv(
        os.path.join(EVAL_DIR, "shap_sample_customer_ids.csv"), index=False
    )

    sub_df = fb.iloc[eval_idx].copy().reset_index(drop=True)
    sub_folds = oof.loc[eval_idx, "fold"].to_numpy()
    shap_mat = np.zeros((len(eval_idx), len(feats_b) + 1), dtype=np.float64)

    for f_idx in range(5):
        f_pos = np.where(sub_folds == f_idx)[0]
        if len(f_pos) == 0:
            continue
        x_part = sub_df.loc[f_pos, feats_b]
        m_files = sorted([os.path.join(MODEL_DIR, fn) for fn in os.listdir(MODEL_DIR) if fn.startswith(f"model_b_fold{f_idx}_seed")])
        fold_c = np.zeros((len(f_pos), len(feats_b) + 1), dtype=np.float64)
        for mf in m_files:
            bst = lgb.Booster(model_file=mf)
            fold_c += bst.predict(x_part, pred_contrib=True)
        shap_mat[f_pos] = fold_c / len(m_files)

    base_scores = 706.0 - factor * (shap_mat[:, -1] - anchor_logit)
    pts_mat = -factor * shap_mat[:, :-1]
    group_codes = sorted(fmap_df["group_code"].unique())

    shap_points_df = pd.DataFrame({
        "SK_ID_CURR": sub_df["SK_ID_CURR"].to_numpy(),
        "cohort_t2_N12": cohorts.loc[eval_idx, "cohort_t2_N12"].to_numpy(),
        "base_score": base_scores,
        "score_a": scored.loc[eval_idx, "score_a"].to_numpy(),
        "score_b": scored.loc[eval_idx, "score_b"].to_numpy(),
    })
    for g_code in group_codes:
        g_idx = [i for i in range(len(feats_b)) if fmap_rows[i]["group_code"] == g_code]
        shap_points_df[f"pts_{g_code}"] = pts_mat[:, g_idx].sum(axis=1)

    shap_points_df.to_parquet(os.path.join(OUT_DIR, "shap_points_by_group.parquet"), index=False)
    pts_cols = [f"pts_{g}" for g in group_codes]
    shap_points_df.groupby("cohort_t2_N12")[pts_cols + ["base_score", "score_a", "score_b"]].mean().reset_index().to_csv(
        os.path.join(EVAL_DIR, "shap_group_points_by_cohort.csv"), index=False
    )

    rehab_up = shap_points_df[(shap_points_df["cohort_t2_N12"] == "rehabilitated") & ((shap_points_df["score_b"] - shap_points_df["score_a"]) >= 40.0)].sort_values("score_b", ascending=False)
    rehab_same = shap_points_df[(shap_points_df["cohort_t2_N12"] == "rehabilitated") & ((shap_points_df["score_b"] - shap_points_df["score_a"]).abs() <= 5.0)]
    unrehab_ex = shap_points_df[shap_points_df["cohort_t2_N12"] == "unrehabilitated"].sort_values("score_b", ascending=True)
    never_ex = shap_points_df[shap_points_df["cohort_t2_N12"] == "never_bad"].sort_values("score_b", ascending=False)

    demo_rows = []
    for case_label, sub_table in [
        ("rehabilitated_upgraded", rehab_up),
        ("rehabilitated_unchanged", rehab_same),
        ("unrehabilitated_high_risk", unrehab_ex),
        ("never_bad_clean_history", never_ex),
    ]:
        if len(sub_table) > 0:
            r0 = sub_table.iloc[0]
            sk_id = int(r0["SK_ID_CURR"])
            oof_row = oof[oof["SK_ID_CURR"] == sk_id].iloc[0]
            sc_row = scored[scored["SK_ID_CURR"] == sk_id].iloc[0]
            card = {
                "case_type": case_label, "sk_id_curr": sk_id, "target_actual": int(oof_row["TARGET"]),
                "score_a": round(float(sc_row["score_a"]), 2), "grade_a": int(sc_row["grade_a"]),
                "pred_risk_a_pct": round(float(oof_row["pred_risk_model_a"] * 100.0), 2),
                "score_b": round(float(sc_row["score_b"]), 2), "grade_b": int(sc_row["grade_b"]),
                "pred_risk_b_pct": round(float(oof_row["pred_risk_model_b"] * 100.0), 2),
                "score_diff": round(float(sc_row["score_diff"]), 2),
                "base_score_b": round(float(r0["base_score"]), 2),
                "group_points_b": {g: round(float(r0[f"pts_{g}"]), 2) for g in group_codes},
            }
            demo_rows.append(card)
            plot_customer_waterfall(
                card,
                os.path.join(EVAL_DIR, f"waterfall_{case_label}.png"),
                os.path.join(EVAL_DIR, f"waterfall_{case_label}.csv"),
            )

    with open(os.path.join(EVAL_DIR, "demo_customer_scorecards.json"), "w", encoding="utf-8") as f:
        json.dump(demo_rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_part3_duy()