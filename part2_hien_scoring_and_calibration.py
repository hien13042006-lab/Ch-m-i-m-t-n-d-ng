import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
EVAL_DIR = os.path.join(OUT_DIR, "eval_results")
os.makedirs(EVAL_DIR, exist_ok=True)
QUICK_TEST = os.environ.get("HC_QUICK_TEST") == "1"


def build_decile_calibration(y_true, y_prob, model_name, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    order = pd.Series(y_prob).rank(method="first").to_numpy()
    deciles = pd.qcut(order, 10, labels=False) + 1
    rows = []
    for d in range(1, 11):
        m = deciles == d
        yt, yp = y_true[m], y_prob[m]
        n = len(yt)
        p_mean, obs = float(np.mean(yp)), float(np.mean(yt))
        rel_err = abs(p_mean - obs) / max(obs, 1e-8)
        b_obs = np.array([np.mean(yt[rng.integers(0, n, size=n)]) for _ in range(n_boot)], dtype=np.float64)
        lo, hi = np.percentile(b_obs, [2.5, 97.5])
        rows.append({
            "model": model_name, "decile": int(d), "n_customers": int(n),
            "pred_risk_mean": p_mean, "observed_bad_rate": obs,
            "obs_ci95_low": float(lo), "obs_ci95_high": float(hi),
            "relative_error_pct": float(rel_err * 100.0),
            "within_20pct_limit": bool(rel_err <= 0.20),
        })
    return pd.DataFrame(rows)


def run_part2_hien():
    oof = pd.read_csv(os.path.join(OUT_DIR, "oof_predictions.csv")).sort_values("SK_ID_CURR").reset_index(drop=True)
    cohorts = pd.read_csv(os.path.join(OUT_DIR, "frozen_cohort_labels_N12.csv")).sort_values("SK_ID_CURR").reset_index(drop=True)

    n_boot = 100 if QUICK_TEST else 1000
    cal_a = build_decile_calibration(oof["TARGET"].to_numpy(), oof["pred_risk_model_a"].to_numpy(), "model_a", n_boot=n_boot)
    cal_b = build_decile_calibration(oof["TARGET"].to_numpy(), oof["pred_risk_model_b"].to_numpy(), "model_b", n_boot=n_boot)
    cal_ab = pd.concat([cal_a, cal_b], ignore_index=True)
    cal_ab.to_csv(os.path.join(OUT_DIR, "calibration_deciles_ab.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    max_v = float(max(cal_ab["pred_risk_mean"].max(), cal_ab["observed_bad_rate"].max()) * 1.05)
    ax.plot([0, max_v], [0, max_v], "--", color="#333333", linewidth=1.2, label="Đường chuẩn (y = x)")
    ax.errorbar(
        cal_a["pred_risk_mean"], cal_a["observed_bad_rate"],
        yerr=[cal_a["observed_bad_rate"] - cal_a["obs_ci95_low"], cal_a["obs_ci95_high"] - cal_a["observed_bad_rate"]],
        fmt="o-", color="#7F7F7F", capsize=3, linewidth=1.8, label="Mô hình nền (A)",
    )
    ax.errorbar(
        cal_b["pred_risk_mean"], cal_b["observed_bad_rate"],
        yerr=[cal_b["observed_bad_rate"] - cal_b["obs_ci95_low"], cal_b["obs_ci95_high"] - cal_b["observed_bad_rate"]],
        fmt="s-", color="#1F77B4", capsize=3, linewidth=1.8, label="Mô hình mở rộng (B)",
    )
    ax.set_title(f"Đánh giá độ hiệu chuẩn theo thập phân vị: Mô hình nền (A) và Mô hình mở rộng (B)\n(n = {len(oof):,})", fontsize=11)
    ax.set_xlabel("Xác suất dự đoán trung bình")
    ax.set_ylabel("Tỷ lệ khó khăn thanh toán thực tế (CI 95%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "calibration_deciles_ab.png"), dpi=300)
    plt.close(fig)

    p_bar = float(oof["TARGET"].mean())
    anchor_logit = float(np.log(p_bar / (1.0 - p_bar)))
    logit_a, logit_b = oof["logit_model_a"].to_numpy(), oof["logit_model_b"].to_numpy()

    chosen_pdo = 40
    for pdo in sorted([40, 50, 60, 80, 100], reverse=True):
        fac = pdo / np.log(2.0)
        ra = 706.0 - fac * (logit_a - anchor_logit)
        rb = 706.0 - fac * (logit_b - anchor_logit)
        if np.mean((ra <= 1.0) | (ra >= 1001.0)) <= 0.001 and np.mean((rb <= 1.0) | (rb >= 1001.0)) <= 0.001:
            chosen_pdo = int(pdo)
            break

    factor = float(chosen_pdo / np.log(2.0))
    offset = float(706.0 + factor * anchor_logit)
    score_a = np.clip(706.0 - factor * (logit_a - anchor_logit), 1.0, 1001.0)
    score_b = np.clip(706.0 - factor * (logit_b - anchor_logit), 1.0, 1001.0)

    quantiles = np.quantile(score_b, np.linspace(0.0, 1.0, 11))
    internal_cuts = quantiles[1:-1]

    def assign_grade(scores):
        return (10 - np.digitize(scores, internal_cuts, right=False)).astype(int)

    grade_a, grade_b = assign_grade(score_a), assign_grade(score_b)
    grade_of_706 = int(assign_grade(np.array([706.0]))[0])
    grade_labels = {1: "rất tốt", 2: "rất tốt", 3: "tốt", 4: "tốt", 5: "trung bình", 6: "trung bình", 7: "dưới trung bình", 8: "dưới trung bình", 9: "xấu", 10: "xấu"}

    scored_df = pd.DataFrame({
        "SK_ID_CURR": oof["SK_ID_CURR"].to_numpy(),
        "score_a": score_a, "score_b": score_b,
        "grade_a": grade_a, "grade_b": grade_b,
        "score_diff": score_b - score_a,
    }).sort_values("SK_ID_CURR").reset_index(drop=True)
    scored_df.to_csv(os.path.join(OUT_DIR, "scored_customers.csv"), index=False)

    grade_rows = []
    for g in range(1, 11):
        mb = grade_b == g
        ma = grade_a == g
        low_b = float(quantiles[10 - g]) if g < 10 else 1.0
        high_b = float(quantiles[11 - g]) if g > 1 else 1001.0
        bad_b = float(oof.loc[mb, "TARGET"].mean())
        bad_a = float(oof.loc[ma, "TARGET"].mean()) if ma.sum() > 0 else np.nan
        grade_rows.append({
            "grade": int(g), "label": grade_labels[g],
            "score_min": round(low_b, 2), "score_max": round(high_b, 2),
            "n_customers": int(mb.sum()), "pct_customers": float(mb.mean() * 100.0),
            "bad_rate": bad_b, "bad_rate_a": bad_a, "bad_per_100": round(bad_b * 100.0, 2),
        })
    grade_table_df = pd.DataFrame(grade_rows)
    grade_table_df.to_csv(os.path.join(EVAL_DIR, "grade_table_10_b.csv"), index=False)

    expected_prop = np.array([np.mean(grade_b == g) for g in range(1, 11)])
    psi_by_fold = {}
    for f_idx in range(5):
        fg = grade_b[(oof["fold"] == f_idx).to_numpy()]
        act_prop = np.clip(np.array([np.mean(fg == g) for g in range(1, 11)]), 1e-6, 1.0)
        psi_by_fold[f"fold_{f_idx}"] = float(np.sum((act_prop - expected_prop) * np.log(act_prop / expected_prop)))

    scoring_params = {
        "pdo": chosen_pdo, "factor": factor, "offset": offset,
        "anchor_bad_rate": p_bar, "anchor_logit": anchor_logit, "anchor_score": 706.0,
        "clip_min": 1.0, "clip_max": 1001.0,
        "clip_rate_a": float(np.mean((score_a <= 1.0) | (score_a >= 1001.0))),
        "clip_rate_b": float(np.mean((score_b <= 1.0) | (score_b >= 1001.0))),
        "decile_cutoffs": [float(x) for x in quantiles],
        "internal_cutoffs": [float(x) for x in internal_cuts],
        "grade_labels": {str(k): v for k, v in grade_labels.items()},
        "grade_of_score_706": grade_of_706,
        "psi_by_fold_b": psi_by_fold,
        "max_psi_b": float(max(psi_by_fold.values())),
        "strictly_monotonic_bad_rate_b": bool(np.all(np.diff(grade_table_df["bad_rate"].to_numpy()) > 0)),
    }
    with open(os.path.join(OUT_DIR, "scoring_params.json"), "w", encoding="utf-8") as f:
        json.dump(scoring_params, f, indent=2, ensure_ascii=False)

    for t in [2, 3, 4]:
        reh_ids = set(cohorts.loc[cohorts[f"cohort_t{t}_N12"] == "rehabilitated", "SK_ID_CURR"])
        sc_reh = scored_df[scored_df["SK_ID_CURR"].isin(reh_ids)].merge(oof[["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR")
        mat = pd.crosstab(sc_reh["grade_a"], sc_reh["grade_b"]).reindex(index=range(1, 11), columns=range(1, 11), fill_value=0)
        mat.to_csv(os.path.join(EVAL_DIR, f"grade_transition_matrix_t{t}.csv"))

        up_m, same_m, down_m = sc_reh["grade_b"] < sc_reh["grade_a"], sc_reh["grade_b"] == sc_reh["grade_a"], sc_reh["grade_b"] > sc_reh["grade_a"]
        pd.DataFrame([
            {"direction": "upgraded_better_grade", "n_customers": int(up_m.sum()), "pct_rehab": float(up_m.mean() * 100.0), "observed_bad_rate": float(sc_reh.loc[up_m, "TARGET"].mean()) if up_m.sum() > 0 else np.nan},
            {"direction": "unchanged_grade", "n_customers": int(same_m.sum()), "pct_rehab": float(same_m.mean() * 100.0), "observed_bad_rate": float(sc_reh.loc[same_m, "TARGET"].mean()) if same_m.sum() > 0 else np.nan},
            {"direction": "downgraded_worse_grade", "n_customers": int(down_m.sum()), "pct_rehab": float(down_m.mean() * 100.0), "observed_bad_rate": float(sc_reh.loc[down_m, "TARGET"].mean()) if down_m.sum() > 0 else np.nan},
        ]).to_csv(os.path.join(EVAL_DIR, f"grade_transition_summary_t{t}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.axis("off")
    tbl_data = [
        [f"Hạng {r['grade']}", r["label"], f"{r['score_min']:.0f} - {r['score_max']:.0f}", f"{r['n_customers']:,}", f"{r['pct_customers']:.1f}%", f"{r['bad_per_100']:.2f}%"]
        for _, r in grade_table_df.iterrows()
    ]
    table = ax.table(
        cellText=tbl_data,
        colLabels=["Hạng", "Nhãn", "Khoảng điểm", "Số khách", "% Khách", "Khó khăn TT / 100 người"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title(f"Bảng 10 hạng tín dụng của Mô hình mở rộng (B) (n = {len(oof):,})", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "grade_table_10_b.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(1, 11)
    ax.bar(x - 0.18, grade_table_df["bad_rate_a"] * 100.0, width=0.35, color="#7F7F7F", label="Mô hình nền (A)")
    ax.bar(x + 0.18, grade_table_df["bad_rate"] * 100.0, width=0.35, color="#1F77B4", label="Mô hình mở rộng (B)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{g}" for g in x])
    ax.set_xlabel("Hạng tín dụng (1 = Rất tốt, 10 = Xấu)")
    ax.set_ylabel("Tỷ lệ khó khăn thanh toán thực tế (%)")
    ax.set_title(f"Tỷ lệ khó khăn thanh toán theo 10 hạng tín dụng (n = {len(oof):,})", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "bad_rate_by_grade_ab.png"), dpi=300)
    plt.close(fig)

    mat_t2 = pd.read_csv(os.path.join(EVAL_DIR, "grade_transition_matrix_t2.csv"), index_col=0)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    im = ax.imshow(mat_t2.to_numpy(), cmap="Blues")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels([f"B-{g}" for g in range(1, 11)])
    ax.set_yticklabels([f"A-{g}" for g in range(1, 11)])
    ax.set_xlabel("Hạng theo Mô hình mở rộng (B)")
    ax.set_ylabel("Hạng theo Mô hình nền (A)")
    ax.set_title(f"Ma trận chuyển hạng nhóm đã phục hồi (t=2, n = {int(mat_t2.to_numpy().sum()):,})", fontsize=11)
    for i in range(10):
        for j in range(10):
            val = int(mat_t2.iloc[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", color="white" if val > mat_t2.to_numpy().max() * 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "grade_transition_matrix_t2.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    run_part2_hien()