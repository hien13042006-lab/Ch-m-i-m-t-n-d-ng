"""Demo tín dụng trên dự đoán OOF đã lưu. Run: python -m streamlit run streamlit_app.py."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
EVAL = "eval_results/"
COLORS = {"A": "#8795a5", "B": "#087f8c"}
GROUPS = {
    "ext_source_scores": "Điểm nguồn ngoài",
    "internal_installment_behavior": "Hành vi trả góp",
    "internal_pos_cash_behavior": "Hành vi POS / tiền mặt",
    "internal_credit_card_behavior": "Hành vi thẻ tín dụng",
    "internal_previous_app_history": "Hồ sơ vay trước",
    "bureau_delinquency_history": "Lịch sử quá hạn bureau",
    "bureau_credit_exposure": "Dư nợ và khoản vay bureau",
    "application_financial_ratios": "Tỷ lệ tài chính",
    "applicant_demographics_stability": "Thông tin cá nhân và ổn định",
    "other_factors": "Các yếu tố khác",
}
COHORTS = {
    "rehabilitated": "Đã phục hồi",
    "unrehabilitated": "Chưa phục hồi",
    "never_bad": "Chưa từng xấu đạt ngưỡng",
    "no_bureau_balance": "Không có lịch sử bureau theo tháng",
    "excluded_missing_data": "Thiếu dữ liệu đối chiếu",
}
PAGES = ["Tổng quan", "Hồ sơ khách hàng", "Điểm và hạng"]


@st.cache_data(show_spinner=False)
def read_file(name, modified_ns):
    path = ROOT / name
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def data(name):
    path = ROOT / name
    if not path.is_file():
        st.error(f"Chưa có file {name}. Hãy chạy Part tương ứng để tạo kết quả.")
        st.stop()
    return read_file(name, path.stat().st_mtime_ns)


def figure(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def download(df, filename):
    st.download_button("Tải bảng CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv")


def confidence(low, high, scale=1):
    return f"[{low * scale:.3f}; {high * scale:.3f}]"


def overview():
    st.title("Rủi ro rõ hơn. Quyết định có cơ sở hơn.")
    st.write("So sánh mô hình A dùng hồ sơ và bureau với mô hình B bổ sung hành vi trả nợ nội bộ.")
    metrics = data(EVAL + "metrics_auc_bootstrap.csv")
    row = metrics.loc[metrics["group"] == "full_population"].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Khách hàng", f"{int(row.n_customers):,}")
    cols[1].metric("Tỷ lệ xấu thực tế", f"{row.observed_bad_rate:.2%}")
    cols[2].metric("ROC-AUC · B", f"{row.auc_b:.4f}", f"{row.delta_auc:+.4f} so với A")
    cols[3].metric("Average Precision · B", f"{row.pr_auc_b:.4f}", f"{row.delta_pr_auc:+.4f} so với A")
    st.subheader("Đối chứng trên cùng khách hàng")
    comparison = pd.DataFrame([
        {"Chỉ số": label, "A": row[f"{key}_a"], "B": row[f"{key}_b"],
         "Chênh lệch B − A": row[delta],
         "CI 95% chênh lệch": confidence(row[f"{delta}_ci95_low"], row[f"{delta}_ci95_high"]),
         "Bằng chứng cải thiện": "Có" if row[f"{delta}_ci95_low"] > 0 else "Chưa đủ"}
        for label, key, delta in [("ROC-AUC", "auc", "delta_auc"),
                                  ("Average Precision (PR-AUC trong CSV)", "pr_auc", "delta_pr_auc")]
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.subheader("Dự đoán có sát thực tế?")
    cal = data("calibration_deciles_ab.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    maximum = max(cal.pred_risk_mean.max(), cal.observed_bad_rate.max()) * 1.08
    ax.plot([0, maximum], [0, maximum], "--", color="#adb5bd", label="Dự đoán = thực tế")
    for model, frame in cal.groupby("model", sort=True):
        tag = model[-1].upper()
        ax.vlines(frame.pred_risk_mean, frame.obs_ci95_low, frame.obs_ci95_high,
                  color=COLORS[tag], linewidth=2)
        ax.plot(frame.pred_risk_mean, frame.observed_bad_rate, "o-", color=COLORS[tag], label=f"Mô hình {tag}")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Xác suất dự đoán trung bình", ylabel="Tỷ lệ xấu thực tế")
    ax.legend()
    ax.grid(alpha=.15)
    figure(fig)
    passed = int(cal.within_20pct_limit.astype(str).str.lower().eq("true").sum())
    st.caption(f"Mỗi điểm là một decile. Thanh dọc: CI bootstrap 95% của tỷ lệ xấu. {passed}/{len(cal)} nhóm đạt sai số tương đối ≤20%; đây là tiêu chí dự án.")
    with st.expander("Xem số liệu calibration"):
        st.dataframe(cal, hide_index=True, width="stretch")
        download(cal, "calibration_deciles_ab.csv")


@st.cache_data(show_spinner=False)
def join_customers(oof, scores, cohorts, shap):
    for name, frame in [("OOF", oof), ("điểm", scores), ("cohort", cohorts), ("SHAP", shap)]:
        if frame.SK_ID_CURR.duplicated().any():
            raise ValueError(f"Dữ liệu {name} trùng SK_ID_CURR.")
    ids = set(oof.SK_ID_CURR)
    if ids != set(scores.SK_ID_CURR) or ids != set(cohorts.SK_ID_CURR):
        raise ValueError("Danh sách khách hàng OOF, điểm và cohort không khớp. Hãy chạy lại các Part theo thứ tự.")
    # Select only IDs explained by SHAP; do not use stale example JSON scores.
    return (shap[["SK_ID_CURR"]].merge(oof, on="SK_ID_CURR", validate="one_to_one")
            .merge(scores, on="SK_ID_CURR", validate="one_to_one")
            .merge(cohorts, on="SK_ID_CURR", validate="one_to_one"))


def customers():
    st.title("Hồ sơ khách hàng")
    st.write("Xem rủi ro, điểm tín dụng và những nhóm thông tin đóng góp vào điểm B.")
    shap = data("shap_points_by_group.parquet")
    joined = join_customers(data("oof_predictions.csv"), data("scored_customers.csv"),
                            data("frozen_cohort_labels_N12.csv"), shap)
    chosen = st.selectbox("Tình huống minh họa", ["Tất cả khách có SHAP", "Phục hồi · tăng ít nhất 40 điểm",
                         "Phục hồi · điểm gần như không đổi", "Phục hồi · xuống hạng", "Chưa phục hồi", "Chưa từng xấu đạt ngưỡng"])
    mask = joined.cohort_t2_N12.eq("rehabilitated")
    selections = {
        "Tất cả khách có SHAP": pd.Series(True, index=joined.index),
        "Phục hồi · tăng ít nhất 40 điểm": mask & joined.score_diff.ge(40),
        "Phục hồi · điểm gần như không đổi": mask & joined.score_diff.abs().le(5),
        "Phục hồi · xuống hạng": mask & joined.grade_b.gt(joined.grade_a),
        "Chưa phục hồi": joined.cohort_t2_N12.eq("unrehabilitated"),
        "Chưa từng xấu đạt ngưỡng": joined.cohort_t2_N12.eq("never_bad"),
    }
    subset = joined.loc[selections[chosen]].sort_values("score_diff", ascending=False)
    if subset.empty:
        st.info("Không có khách hàng phù hợp trong mẫu SHAP hiện tại.")
        return
    customer_id = st.selectbox("Mã khách hàng (có thể gõ để tìm)", subset.SK_ID_CURR.astype(int).tolist())
    row = subset.loc[subset.SK_ID_CURR == customer_id].iloc[0]
    st.caption(f"{COHORTS.get(row.cohort_t2_N12, row.cohort_t2_N12)} · cấu hình t=2 · {len(subset):,} hồ sơ phù hợp")
    a, b = st.columns(2)
    for col, tag in [(a, "a"), (b, "b")]:
        with col.container(border=True):
            st.subheader(f"Mô hình {tag.upper()}")
            st.metric("Điểm tín dụng", f"{row[f'score_{tag}']:.2f}")
            st.write(f"**Hạng {int(row[f'grade_{tag}'])}/10** · Rủi ro dự đoán **{row[f'pred_risk_model_{tag}']:.2%}**")
    st.metric("Thay đổi điểm B − A", f"{row.score_diff:+.2f}")
    st.caption("Điểm cao hơn: rủi ro thấp hơn. Hạng 1 tốt nhất, hạng 10 xấu nhất. Tăng điểm có thể chưa đủ để lên hạng.")
    contribution = shap.loc[shap.SK_ID_CURR == customer_id].iloc[0]
    params = data("scoring_params.json")
    pts = contribution[[c for c in shap.columns if c.startswith("pts_")]].astype(float)
    reconstructed = float(np.clip(contribution.base_score + pts.sum(), params["clip_min"], params["clip_max"]))
    stale = (abs(reconstructed - row.score_b) > 1 or
             abs(contribution.score_b - row.score_b) > 1e-6 or
             abs(contribution.score_a - row.score_a) > 1e-6 or
             contribution.cohort_t2_N12 != row.cohort_t2_N12)
    if stale:
        st.warning("SHAP không khớp điểm hoặc nhóm hiện tại. Hãy chạy lại Part 3 sau Part 2 để cập nhật giải thích.")
    else:
        st.subheader("Những yếu tố cộng / trừ điểm B")
        pts = pts.reindex(pts.abs().sort_values(ascending=False).index)
        labels = [GROUPS.get(c.removeprefix("pts_"), c) for c in pts.index]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, pts, color=[COLORS["B"] if v >= 0 else "#d45d56" for v in pts])
        ax.invert_yaxis()
        ax.axvline(0, color="#adb5bd", linewidth=1)
        ax.set_xlabel("Điểm đóng góp: dương tăng điểm, âm giảm điểm")
        figure(fig)
        st.caption(f"Điểm nền {contribution.base_score:.2f} + tổng đóng góp {pts.sum():+.2f}, sau chặn biên = {reconstructed:.2f}.")
    with st.expander("Đối chiếu nhãn thực tế (chỉ có trong dữ liệu lịch sử)"):
        st.write(f"TARGET = {int(row.TARGET)} — " + ("Có khó khăn thanh toán" if row.TARGET == 1 else "Không thuộc lớp khó khăn thanh toán"))
        st.caption("Nhãn thực tế được hiển thị để đối chiếu, không phải đầu vào để gán điểm cho riêng khách hàng.")
    download(pd.DataFrame([row]), f"customer_{customer_id}.csv")


def grades():
    st.title("Điểm và hạng tín dụng")
    table = data(EVAL + "grade_table_10_b.csv")
    params = data("scoring_params.json")
    c1, c2, c3 = st.columns(3)
    c1.metric("Thang điểm", "1 – 1001")
    c2.metric("PDO", str(params["pdo"]))
    c3.metric("PSI lớn nhất giữa các fold", f"{params['max_psi_b']:.5f}")
    st.caption("Ngưỡng phân hạng lấy từ phân vị điểm B, áp dụng chung cho A/B. PSI ở đây so sánh phân bố hạng giữa fold và toàn mẫu, không phải theo thời gian.")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(table.grade, table.bad_rate_a, "o-", color=COLORS["A"], label="A")
    ax.plot(table.grade, table.bad_rate, "s-", color=COLORS["B"], label="B")
    ax.set(xlabel="Hạng (1 tốt nhất → 10 xấu nhất)", ylabel="Tỷ lệ xấu thực tế", xticks=range(1, 11))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    ax.grid(alpha=.15)
    figure(fig)
    st.dataframe(table, hide_index=True, width="stretch")
    download(table, "grade_table_10_b.csv")
    st.subheader("Khách hàng phục hồi chuyển hạng thế nào?")
    t = st.selectbox("Cấu hình nhóm phục hồi", [2, 3, 4], key="grade_t")
    matrix = data(EVAL + f"grade_transition_matrix_t{t}.csv").set_index("grade_a")
    summary = data(EVAL + f"grade_transition_summary_t{t}.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=range(1, 11), yticklabels=range(1, 11),
           xlabel="Hạng B", ylabel="Hạng A")
    limit = matrix.to_numpy().max() / 2
    for i in range(10):
        for j in range(10):
            value = int(matrix.iloc[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8,
                        color="white" if value > limit else "#1e293b")
    fig.colorbar(im, ax=ax, label="Số khách hàng")
    figure(fig)
    st.caption(f"Tổng {int(matrix.to_numpy().sum()):,} khách hàng phục hồi. Hàng: A; cột: B. Cột nhỏ hơn hàng nghĩa là lên hạng.")
    translated = summary.copy()
    translated["direction"] = translated.direction.replace({"upgraded_better_grade": "Lên hạng", "unchanged_grade": "Giữ nguyên", "downgraded_worse_grade": "Xuống hạng"})
    st.dataframe(translated, hide_index=True, width="stretch")
    download(summary, f"grade_transition_summary_t{t}.csv")
    with st.expander("Tham số điểm và PSI"):
        st.json(params)


def main():
    st.set_page_config(page_title="Home Credit · Behavioral Scorecard", page_icon="◈", layout="wide")
    st.markdown("""<style>
    .stApp {background: #f6f8fb;}
    .block-container {max-width: 1200px; padding-top: 2.4rem;}
    h1, h2, h3 {color: #17334a;}
    [data-testid="stMetric"] {background:white; padding:18px; border:1px solid #e2e8f0; border-radius:12px;}
    [data-testid="stSidebar"] {background:#edf3f7;}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### ◈ HOME CREDIT")
        st.caption("BEHAVIORAL SCORECARD · NHÓM 12")
        page = st.radio("Khám phá", PAGES)
    try:
        {PAGES[0]: overview, PAGES[1]: customers, PAGES[2]: grades}[page]()
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"Dữ liệu đầu ra chưa nhất quán: {exc}")
        st.info("Chạy lại Part 1 → 2 → 3 → 4 với cùng bộ dữ liệu rồi tải lại trang.")


if __name__ == "__main__":
    main()
