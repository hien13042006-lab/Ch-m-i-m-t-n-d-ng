import os
import gc
import json
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
QUICK_TEST = os.environ.get("HC_QUICK_TEST") == "1"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def find_data_dir():
    for r in [os.environ.get("HC_DATA_DIR"), "/kaggle/input", "."]:
        if not r or not os.path.isdir(r):
            continue
        for root, _, files in os.walk(r):
            if "application_train.csv" in files:
                return root
    raise FileNotFoundError("application_train.csv not found")


def reduce_mem(df):
    for col in df.columns:
        t = df[col].dtype
        if str(t) != "category" and t != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(t)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df


def build_features_a(data_dir):
    bb = pd.read_csv(
        f"{data_dir}/bureau_balance.csv",
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
        dtype={"SK_ID_BUREAU": np.int32, "MONTHS_BALANCE": np.int8, "STATUS": "category"},
    )
    s_map = {"C": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "X": -1}
    bb["dpd_level"] = bb["STATUS"].map(s_map).astype(np.int8)
    for s in ["0", "1", "2", "3", "4", "5", "C", "X"]:
        bb[f"STATUS_{s}"] = (bb["STATUS"] == s).astype(np.int8)
    bb["is_any_dpd"] = (bb["dpd_level"] >= 1).astype(np.int8)
    bb["is_bad_31plus"] = (bb["dpd_level"] >= 2).astype(np.int8)
    bb["is_bad_91plus"] = (bb["dpd_level"] >= 4).astype(np.int8)

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        BB_DPD_MAX=("dpd_level", "max"),
        BB_ANY_DPD_COUNT=("is_any_dpd", "sum"),
        BB_BAD_31PLUS_COUNT=("is_bad_31plus", "sum"),
        BB_BAD_91PLUS_COUNT=("is_bad_91plus", "sum"),
        BB_STATUS_0_RATIO=("STATUS_0", "mean"),
        BB_STATUS_C_RATIO=("STATUS_C", "mean"),
        BB_STATUS_1_RATIO=("STATUS_1", "mean"),
        BB_STATUS_2_RATIO=("STATUS_2", "mean"),
    ).reset_index()
    bb_agg = reduce_mem(bb_agg)
    del bb
    gc.collect()

    bur = pd.read_csv(f"{data_dir}/bureau.csv")
    bur["SK_ID_CURR"] = bur["SK_ID_CURR"].astype(np.int32)
    bur["SK_ID_BUREAU"] = bur["SK_ID_BUREAU"].astype(np.int32)
    bur = bur.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    del bb_agg
    gc.collect()

    bur["is_active"] = (bur["CREDIT_ACTIVE"] == "Active").astype(np.int8)
    bur["is_closed"] = (bur["CREDIT_ACTIVE"] == "Closed").astype(np.int8)
    bur["debt_to_credit_ratio"] = bur["AMT_CREDIT_SUM_DEBT"] / (bur["AMT_CREDIT_SUM"] + 1e-5)
    bur["overdue_to_credit_ratio"] = bur["AMT_CREDIT_SUM_OVERDUE"] / (bur["AMT_CREDIT_SUM"] + 1e-5)

    bur_agg_dict = {
        "SK_ID_BUREAU": ["count"],
        "DAYS_CREDIT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["sum", "max", "mean"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
        "AMT_ANNUITY": ["sum", "max"],
        "is_active": ["sum", "mean"],
        "is_closed": ["sum"],
        "debt_to_credit_ratio": ["mean", "max"],
        "overdue_to_credit_ratio": ["mean", "max"],
        "BB_MONTHS_COUNT": ["sum", "mean"],
        "BB_MONTHS_MIN": ["min"],
        "BB_DPD_MAX": ["max"],
        "BB_ANY_DPD_COUNT": ["sum", "max"],
        "BB_BAD_31PLUS_COUNT": ["sum", "max"],
        "BB_BAD_91PLUS_COUNT": ["sum", "max"],
        "BB_STATUS_0_RATIO": ["mean"],
        "BB_STATUS_C_RATIO": ["mean"],
    }
    bur_client = bur.groupby("SK_ID_CURR").agg(bur_agg_dict)
    bur_client.columns = pd.Index([f"B_{c[0]}_{c[1].upper()}" for c in bur_client.columns.tolist()])
    bur_client.reset_index(inplace=True)
    bur_client["B_TOTAL_DEBT_TO_TOTAL_CREDIT"] = bur_client["B_AMT_CREDIT_SUM_DEBT_SUM"] / (bur_client["B_AMT_CREDIT_SUM_SUM"] + 1e-5)
    bur_client["B_ACTIVE_LOANS_RATIO"] = bur_client["B_is_active_SUM"] / (bur_client["B_SK_ID_BUREAU_COUNT"] + 1e-5)
    bur_client = reduce_mem(bur_client)
    del bur
    gc.collect()

    app = pd.read_csv(f"{data_dir}/application_train.csv")
    app["RATIO_ANNUITY_TO_INCOME"] = app["AMT_ANNUITY"] / (app["AMT_INCOME_TOTAL"] + 1e-5)
    app["RATIO_CREDIT_TO_INCOME"] = app["AMT_CREDIT"] / (app["AMT_INCOME_TOTAL"] + 1e-5)
    app["RATIO_CREDIT_TO_ANNUITY"] = app["AMT_CREDIT"] / (app["AMT_ANNUITY"] + 1e-5)
    app["RATIO_GOODS_TO_CREDIT"] = app["AMT_GOODS_PRICE"] / (app["AMT_CREDIT"] + 1e-5)
    app["RATIO_INCOME_PER_FAMILY_MEMBER"] = app["AMT_INCOME_TOTAL"] / (app["CNT_FAM_MEMBERS"] + 1e-5)
    app["DAYS_EMPLOYED_ANOM"] = (app["DAYS_EMPLOYED"] == 365243).astype(np.int8)
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)
    app["RATIO_EMPLOYED_TO_BIRTH"] = app["DAYS_EMPLOYED"] / (app["DAYS_BIRTH"] + 1e-5)

    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    app["EXT_SOURCES_MEAN"] = app[ext].mean(axis=1)
    app["EXT_SOURCES_STD"] = app[ext].std(axis=1)
    app["EXT_SOURCES_MIN"] = app[ext].min(axis=1)
    app["EXT_SOURCES_MAX"] = app[ext].max(axis=1)

    fa = app.merge(bur_client, on="SK_ID_CURR", how="left")
    del app, bur_client
    gc.collect()

    fa["HAS_BUREAU_RECORD"] = fa["B_SK_ID_BUREAU_COUNT"].notnull().astype(np.int8)
    fa["CROSS_TOTAL_DEBT_TO_INCOME"] = (fa["AMT_CREDIT"] + fa["B_AMT_CREDIT_SUM_DEBT_SUM"].fillna(0)) / (fa["AMT_INCOME_TOTAL"] + 1e-5)

    for c in fa.columns:
        if c not in ("SK_ID_CURR", "TARGET") and not pd.api.types.is_numeric_dtype(fa[c]) and str(fa[c].dtype) != "category":
            fa[c] = fa[c].astype("category")

    fa = reduce_mem(fa).sort_values("SK_ID_CURR").reset_index(drop=True)
    fa.to_parquet(os.path.join(OUT_DIR, "features_model_a.parquet"), index=False)
    return fa


def build_features_b(data_dir, fa):
    train_ids = set(fa["SK_ID_CURR"])

    ins = pd.read_csv(f"{data_dir}/installments_payments.csv")
    ins = ins[ins["SK_ID_CURR"].isin(train_ids)]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    ins["PAYMENT_DEFICIT"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]).clip(lower=0)
    ins["PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1e-5)
    ins["IS_LATE"] = (ins["DPD"] > 0).astype(np.int8)
    ins["IS_UNDERPAID"] = (ins["PAYMENT_DEFICIT"] > 0).astype(np.int8)

    ins_agg = ins.groupby("SK_ID_CURR").agg(
        INS_NUM_PAYMENTS=("SK_ID_PREV", "count"),
        INS_DPD_MAX=("DPD", "max"),
        INS_DPD_MEAN=("DPD", "mean"),
        INS_DBD_MEAN=("DBD", "mean"),
        INS_LATE_RATIO=("IS_LATE", "mean"),
        INS_UNDERPAID_RATIO=("IS_UNDERPAID", "mean"),
        INS_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INS_PAYMENT_DEFICIT_SUM=("PAYMENT_DEFICIT", "sum"),
        INS_DAYS_LAST_PAYMENT=("DAYS_ENTRY_PAYMENT", "max"),
    )
    for days, lbl in [(180, "6M"), (365, "12M")]:
        sub = ins[ins["DAYS_ENTRY_PAYMENT"] >= -days].groupby("SK_ID_CURR").agg(**{
            f"INS_DPD_MAX_{lbl}": ("DPD", "max"),
            f"INS_DPD_MEAN_{lbl}": ("DPD", "mean"),
            f"INS_LATE_RATIO_{lbl}": ("IS_LATE", "mean"),
            f"INS_UNDERPAID_RATIO_{lbl}": ("IS_UNDERPAID", "mean"),
        })
        ins_agg = ins_agg.join(sub)
    ins_agg["INS_LATE_TREND_6M_VS_ALL"] = ins_agg["INS_LATE_RATIO_6M"] / (ins_agg["INS_LATE_RATIO"] + 1e-5)
    ins_client = reduce_mem(ins_agg.reset_index())
    del ins, ins_agg
    gc.collect()

    pos = pd.read_csv(f"{data_dir}/POS_CASH_balance.csv")
    pos = pos[pos["SK_ID_CURR"].isin(train_ids)]
    pos["IS_OVERDUE"] = (pos["SK_DPD"] > 0).astype(np.int8)
    pos["IS_COMPLETED"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype(np.int8)
    pos["IS_ACTIVE"] = (pos["NAME_CONTRACT_STATUS"] == "Active").astype(np.int8)
    pos_agg = pos.groupby("SK_ID_CURR").agg(
        POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        POS_MONTHS_LAST=("MONTHS_BALANCE", "max"),
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_MEAN=("SK_DPD", "mean"),
        POS_OVERDUE_RATIO=("IS_OVERDUE", "mean"),
        POS_COMPLETED_SUM=("IS_COMPLETED", "sum"),
        POS_ACTIVE_SUM=("IS_ACTIVE", "sum"),
        POS_REMAINING_INSTALMENTS_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
    ).join(
        pos[pos["MONTHS_BALANCE"] >= -12].groupby("SK_ID_CURR").agg(
            POS_DPD_MAX_12M=("SK_DPD", "max"),
            POS_OVERDUE_RATIO_12M=("IS_OVERDUE", "mean"),
        )
    )
    pos_client = reduce_mem(pos_agg.reset_index())
    del pos, pos_agg
    gc.collect()

    cc = pd.read_csv(f"{data_dir}/credit_card_balance.csv")
    cc = cc[cc["SK_ID_CURR"].isin(train_ids)]
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-5)
    cc["CASH_DRAWINGS_RATIO"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / (cc["AMT_BALANCE"] + 1e-5)
    cc["PAID_MIN_RATIO"] = cc["AMT_PAYMENT_CURRENT"] / (cc["AMT_INST_MIN_REGULARITY"] + 1e-5)
    cc["IS_DPD"] = (cc["SK_DPD"] > 0).astype(np.int8)
    cc_agg = cc.groupby("SK_ID_CURR").agg(
        CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("UTILIZATION", "max"),
        CC_CASH_DRAWINGS_RATIO_MEAN=("CASH_DRAWINGS_RATIO", "mean"),
        CC_PAID_MIN_RATIO_MEAN=("PAID_MIN_RATIO", "mean"),
        CC_DPD_MAX=("SK_DPD", "max"),
        CC_DPD_MONTHS_SUM=("IS_DPD", "sum"),
    ).join(
        cc[cc["MONTHS_BALANCE"] >= -6].groupby("SK_ID_CURR").agg(
            CC_UTILIZATION_MEAN_6M=("UTILIZATION", "mean"),
            CC_DPD_MAX_6M=("SK_DPD", "max"),
        )
    )
    cc_client = reduce_mem(cc_agg.reset_index())
    del cc, cc_agg
    gc.collect()

    prev = pd.read_csv(f"{data_dir}/previous_application.csv")
    prev = prev[prev["SK_ID_CURR"].isin(train_ids)]
    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)
    prev["APPLICATION_CREDIT_RATIO"] = prev["AMT_CREDIT"] / (prev["AMT_APPLICATION"] + 1e-5)
    prev_agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_SUM=("IS_APPROVED", "sum"),
        PREV_REFUSED_SUM=("IS_REFUSED", "sum"),
        PREV_APPROVED_RATIO=("IS_APPROVED", "mean"),
        PREV_REFUSED_RATIO=("IS_REFUSED", "mean"),
        PREV_DAYS_DECISION_LAST=("DAYS_DECISION", "max"),
        PREV_CREDIT_APPL_RATIO_MEAN=("APPLICATION_CREDIT_RATIO", "mean"),
        PREV_DOWN_PAYMENT_RATE_MEAN=("RATE_DOWN_PAYMENT", "mean"),
    )
    prev_client = reduce_mem(prev_agg.reset_index())
    del prev, prev_agg
    gc.collect()

    fb = fa.merge(ins_client, on="SK_ID_CURR", how="left")
    fb = fb.merge(pos_client, on="SK_ID_CURR", how="left")
    fb = fb.merge(cc_client, on="SK_ID_CURR", how="left")
    fb = fb.merge(prev_client, on="SK_ID_CURR", how="left")
    fb["HAS_INTERNAL_HISTORY"] = (
        fb["INS_NUM_PAYMENTS"].notnull() | fb["POS_MONTHS_COUNT"].notnull() | fb["PREV_APP_COUNT"].notnull()
    ).astype(np.int8)
    fb = reduce_mem(fb).sort_values("SK_ID_CURR").reset_index(drop=True)
    fb.to_parquet(os.path.join(OUT_DIR, "features_model_b.parquet"), index=False)
    return fb


def save_feature_list(feats, path):
    body = "\n".join(feats)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# count={len(feats)} sha256={sha}\n{body}\n")
    return sha


def run_part1_hieu():
    data_dir = find_data_dir()
    fa_path = os.path.join(OUT_DIR, "features_model_a.parquet")
    fb_path = os.path.join(OUT_DIR, "features_model_b.parquet")

    fa = pd.read_parquet(fa_path) if os.path.exists(fa_path) else build_features_a(data_dir)
    fb = pd.read_parquet(fb_path) if os.path.exists(fb_path) else build_features_b(data_dir, fa)

    forbidden = {"SK_ID_CURR", "TARGET", "fold", "cohort_t2_N12", "cohort_t3_N12", "cohort_t4_N12"}
    feats_a = [c for c in fa.columns if c not in forbidden]
    feats_b = [c for c in fb.columns if c not in forbidden]

    hash_a = save_feature_list(feats_a, os.path.join(OUT_DIR, "feature_list_a.txt"))
    hash_b = save_feature_list(feats_b, os.path.join(OUT_DIR, "feature_list_b.txt"))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_arr = np.full(len(fa), -1, dtype=np.int8)
    for f_idx, (_, val_idx) in enumerate(skf.split(fa, fa["TARGET"])):
        fold_arr[val_idx] = f_idx

    pd.DataFrame({"SK_ID_CURR": fa["SK_ID_CURR"].to_numpy(), "fold": fold_arr}).to_csv(
        os.path.join(OUT_DIR, "folds.csv"), index=False
    )

    mono_rules = {
        "EXT_SOURCE_1": -1, "EXT_SOURCE_2": -1, "EXT_SOURCE_3": -1,
        "EXT_SOURCES_MEAN": -1, "EXT_SOURCES_MIN": -1, "EXT_SOURCES_MAX": -1,
        "RATIO_ANNUITY_TO_INCOME": 1, "RATIO_CREDIT_TO_INCOME": 1, "CROSS_TOTAL_DEBT_TO_INCOME": 1,
        "B_CREDIT_DAY_OVERDUE_MAX": 1, "B_AMT_CREDIT_SUM_OVERDUE_SUM": 1,
        "B_BB_DPD_MAX_MAX": 1, "B_BB_BAD_31PLUS_COUNT_SUM": 1, "B_BB_BAD_91PLUS_COUNT_SUM": 1,
    }
    pd.DataFrame([{"feature": f, "sign": mono_rules[f]} for f in feats_a if f in mono_rules]).to_csv(
        os.path.join(OUT_DIR, "monotone_constraints.csv"), index=False
    )

    lgb_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.05 if not QUICK_TEST else 0.1,
        "num_leaves": 31, "max_depth": 6,
        "min_child_samples": 100 if not QUICK_TEST else 20,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "reg_alpha": 1.0, "reg_lambda": 5.0, "n_jobs": -1, "verbose": -1,
    }

    def mono_vec(df, feats):
        return [
            0 if (str(df[c].dtype) == "category" or not pd.api.types.is_numeric_dtype(df[c]))
            else int(mono_rules.get(c, 0))
            for c in feats
        ]

    def probe_rounds(df, feats):
        y = df["TARGET"].to_numpy()
        params = dict(lgb_params, seed=42, monotone_constraints=mono_vec(df, feats))
        iters = []
        for f_idx in range(5):
            val_m = fold_arr == f_idx
            dtr = lgb.Dataset(df.loc[~val_m, feats], label=y[~val_m])
            dva = lgb.Dataset(df.loc[val_m, feats], label=y[val_m], reference=dtr)
            bst = lgb.train(
                params, dtr, num_boost_round=50 if QUICK_TEST else 2500,
                valid_sets=[dva], callbacks=[lgb.early_stopping(15 if QUICK_TEST else 100, verbose=False)],
            )
            iters.append(int(bst.best_iteration))
            del dtr, dva, bst
            gc.collect()
        return int(np.round(np.mean(iters))), iters

    rounds_a, probe_a = probe_rounds(fa, feats_a)
    rounds_b, probe_b = probe_rounds(fb, feats_b)
    fixed_rounds = int(np.round(0.5 * (rounds_a + rounds_b)))

    def train_suite(df, feats, tag, seeds):
        y = df["TARGET"].to_numpy()
        m_vec = mono_vec(df, feats)
        oof_seeds, seed_aucs, records = [], [], []
        for s in seeds:
            oof_s = np.zeros(len(df), dtype=np.float64)
            params = dict(lgb_params, seed=s, feature_fraction_seed=s, bagging_seed=s, monotone_constraints=m_vec)
            for f_idx in range(5):
                val_m = fold_arr == f_idx
                dtr = lgb.Dataset(df.loc[~val_m, feats], label=y[~val_m])
                bst = lgb.train(params, dtr, num_boost_round=fixed_rounds)
                bst.save_model(os.path.join(MODEL_DIR, f"model_{tag}_fold{f_idx}_seed{s}.txt"))
                raw_p = bst.predict(df.loc[val_m, feats], raw_score=True)
                oof_s[val_m] = raw_p
                records.append({"model": f"model_{tag}", "seed": int(s), "fold": int(f_idx), "auc": float(roc_auc_score(y[val_m], sigmoid(raw_p)))})
                del dtr, bst
                gc.collect()
            oof_seeds.append(oof_s)
            seed_aucs.append(float(roc_auc_score(y, sigmoid(oof_s))))

        oof_logit = np.mean(np.vstack(oof_seeds), axis=0)
        oof_prob = sigmoid(oof_logit)
        for f_idx in range(5):
            val_m = fold_arr == f_idx
            records.append({"model": f"model_{tag}", "seed": -1, "fold": int(f_idx), "auc": float(roc_auc_score(y[val_m], oof_prob[val_m]))})
        return oof_logit, oof_prob, seed_aucs, records

    seeds = [42, 123, 2024]
    logit_a, prob_a, sauc_a, rec_a = train_suite(fa, feats_a, "a", seeds)
    logit_b, prob_b, sauc_b, rec_b = train_suite(fb, feats_b, "b", seeds)

    pd.DataFrame(rec_a + rec_b).to_csv(os.path.join(OUT_DIR, "fold_seed_auc_report.csv"), index=False)

    oof_df = pd.DataFrame({
        "SK_ID_CURR": fa["SK_ID_CURR"].to_numpy(),
        "fold": fold_arr,
        "TARGET": fa["TARGET"].to_numpy(),
        "logit_model_a": logit_a,
        "pred_risk_model_a": prob_a,
        "logit_model_b": logit_b,
        "pred_risk_model_b": prob_b,
    }).sort_values("SK_ID_CURR").reset_index(drop=True)
    oof_df.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False, float_format="%.12g")

    cfg = {
        "seeds_model_a": seeds, "seeds_model_b": seeds, "n_folds": 5, "fold_random_state": 42,
        "lgbm_params": lgb_params, "probe_best_iterations_a": probe_a, "probe_best_iterations_b": probe_b,
        "probe_mean_rounds_a": rounds_a, "probe_mean_rounds_b": rounds_b, "fixed_num_boost_round": fixed_rounds,
        "feature_count_a": len(feats_a), "feature_count_b": len(feats_b),
        "feature_sha256_a": hash_a, "feature_sha256_b": hash_b,
        "seed_aucs_a": sauc_a, "seed_aucs_b": sauc_b,
        "seed_auc_std_a": float(np.std(sauc_a)), "seed_auc_std_b": float(np.std(sauc_b)),
        "oof_auc_a": float(roc_auc_score(oof_df["TARGET"], oof_df["pred_risk_model_a"])),
        "oof_auc_b": float(roc_auc_score(oof_df["TARGET"], oof_df["pred_risk_model_b"])),
        "oof_pr_auc_a": float(average_precision_score(oof_df["TARGET"], oof_df["pred_risk_model_a"])),
        "oof_pr_auc_b": float(average_precision_score(oof_df["TARGET"], oof_df["pred_risk_model_b"])),
    }
    with open(os.path.join(OUT_DIR, "config_shared.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_part1_hieu()