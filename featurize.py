#!/usr/bin/env python3
"""
Feature engineering script for German Credit (OpenML id=31) — produces engineered train/test datasets
and saves encoders/parameters for reproducible pipelines.

Key actions:
- Loads dataset (OpenML id=31 by default) or from local file (--data-path).
- Identifies binary target (expects 0/1 already mapped).
- Performs a stratified train/test split (test_size=0.2, random_state=42).
- Applies winsorize to credit_amount using the chosen range 1% - 99% (fit on train, apply to test).
- Generates engineered features:
    * credit_amount_wins (winsorized)
    * credit_amount_log = log1p(credit_amount)
    * high_amount_flag (credit_amount > train 99th percentile)
    * monthly_payment = credit_amount / duration
    * credit_per_existing = credit_amount / existing_credits
    * installment_rel = monthly_payment / (installment_commitment + eps)
    * age_bucket (bins)
    * num_dependents_flag (>=2)
    * ordinal encodings for savings_status, employment, job
    * one-hot encodings for selected nominal columns
- Saves:
    artifacts/
      X_train_feat.csv, X_test_feat.csv, y_train.csv, y_test.csv
      encoders/onehot_encoder.joblib, encoders/ordinal_maps.json, encoders/scaler.joblib
      winsor_params.json
      features_list.json
      featurize_readme.txt
Usage:
    python featurize.py
    python featurize.py --data-path ./credit.csv
"""
import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

RND = 42
OUTDIR = "artifacts/featurized"
ENC_DIR = os.path.join(OUTDIR, "encoders")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(ENC_DIR, exist_ok=True)

EPS = 1e-8

# ----- ordinal mappings (adjustable) -----
ORDINAL_MAPS = {
    "savings_status": {
        # mapping 'no known savings' -> 0 (lowest), then increasing amounts
        "no known savings": 0,
        "<100": 1,
        "100<=X<500": 2,
        "500<=X<1000": 3,
        ">=1000": 4
    },
    "employment": {
        "unemployed": 0,
        "<1": 1,
        "1<=X<4": 2,
        "4<=X<7": 3,
        ">=7": 4
    },
    "job": {
        "unemp/unskilled non res": 0,
        "unskilled resident": 1,
        "skilled": 2,
        "high qualif/self emp/mgmt": 3
    }
}

# nominal columns to one-hot encode (as present in dataset)
ONEHOT_COLS = [
    "purpose", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "own_telephone", "foreign_worker"
]

def load_data(openml_id=31, data_path=None):
    if data_path:
        print(f"Loading local data from {data_path} ...")
        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        return df
    print(f"Fetching OpenML dataset id={openml_id} ...")
    ds = fetch_openml(data_id=openml_id, as_frame=True)
    df = ds.frame.copy()
    return df

def identify_target(df):
    candidates = ['class','Class','default.payment.next.month','default','target','y','label']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: choose binary-like column
    for c in df.columns:
        if df[c].nunique(dropna=True) == 2:
            return c
    return None

def safe_div(a, b):
    return a / (b + EPS)

def apply_ordinal_map(series, mapping, default=np.nan):
    return series.map(mapping).fillna(default)

def featurize(train_df, test_df, target_col, winsor_lower_pct=1.0, winsor_upper_pct=99.0):
    """
    Input: train_df/test_df contain original columns (target removed)
    Returns: X_train_feat_df, X_test_feat_df, y_train, y_test (y passed separately)
    Also fits encoders (onehot) and scaler on numeric features and returns them
    """
    df_tr = train_df.copy()
    df_te = test_df.copy()

    # ---------- Winsorize credit_amount on train (1%-99%) ----------
    col = "credit_amount"
    if col not in df_tr.columns:
        raise RuntimeError(f"Expected column '{col}' not found in data.")
    lo = float(df_tr[col].quantile(winsor_lower_pct/100.0))
    hi = float(df_tr[col].quantile(winsor_upper_pct/100.0))
    winsor_params = {"column": col, "lower_pct": winsor_lower_pct, "upper_pct": winsor_upper_pct, "lo": lo, "hi": hi}
    df_tr[col + "_wins"] = df_tr[col].clip(lo, hi)
    # apply same cutpoints to test
    df_te[col + "_wins"] = df_te[col].clip(lo, hi)

    # ---------- Additional features ----------
    # log transform of credit_amount (original)
    df_tr[col + "_log"] = np.log1p(df_tr[col])
    df_te[col + "_log"] = np.log1p(df_te[col])

    # high amount flag (based on train hi)
    df_tr["high_amount_flag"] = (df_tr[col] > hi).astype(int)
    df_te["high_amount_flag"] = (df_te[col] > hi).astype(int)

    # monthly payment = credit_amount / duration
    df_tr["duration_safe"] = df_tr["duration"].replace({0:1})
    df_te["duration_safe"] = df_te["duration"].replace({0:1})
    df_tr["monthly_payment"] = safe_div(df_tr[col], df_tr["duration_safe"])
    df_te["monthly_payment"] = safe_div(df_te[col], df_te["duration_safe"])

    # credit per existing credit
    df_tr["existing_credits_safe"] = df_tr["existing_credits"].replace({0:1})
    df_te["existing_credits_safe"] = df_te["existing_credits"].replace({0:1})
    df_tr["credit_per_existing"] = safe_div(df_tr[col], df_tr["existing_credits_safe"])
    df_te["credit_per_existing"] = safe_div(df_te[col], df_te["existing_credits_safe"])

    # installment_rel = monthly_payment / installment_commitment
    df_tr["installment_rel"] = safe_div(df_tr["monthly_payment"], df_tr["installment_commitment"])
    df_te["installment_rel"] = safe_div(df_te["monthly_payment"], df_te["installment_commitment"])

    # age buckets
    bins = [0, 25, 35, 50, 120]
    labels = ["<25", "25-34", "35-49", "50+"]
    df_tr["age_bucket"] = pd.cut(df_tr["age"], bins=bins, labels=labels, include_lowest=True)
    df_te["age_bucket"] = pd.cut(df_te["age"], bins=bins, labels=labels, include_lowest=True)

    # num_dependents flag
    df_tr["many_dependents_flag"] = (df_tr["num_dependents"] >= 2).astype(int)
    df_te["many_dependents_flag"] = (df_te["num_dependents"] >= 2).astype(int)

    # drop temporary safe cols
    df_tr = df_tr.drop(columns=["duration_safe","existing_credits_safe"], errors='ignore')
    df_te = df_te.drop(columns=["duration_safe","existing_credits_safe"], errors='ignore')

    # ---------- Ordinal encoding (map dictionaries) ----------
    # apply mappings; unknowns -> np.nan (will be one-hot or left as is)
    for feat, mapping in ORDINAL_MAPS.items():
        if feat in df_tr.columns:
            df_tr[feat + "_ord"] = apply_ordinal_map(df_tr[feat].astype(str), mapping, default=np.nan)
            df_te[feat + "_ord"] = apply_ordinal_map(df_te[feat].astype(str), mapping, default=np.nan)

    # ---------- One-hot encoding for nominal cols ----------
    # Fit on train only
    onehot_cols_present = [c for c in ONEHOT_COLS if c in df_tr.columns]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    if len(onehot_cols_present) > 0:
        ohe.fit(df_tr[onehot_cols_present].astype(str).fillna("NA"))
        ohe_tr = pd.DataFrame(ohe.transform(df_tr[onehot_cols_present].astype(str).fillna("NA")),
                              columns=ohe.get_feature_names_out(onehot_cols_present),
                              index=df_tr.index)
        ohe_te = pd.DataFrame(ohe.transform(df_te[onehot_cols_present].astype(str).fillna("NA")),
                              columns=ohe.get_feature_names_out(onehot_cols_present),
                              index=df_te.index)
    else:
        ohe_tr = pd.DataFrame(index=df_tr.index)
        ohe_te = pd.DataFrame(index=df_te.index)

    # ---------- Select numeric features to keep (engineered + existing numeric + ordinal + one-hot) ----------
    # We keep:
    #  - engineered numeric: credit_amount_wins, credit_amount_log, high_amount_flag, monthly_payment,
    #    credit_per_existing, installment_rel, many_dependents_flag
    engineered_num = [
        "credit_amount_wins", "credit_amount_log", "high_amount_flag",
        "monthly_payment", "credit_per_existing", "installment_rel", "many_dependents_flag"
    ]
    # include some original numeric features as well
    original_numeric = [c for c in ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"] if c in df_tr.columns]
    # ordinal numeric features
    ordinal_num = [f + "_ord" for f in ORDINAL_MAPS.keys() if (f + "_ord") in df_tr.columns]

    numeric_keep = engineered_num + original_numeric + ordinal_num

    # Build final DataFrames
    X_train_num = df_tr[numeric_keep].copy()
    X_test_num = df_te[numeric_keep].copy()

    # Concatenate one-hot columns
    if not ohe_tr.empty:
        X_train_feat = pd.concat([X_train_num.reset_index(drop=True), ohe_tr.reset_index(drop=True)], axis=1)
        X_test_feat = pd.concat([X_test_num.reset_index(drop=True), ohe_te.reset_index(drop=True)], axis=1)
    else:
        X_train_feat = X_train_num.copy()
        X_test_feat = X_test_num.copy()

    # Fill any remaining NaNs in ordinal columns or numeric with median from train
    medians = X_train_feat.median()
    X_train_feat = X_train_feat.fillna(medians)
    X_test_feat = X_test_feat.fillna(medians)

    # ---------- Fit StandardScaler on numeric part (not on one-hot columns) ----------
    # Identify which columns are numeric (we consider engineered + original + ordinal as numeric)
    # For simplicity, scale all columns except one-hot dummy columns (which contain '_' from get_feature_names_out)
    onehot_feature_names = list(ohe.get_feature_names_out(onehot_cols_present)) if len(onehot_cols_present) else []
    cols_to_scale = [c for c in X_train_feat.columns if c not in onehot_feature_names]
    scaler = StandardScaler().fit(X_train_feat[cols_to_scale])
    X_train_scaled_part = pd.DataFrame(scaler.transform(X_train_feat[cols_to_scale]), columns=cols_to_scale, index=X_train_feat.index)
    X_test_scaled_part = pd.DataFrame(scaler.transform(X_test_feat[cols_to_scale]), columns=cols_to_scale, index=X_test_feat.index)

    # recombine scaled numeric part + one-hot (unchanged)
    if onehot_feature_names:
        X_train_final = pd.concat([X_train_scaled_part.reset_index(drop=True), X_train_feat[onehot_feature_names].reset_index(drop=True)], axis=1)
        X_test_final = pd.concat([X_test_scaled_part.reset_index(drop=True), X_test_feat[onehot_feature_names].reset_index(drop=True)], axis=1)
    else:
        X_train_final = X_train_scaled_part.copy()
        X_test_final = X_test_scaled_part.copy()

    # final features list
    feature_list = X_train_final.columns.tolist()

    # return everything and fitted artifacts
    artifacts = {
        "onehot_encoder": ohe,
        "scaler": scaler,
        "winsor_params": winsor_params,
        "ordinal_maps": ORDINAL_MAPS,
        "feature_list": feature_list
    }
    return X_train_final, X_test_final, artifacts

def main(args):
    df = load_data(openml_id=args.openml_id, data_path=args.data_path)
    print("Loaded data shape:", df.shape)

    target_col = identify_target(df)
    if target_col is None:
        raise RuntimeError("Cannot find target column automatically. Please specify --target-col.")
    print("Using target column:", target_col)

    # Normalize target to 0/1 numeric (handle string labels like 'good'/'bad')
    y_raw = df[target_col].astype(str)
    if set(y_raw.unique()) <= {'0','1','0.0','1.0'}:
        y = y_raw.astype(float).astype(int)
    else:
        # Map common string labels: 'bad'/'yes'/'default' -> 1, 'good'/'no' -> 0
        map_dict = {}
        uniques = list(y_raw.unique())
        for u in uniques:
            lu = u.lower()
            if 'bad' in lu or 'yes' in lu or '1' in lu or 'default' in lu or 't' in lu:
                map_dict[u] = 1
            else:
                map_dict[u] = 0
        y = y_raw.map(map_dict).astype(int)
        print("Target mapping (inferred):", map_dict)
    
    X = df.drop(columns=[target_col]).copy()

    # Split (same convention as earlier scripts)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RND)
    print("Train/test shapes:", X_train.shape, X_test.shape)

    # Featurize with winsorize 1%-99% for credit_amount as requested
    X_train_feat, X_test_feat, artifacts = featurize(X_train, X_test, target_col, winsor_lower_pct=1.0, winsor_upper_pct=99.0)

    # Save feature CSVs and y
    X_train_feat.to_csv(os.path.join(OUTDIR, "X_train_feat.csv"), index=False)
    X_test_feat.to_csv(os.path.join(OUTDIR, "X_test_feat.csv"), index=False)
    y_train.to_csv(os.path.join(OUTDIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTDIR, "y_test.csv"), index=False)

    # Save artifacts: onehot encoder (joblib), scaler (joblib), ordinal maps (json), winsor params (json), feature list
    joblib.dump(artifacts["onehot_encoder"], os.path.join(ENC_DIR, "onehot_encoder.joblib"))
    joblib.dump(artifacts["scaler"], os.path.join(ENC_DIR, "scaler.joblib"))
    with open(os.path.join(ENC_DIR, "ordinal_maps.json"), "w", encoding="utf8") as f:
        json.dump(artifacts["ordinal_maps"], f, indent=2, ensure_ascii=False)
    with open(os.path.join(ENC_DIR, "winsor_params.json"), "w", encoding="utf8") as f:
        json.dump(artifacts["winsor_params"], f, indent=2)
    with open(os.path.join(ENC_DIR, "feature_list.json"), "w", encoding="utf8") as f:
        json.dump(artifacts["feature_list"], f, indent=2)

    # Save short readme
    with open(os.path.join(OUTDIR, "featurize_readme.txt"), "w", encoding="utf8") as f:
        f.write("Featurization summary\n")
        f.write("=====================\n")
        f.write(f"Winsorize applied to {artifacts['winsor_params']['column']} with lower_pct={artifacts['winsor_params']['lower_pct']} "
                f"upper_pct={artifacts['winsor_params']['upper_pct']} (lo={artifacts['winsor_params']['lo']}, hi={artifacts['winsor_params']['hi']})\n\n")
        f.write("Ordinal maps saved in encoders/ordinal_maps.json\n")
        f.write("OneHotEncoder saved in encoders/onehot_encoder.joblib\n")
        f.write("StandardScaler saved in encoders/scaler.joblib\n")
        f.write("Feature list saved in encoders/feature_list.json\n")
        f.write("Train/test CSV: X_train_feat.csv, X_test_feat.csv, y_train.csv, y_test.csv\n")

    print("Featurization complete. Artifacts written to", OUTDIR)
    print("Encoders and params written to", ENC_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize German credit dataset (winsorize range preselected 1%-99%)")
    parser.add_argument("--openml-id", type=int, default=31, help="OpenML dataset id (default 31)")
    parser.add_argument("--data-path", type=str, default=None, help="Local CSV/Parquet file path (optional)")
    args = parser.parse_args()
    main(args)