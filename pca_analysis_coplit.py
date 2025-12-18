#!/usr/bin/env python3
"""
PCA analysis for featurized German credit data.

- Loads artifacts/featurized/X_train_feat.csv and X_test_feat.csv
- Selects numeric continuous columns (configurable)
- Checks whether numeric part is already standardized; if not, fits StandardScaler on train and transforms both
- Runs PCA, plots scree & cumulative variance, picks number of components automatically (cumulative >= 0.90 by default)
- Saves:
    artifacts/pca_analysis/X_train_pca.csv
    artifacts/pca_analysis/X_test_pca.csv
    artifacts/pca_analysis/pca_model.joblib
    artifacts/pca_analysis/pca_explained_variance.csv
    artifacts/pca_analysis/plots/ (scree.png, cumvar.png)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

RND = 42
FEAT_DIR = "artifacts/featurized"
OUTDIR = "artifacts/pca_analysis"
PLOTS = os.path.join(OUTDIR, "plots")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)
sns.set(style="whitegrid")

# Default numeric candidate columns (these were the engineered + original numeric + ordinal used in featurize.py)
NUMERIC_CANDIDATES = [
    "credit_amount_wins", "credit_amount_log", "high_amount_flag",
    "monthly_payment", "credit_per_existing", "installment_rel", "many_dependents_flag",
    "duration", "credit_amount", "installment_commitment", "residence_since", "age",
    "existing_credits", "num_dependents",
    "savings_status_ord", "employment_ord", "job_ord"
]

def load_feat():
    xtr_path = os.path.join(FEAT_DIR, "X_train_feat.csv")
    xte_path = os.path.join(FEAT_DIR, "X_test_feat.csv")
    if not os.path.exists(xtr_path) or not os.path.exists(xte_path):
        raise FileNotFoundError("Featurized CSVs not found. Run featurize.py first.")
    X_train = pd.read_csv(xtr_path)
    X_test = pd.read_csv(xte_path)
    return X_train, X_test

def select_numeric_columns(X_train, X_test):
    # Choose numeric candidates that actually exist in the files
    avail = [c for c in NUMERIC_CANDIDATES if c in X_train.columns]
    if not avail:
        # fallback: treat all columns with dtype numeric as numeric
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    return avail

def check_standardized(df, cols, tol_mean=1e-2, tol_std_low=0.9, tol_std_high=1.1):
    # compute means and stds
    m = df[cols].mean()
    s = df[cols].std(ddof=0)
    mean_ok = np.all(np.abs(m.values) < tol_mean)
    std_ok = np.all((s.values > tol_std_low) & (s.values < tol_std_high))
    return mean_ok and std_ok, m, s

def run_pca(X_train_num, X_test_num, var_threshold=0.90):
    pca_all = PCA(n_components=min(X_train_num.shape[0], X_train_num.shape[1]), random_state=RND)
    pca_all.fit(X_train_num)
    evr = pca_all.explained_variance_ratio_
    cum = np.cumsum(evr)
    # automatic selection
    n_components = int(np.searchsorted(cum, var_threshold) + 1)
    if n_components <= 0:
        n_components = min(len(evr), X_train_num.shape[1])
    # ensure not larger than features
    n_components = min(n_components, X_train_num.shape[1])
    # refit with selected components
    pca = PCA(n_components=n_components, random_state=RND)
    X_tr_pca = pca.fit_transform(X_train_num)
    X_te_pca = pca.transform(X_test_num)
    return pca, evr, cum, X_tr_pca, X_te_pca, n_components

def save_outputs(pca, evr, cum, X_tr_pca, X_te_pca, numeric_cols):
    # save arrays
    df_tr = pd.DataFrame(X_tr_pca, columns=[f"PC{i+1}" for i in range(X_tr_pca.shape[1])])
    df_te = pd.DataFrame(X_te_pca, columns=[f"PC{i+1}" for i in range(X_te_pca.shape[1])])
    df_tr.to_csv(os.path.join(OUTDIR, "X_train_pca.csv"), index=False)
    df_te.to_csv(os.path.join(OUTDIR, "X_test_pca.csv"), index=False)
    # save pca model
    joblib.dump(pca, os.path.join(OUTDIR, "pca_model.joblib"))
    # save explained variance
    df_ev = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": evr,
        "cumulative_variance": cum
    })
    df_ev.to_csv(os.path.join(OUTDIR, "pca_explained_variance.csv"), index=False)
    # save loadings
    loadings = pd.DataFrame(pca.components_.T, index=numeric_cols, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    loadings.to_csv(os.path.join(OUTDIR, "pca_loadings.csv"))
    # plots
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(evr)+1), evr, marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "scree.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(cum)+1), cum, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Variance Explained")
    plt.axhline(0.9, color='red', linestyle='--', label='90%')
    plt.axhline(0.95, color='orange', linestyle='--', label='95%')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "cumvar.png"), dpi=150)
    plt.close()

    print("Saved PCA outputs to", OUTDIR)
    print("Plots saved to", PLOTS)

if __name__ == "__main__":
    print("Loading featurized data...")
    X_train, X_test = load_feat()
    numeric_cols = select_numeric_columns(X_train, X_test)
    print("Numeric columns used for PCA:", numeric_cols)

    # ensure consistent indexing
    X_train_num = X_train[numeric_cols].reset_index(drop=True)
    X_test_num = X_test[numeric_cols].reset_index(drop=True)

    # check if numeric columns appear to be standardized
    is_std, means, stds = check_standardized(X_train_num, numeric_cols)
    if is_std:
        print("Numeric columns appear standardized (mean ~0, std ~1). Proceeding without rescaling.")
        X_train_for_pca = X_train_num.values
        X_test_for_pca = X_test_num.values
        scaler_used = None
    else:
        print("Numeric columns NOT standardized. Fitting StandardScaler on train numeric columns and transforming...")
        scaler = StandardScaler()
        X_train_for_pca = scaler.fit_transform(X_train_num)
        X_test_for_pca = scaler.transform(X_test_num)
        joblib.dump(scaler, os.path.join(OUTDIR, "pca_scaler.joblib"))
        scaler_used = os.path.join(OUTDIR, "pca_scaler.joblib")
        print("Scaler saved to", scaler_used)

    # run PCA with automatic selection (default 90% cumulative)
    pca, evr, cum, X_tr_pca, X_te_pca, n_comp = run_pca(pd.DataFrame(X_train_for_pca), pd.DataFrame(X_test_for_pca), var_threshold=0.90)

    print(f"Selected n_components = {n_comp} (cumulative variance = {cum[n_comp-1]:.4f})")
    save_outputs(pca, evr, cum, X_tr_pca, X_te_pca, numeric_cols)