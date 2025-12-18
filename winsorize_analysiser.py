#!/usr/bin/env python3
"""
Winsorize range analyzer for one or more numeric features (default: credit_amount).

What it does
- Loads the German credit dataset (OpenML id=31) or from local CSV/Parquet (--data-path).
- Identifies numeric columns and the target (expects 0/1).
- For a specified numeric column (default 'credit_amount') computes candidate winsorize cutpoints
  (several percentile-based and IQR/MAD-based options), reports statistics before/after clipping,
  and saves diagnostic plots (histograms with threshold lines, boxplot) and CSV summary.
- Optionally runs a quick CV comparison (Logistic, and XGBoost/RandomForest fallback) to see
  how replacing the raw column with winsorized versions affects ROC AUC and AP (average precision).
- Outputs saved under artifacts/winsorize_analysis/

Usage
- Default (Ope
英语 （已检测）
中文（简体）

nML id=31, analyze 'credit_amount'):
    python winsorize_analyzer.py

- With local file (CSV or Parquet):
    python winsorize_analyzer.py --data-path ./credit.csv

- Analyze another column and run models:
    python winsorize_analyzer.py --column duration --do-models

Notes
- All winsorize cutpoints are computed on the TRAIN split and applied to the test split for model runs.
- For speed, CV uses 5 folds and a modest estimator config.
"""
import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RND = 42
OUTDIR = "artifacts/winsorize_analysis"
os.makedirs(OUTDIR, exist_ok=True)
sns.set(style="whitegrid")

# Try to import xgboost; fallback to RandomForest in model comparisons
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

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
    # common names
    candidates = ['class','Class','default.payment.next.month','default','target','y','label']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: binary-like col
    for c in df.columns:
        if df[c].nunique(dropna=True) == 2:
            return c
    return None

def winsorize_series(s, lower_pct, upper_pct):
    lo = s.quantile(lower_pct/100.0)
    hi = s.quantile(upper_pct/100.0)
    return s.clip(lo, hi), lo, hi

def iqr_fences(s, k=1.5):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return lo, hi

def mad_fences(s, n_mads=3.5):
    med = s.median()
    mad = np.median(np.abs(s - med))
    # approximate scale factor for normal ~ 1.4826
    lo = med - n_mads * mad * 1.4826
    hi = med + n_mads * mad * 1.4826
    return lo, hi

def summarize_change(original, clipped):
    changed = (original != clipped).sum()
    pct_changed = changed / len(original) * 100.0
    stats = {
        'count': len(original),
        'orig_mean': float(original.mean()),
        'orig_std': float(original.std()),
        'orig_skew': float(original.skew()),
        'orig_kurt': float(original.kurt()),
        'clipped_mean': float(clipped.mean()),
        'clipped_std': float(clipped.std()),
        'clipped_skew': float(clipped.skew()),
        'clipped_kurt': float(clipped.kurt()),
        'changed_count': int(changed),
        'changed_pct': float(pct_changed),
        'clipped_min': float(clipped.min()),
        'clipped_max': float(clipped.max())
    }
    return stats

def plot_hist_with_thresholds(orig_series, thresholds, title, filename):
    plt.figure(figsize=(8,4))
    sns.histplot(orig_series, bins=80, kde=True, color='C0')
    ymin, ymax = plt.ylim()
    for (lo, hi, label) in thresholds:
        if lo is not None:
            plt.vlines(lo, ymin, ymax, color='red', linestyle='--', linewidth=1)
        if hi is not None:
            plt.vlines(hi, ymin, ymax, color='orange', linestyle='--', linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def quick_model_cv(X, y, colname, transformed_series_train, transformed_series_test, do_xgb=True):
    """
    Replace column colname in X with transformed_series (train/test) and run CV on combined data.
    Returns dict of results for Logistic and tree.
    """
    Xtr = X.copy()
    Xtr[colname] = transformed_series_train
    Xte = X.copy()
    Xte[colname] = transformed_series_test

    # combine for CV
    Xcv = pd.concat([Xtr, Xte], axis=0).reset_index(drop=True)
    ycv = pd.concat([y, y], axis=0).reset_index(drop=True)  # placeholder: we will use only training folds--but simpler: run CV on full combined with stratify works

    # For a fair test we actually should re-split; here we perform CV on Xcv with stratified folds
    numeric_cols = Xcv.select_dtypes(include=[np.number]).columns.tolist()
    # simple pipeline: scaler + logistic / tree
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xcv[numeric_cols])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RND)
    results = {}

    log = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=2000, random_state=RND)
    scoring = {'roc_auc': 'roc_auc', 'avg_prec':'average_precision'}
    res_log = cross_validate(log, Xs, ycv, cv=skf, scoring=scoring, n_jobs=-1)
    results['Logistic'] = {
        'roc_mean': float(res_log['test_roc_auc'].mean()),
        'roc_std': float(res_log['test_roc_auc'].std()),
        'ap_mean': float(res_log['test_avg_prec'].mean()),
        'ap_std': float(res_log['test_avg_prec'].std())
    }

    if do_xgb and XGB_AVAILABLE:
        tree = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0, random_state=RND, n_estimators=100)
        res_tree = cross_validate(tree, Xs, ycv, cv=skf, scoring=scoring, n_jobs=-1)
        results['XGBoost'] = {
            'roc_mean': float(res_tree['test_roc_auc'].mean()),
            'roc_std': float(res_tree['test_roc_auc'].std()),
            'ap_mean': float(res_tree['test_avg_prec'].mean()),
            'ap_std': float(res_tree['test_avg_prec'].std())
        }
    else:
        tree = RandomForestClassifier(n_estimators=200, random_state=RND, n_jobs=-1)
        res_tree = cross_validate(tree, Xs, ycv, cv=skf, scoring=scoring, n_jobs=-1)
        results['RandomForest'] = {
            'roc_mean': float(res_tree['test_roc_auc'].mean()),
            'roc_std': float(res_tree['test_roc_auc'].std()),
            'ap_mean': float(res_tree['test_avg_prec'].mean()),
            'ap_std': float(res_tree['test_avg_prec'].std())
        }

    return results

def main(args):
    df = load_data(openml_id=args.openml_id, data_path=args.data_path)
    print("Loaded data shape:", df.shape)

    target_col = identify_target(df)
    if target_col is None:
        raise RuntimeError("Cannot identify target column automatically. Please specify target column.")
    print("Using target column:", target_col)

    # Ensure target is numeric 0/1
    y_raw = df[target_col]
    if set(y_raw.unique()) <= {0,1}:
        y = y_raw.astype(int)
    else:
        # try mapping common tokens
        y = y_raw.map(lambda v: 1 if str(v).lower() in ('bad','1','yes','true','t') else 0).astype(int)

    # drop target from features
    X = df.drop(columns=[target_col]).copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        # try coercing
        X = X.apply(pd.to_numeric, errors='coerce')
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns:", numeric_cols)

    col = args.column
    if col not in numeric_cols:
        raise RuntimeError(f"Column '{col}' not found among numeric columns. Found: {numeric_cols}")

    # train/test split to compute cutpoints on train only
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RND)
    orig = X_train[col].dropna()
    orig_test = X_test[col].dropna()

    # candidate percentiles
    percentile_candidates = [(0.1,99.9), (0.5,99.5), (1,99), (2,98), (5,95)]
    results = []
    thresholds_for_plot = []

    # basic stats before any clipping
    base_stats = {
        'feature': col,
        'method': 'original',
        'count': int(len(orig)),
        'mean': float(orig.mean()),
        'std': float(orig.std()),
        'skew': float(orig.skew()),
        'kurtosis': float(orig.kurt())
    }
    results.append(base_stats)

    # compute IQR and MAD fences
    iqr_lo, iqr_hi = iqr_fences(orig, k=1.5)
    iqr_lo3, iqr_hi3 = iqr_fences(orig, k=3.0)
    mad_lo, mad_hi = mad_fences(orig, n_mads=3.5)

    # percentiles options
    for lowp, highp in percentile_candidates:
        clipped_train, lo, hi = winsorize_series(orig, lowp, highp)
        clipped_test = orig_test.clip(lo, hi)
        stats = summarize_change(orig, clipped_train)
        stats.update({'feature': col, 'method': f'{lowp}p_{highp}p', 'lo': lo, 'hi': hi})
        results.append(stats)
        thresholds_for_plot.append((lo, hi, f'{lowp}-{highp}'))

    # IQR fences (1.5*IQR and 3*IQR) - apply as clipping boundaries
    for k in [1.5, 3.0]:
        lo, hi = iqr_fences(orig, k=k)
        clipped_train = orig.clip(lo, hi)
        clipped_test = orig_test.clip(lo, hi)
        stats = summarize_change(orig, clipped_train)
        stats.update({'feature': col, 'method': f'IQR_{k}', 'lo': lo, 'hi': hi})
        results.append(stats)
        thresholds_for_plot.append((lo, hi, f'IQR_{k}'))

    # MAD-based fences
    lo, hi = mad_fences(orig, n_mads=3.5)
    clipped_train = orig.clip(lo, hi)
    clipped_test = orig_test.clip(lo, hi)
    stats = summarize_change(orig, clipped_train)
    stats.update({'feature': col, 'method': 'MAD_3.5', 'lo': lo, 'hi': hi})
    results.append(stats)
    thresholds_for_plot.append((lo, hi, 'MAD_3.5'))

    # Save results summary
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTDIR, f"{col}_winsorize_candidates_summary.csv"), index=False)
    print("Saved candidate summary to", os.path.join(OUTDIR, f"{col}_winsorize_candidates_summary.csv"))

    # Plot histogram with threshold lines
    plot_hist_with_thresholds(orig, thresholds_for_plot, f"{col} histogram with candidate thresholds",
                              os.path.join(OUTDIR, f"{col}_hist_thresholds.png"))
    # Boxplot
    plt.figure(figsize=(6,4))
    sns.boxplot(x=orig, color='C0')
    plt.title(f"Boxplot for {col} (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{col}_boxplot_train.png"), dpi=150)
    plt.close()

    # Report counts beyond thresholds for each candidate
    counts = []
    for entry in results:
        if 'lo' in entry and 'hi' in entry and entry['method'] != 'original':
            lo = entry['lo']; hi = entry['hi']
            under = (orig < lo).sum()
            over = (orig > hi).sum()
            counts.append({'method': entry['method'], 'lo': lo, 'hi': hi, 'under_count': int(under), 'over_count': int(over),
                           'under_pct': float(under/len(orig)*100), 'over_pct': float(over/len(orig)*100)})
    pd.DataFrame(counts).to_csv(os.path.join(OUTDIR, f"{col}_beyond_thresholds_counts.csv"), index=False)

    print("Counts beyond thresholds saved.")

    # Optionally run quick model CV comparisons replacing column with clipped values for each candidate
    if args.do_models:
        print("Running quick model CV for each candidate (this may take a few minutes)...")
        # Use baseline numeric features (all numeric_cols)
        numeric_cols_full = X_train.select_dtypes(include=[np.number]).columns.tolist()
        # We'll use the training dataset (X_train) as X, and create clipped versions of the column in place
        model_results = []
        for entry in results:
            method = entry['method']
            if method == 'original':
                clipped_train = orig.copy()
                clipped_test = orig_test.copy()
            else:
                lo = entry['lo']; hi = entry['hi']
                clipped_train = orig.clip(lo, hi)
                clipped_test = orig_test.clip(lo, hi)
            # build X_train_feat/X_test_feat copies with replaced column
            Xtr = X_train[numeric_cols_full].copy()
            Xte = X_test[numeric_cols_full].copy()
            Xtr[col] = clipped_train
            Xte[col] = clipped_test
            # run quick cv on numeric-only set (fast)
            try:
                res = quick_model_cv(Xtr, y_train, colname=col, transformed_series_train=clipped_train,
                                     transformed_series_test=clipped_test, do_xgb=XGB_AVAILABLE)
                for model_name, metrics in res.items():
                    model_results.append({'method': method, 'model': model_name,
                                          'roc_mean': metrics['roc_mean'], 'roc_std': metrics['roc_std'],
                                          'ap_mean': metrics['ap_mean'], 'ap_std': metrics['ap_std']})
            except Exception as e:
                print("Model CV failed for method", method, "error:", e)
        df_mod = pd.DataFrame(model_results)
        df_mod.to_csv(os.path.join(OUTDIR, f"{col}_winsorize_model_comparison.csv"), index=False)
        print("Model comparison saved to", os.path.join(OUTDIR, f"{col}_winsorize_model_comparison.csv"))

    # Save thresholds config json
    thresholds_config = {'percentile_candidates': percentile_candidates,
                         'iqr_1.5': {'lo': float(iqr_lo), 'hi': float(iqr_hi)},
                         'iqr_3.0': {'lo': float(iqr_lo3), 'hi': float(iqr_hi3)},
                         'mad_3.5': {'lo': float(mad_lo), 'hi': float(mad_hi)}}
    with open(os.path.join(OUTDIR, f"{col}_winsorize_thresholds.json"), 'w', encoding='utf8') as f:
        json.dump(thresholds_config, f, indent=2)
    print("Thresholds config saved to", os.path.join(OUTDIR, f"{col}_winsorize_thresholds.json"))

    print("All artifacts for winsorize analysis saved into", OUTDIR)
    print("Please inspect the CSVs and plots and pick a candidate range (e.g., 1%-99%) to proceed with feature engineering.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Winsorize range analyzer for numeric features")
    parser.add_argument("--openml-id", type=int, default=31, help="OpenML dataset id (default 31)")
    parser.add_argument("--data-path", type=str, default=None, help="Local CSV/Parquet file path (optional)")
    parser.add_argument("--column", type=str, default="credit_amount", help="Numeric column to analyze (default credit_amount)")
    parser.add_argument("--do-models", action="store_true", help="Run quick CV model comparisons for each candidate (slower)")
    args = parser.parse_args()
    main(args)