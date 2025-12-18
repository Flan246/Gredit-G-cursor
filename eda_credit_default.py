#!/usr/bin/env python3
"""
EDA script for Credit dataset (default OpenML id=31).
Outputs descriptive tables and figures into artifacts/ directory.

Usage:
    python eda_credit_default.py
    python eda_credit_default.py --data-path ./local_credit.csv
    python eda_credit_default.py --openml-id 31

Outputs (artifacts/):
 - summary_stats.csv
 - missing_summary.csv
 - categorical_counts.csv
 - eda_readme.txt
 - figures: target_dist.png, hist_feature_*.png, box_feature_*.png, corr_numeric.png, pairplot.png (optional)
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml

RND = 42
OUTDIR = "artifacts"
os.makedirs(OUTDIR, exist_ok=True)
sns.set(style="whitegrid")

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
    # Try common target names
    candidates = ['default.payment.next.month', 'default', 'Class', 'class', 'target', 'DEFAULT', 'y', 'label']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: pick binary-like column with 2 unique values or <=3 that looks like 0/1/yes/no/good/bad
    for col in df.columns:
        nunq = df[col].nunique(dropna=True)
        if nunq <= 3:
            uniqs = list(df[col].dropna().unique())
            strset = set([str(u).lower() for u in uniqs])
            if strset <= {'0','1','yes','no','y','n','good','bad','true','false'}:
                return col
    # if not found, return None
    return None

def normalize_target(series):
    s = series.astype(str).str.strip()
    # try numeric first
    try:
        nums = pd.to_numeric(s)
        if set(nums.dropna().unique()) <= {0,1}:
            return nums.fillna(0).astype(int), { }
    except Exception:
        pass
    mapping = {}
    uniques = list(s.dropna().unique())
    # prefer mapping 'bad'/'yes'/'1' to 1
    for u in uniques:
        lu = str(u).lower()
        if ('bad' in lu) or ('yes' in lu) or ('1' == lu) or ('true' in lu) or ('t' == lu):
            mapping[u] = 1
        else:
            mapping[u] = 0
    return s.map(mapping).astype(int), mapping

def describe_numeric(df, numeric_cols):
    desc = df[numeric_cols].describe().T
    desc['nan_count'] = df[numeric_cols].isna().sum().values
    desc = desc.reset_index().rename(columns={'index':'feature'})
    return desc

def categorical_counts(df, cat_cols):
    frames = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        tmp = pd.DataFrame({c: vc.index.astype(str), 'count': vc.values})
        tmp['feature'] = c
        frames.append(tmp[['feature', c, 'count']])
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame(columns=['feature','value','count'])

def plot_target_dist(y, outdir):
    plt.figure(figsize=(4,3))
    sns.countplot(x=y, palette='coolwarm')
    plt.xlabel("target")
    plt.ylabel("count")
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "target_dist.png"), dpi=150)
    plt.close()

def plot_numeric_histograms(df, numeric_cols, outdir, bins=50, max_per_file=20):
    # Plot histograms in batches
    cols = numeric_cols
    batch_size = max_per_file
    for i in range(0, len(cols), batch_size):
        batch = cols[i:i+batch_size]
        n = len(batch)
        cols_n = min(n, batch_size)
        fig, axes = plt.subplots((cols_n+1)//2, 2, figsize=(10, 4*((cols_n+1)//2)))
        axes = axes.flatten()
        for ax, col in zip(axes, batch):
            sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax, color='C0')
            ax.set_title(col)
        for ax in axes[len(batch):]:
            fig.delaxes(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_features_{i//batch_size+1}.png"), dpi=150)
        plt.close()

def plot_box_by_target(df, numeric_cols, y, outdir, top_n=12):
    # choose top_n numeric cols by variance
    var = df[numeric_cols].var().sort_values(ascending=False)
    top = var.index.tolist()[:top_n]
    plt.figure(figsize=(12, 4*int(np.ceil(len(top)/3))))
    for i, col in enumerate(top):
        plt.subplot(int(np.ceil(len(top)/3)), 3, i+1)
        sns.boxplot(x=y, y=df[col], palette='Set2')
        plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "box_by_target_topvars.png"), dpi=150)
    plt.close()

def plot_correlation(df, numeric_cols, outdir, top_k=40):
    # compute correlation on numeric cols; if too many, select top_k by variance
    if len(numeric_cols) > top_k:
        var = df[numeric_cols].var().sort_values(ascending=False)
        selected = var.index[:top_k].tolist()
    else:
        selected = numeric_cols
    corr = df[selected].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
    plt.title("Correlation matrix (selected numeric features)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "corr_numeric.png"), dpi=150)
    plt.close()
    return corr

def pairplot_if_small(df, numeric_cols, outdir, max_cols=6):
    if len(numeric_cols) <= max_cols:
        sns.pairplot(df[numeric_cols].dropna().sample(min(2000, len(df))), diag_kind='kde', plot_kws={'s':10, 'alpha':0.6})
        plt.savefig(os.path.join(outdir, "pairplot.png"), dpi=150)
        plt.close()
    else:
        # choose top by variance
        var = df[numeric_cols].var().sort_values(ascending=False)
        top = var.index[:max_cols].tolist()
        sns.pairplot(df[top].dropna().sample(min(2000, len(df))), diag_kind='kde', plot_kws={'s':10, 'alpha':0.6})
        plt.savefig(os.path.join(outdir, "pairplot_topvars.png"), dpi=150)
        plt.close()

def missing_summary(df):
    ms = pd.DataFrame({
        'feature': df.columns,
        'missing_count': df.isna().sum().values,
        'missing_pct': (df.isna().sum().values / len(df) * 100)
    })
    ms = ms.sort_values('missing_pct', ascending=False)
    return ms

def main(args):
    df = load_data(openml_id=args.openml_id, data_path=args.data_path)
    print("Data loaded. Shape:", df.shape)
    target_col = identify_target(df)
    if target_col is None:
        raise RuntimeError("无法自动识别目标列，请手动指定 --target-col")
    print("Identified target column:", target_col)
    y_raw = df[target_col]
    y, mapping = normalize_target(y_raw)
    if mapping:
        print("Target mapping inferred:", mapping)
    df = df.drop(columns=[target_col])
    df[target_col + "_norm"] = y

    # identify numeric and categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude the normalized target if accidentally numeric
    numeric_cols = [c for c in numeric_cols if c != target_col + "_norm"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != target_col + "_norm"]

    print(f"Numeric cols ({len(numeric_cols)}):", numeric_cols[:30])
    print(f"Categorical cols ({len(categorical_cols)}):", categorical_cols[:30])

    # descriptive stats
    num_desc = describe_numeric(df, numeric_cols)
    num_desc.to_csv(os.path.join(OUTDIR, "summary_stats.csv"), index=False)

    cat_counts = categorical_counts(df, categorical_cols)
    cat_counts.to_csv(os.path.join(OUTDIR, "categorical_counts.csv"), index=False)

    ms = missing_summary(df)
    ms.to_csv(os.path.join(OUTDIR, "missing_summary.csv"), index=False)

    # target distribution
    plot_target_dist(y, OUTDIR)

    # histograms
    if len(numeric_cols) > 0:
        plot_numeric_histograms(df, numeric_cols, OUTDIR, bins=50, max_per_file=12)
        plot_box_by_target(df, numeric_cols, y, OUTDIR, top_n=12)
        corr = plot_correlation(df, numeric_cols, OUTDIR, top_k=40)
        pairplot_if_small(df, numeric_cols, OUTDIR, max_cols=6)
    else:
        print("No numeric columns to plot.")

    # Save a short text summary
    with open(os.path.join(OUTDIR, "eda_readme.txt"), "w", encoding="utf8") as f:
        f.write("EDA Summary\n")
        f.write("===========\n\n")
        f.write(f"Data shape: {df.shape}\n")
        f.write(f"Target column (normalized): {target_col + '_norm'}\n")
        f.write(f"Target mapping (inferred): {mapping}\n\n")
        f.write("Numeric columns count: %d\n" % len(numeric_cols))
        f.write("Categorical columns count: %d\n\n" % len(categorical_cols))
        f.write("Top numeric features by variance:\n")
        topvars = df[numeric_cols].var().sort_values(ascending=False).head(10)
        f.write(topvars.to_string())
        f.write("\n\nMissingness (top 20 features):\n")
        f.write(ms.head(20).to_string(index=False))
        f.write("\n\nFiles saved in artifacts/: summary_stats.csv, missing_summary.csv, categorical_counts.csv, hist_features_*.png, box_by_target_topvars.png, corr_numeric.png, pairplot*.png (if generated), target_dist.png\n")

    print("EDA complete. Outputs saved in", OUTDIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for credit dataset")
    parser.add_argument("--openml-id", type=int, default=31, help="OpenML dataset id (default 31)")
    parser.add_argument("--data-path", type=str, default=None, help="Local CSV/Parquet file path (optional)")
    parser.add_argument("--target-col", type=str, default=None, help="If automatic target identification fails, specify target column name")
    args = parser.parse_args()
    main(args)