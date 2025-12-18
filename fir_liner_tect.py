"""
快速判定线性/非线性模式与 Winsorize 敏感性测试脚本
- 从 OpenML 加载 data_id=31（你指定的数据集）
- 对数值特征做三种预处理（原始/ Winsorize / log1p）
- 可视化：PCA(2D), t-SNE(2D), UMAP(2D if available)
- 基线模型比较：LogisticRegression(class_weight='balanced') vs Tree (XGBoost if installed else RandomForest)
- 使用 stratified 5-fold CV，输出 ROC AUC 与 PR-AUC (average precision)
"""
import os
import warnings

JOBLIB_TMP = r"D:\TEMP\joblib_temp"
os.makedirs(JOBLIB_TMP, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = JOBLIB_TMP

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score

RANDOM_STATE = 42
OUTDIR = "artifacts"
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(RANDOM_STATE)

# --- load dataset ---
print("Fetching OpenML data_id=31 ... (this may take a moment)")
ds = fetch_openml(data_id=31, as_frame=True)
df = ds.frame.copy()
print("Loaded shape:", df.shape)
print("Columns (first 50):", df.columns.tolist()[:50])

# --- try to identify target column heuristically ---
possible_targets = ['default.payment.next.month', 'default', 'Class', 'class', 'target', 'default_payment_next_month']
target_col = None
for name in possible_targets:
    if name in df.columns:
        target_col = name
        break
# fallback: if dataset has a column called 'default.payment.next.month' not found, attempt common variations
if target_col is None:
    # Look for binary columns with 0/1 or 'yes'/'no' or 'good'/'bad'
    for col in df.columns:
        if df[col].nunique() <= 3:
            uniques = df[col].dropna().unique()
            if set(map(str,uniques)).issubset({'0','1','0.0','1.0','yes','no','YES','NO','good','bad','Good','Bad'}):
                target_col = col
                break

if target_col is None:
    raise RuntimeError("无法自动识别目标列。请手动检查 df.columns 并修改脚本中的 target_col。")
print("Identified target column:", target_col)
# normalize target to 0/1 numeric
y_raw = df[target_col].astype(str)
map_dict = None
if set(y_raw.unique()) <= {'0','1','0.0','1.0'}:
    y = y_raw.astype(float).astype(int)
else:
    # common case: 'yes'/'no' or 'good'/'bad' or 'pay default' variants
    map_dict = {}
    uniques = list(y_raw.unique())
    # prefer mapping positive/default to 1; try to detect token 'bad' or 'yes' or '1'
    for u in uniques:
        lu = u.lower()
        if 'bad' in lu or 'yes' in lu or '1' in lu or 'default' in lu or 't' in lu:
            map_dict[u] = 1
        else:
            map_dict[u] = 0
    y = y_raw.map(map_dict).astype(int)
    print("Target mapping (inferred):", map_dict)

# drop target col from df
X_full = df.drop(columns=[target_col]).copy()
# select numeric columns as features for this quick check
numeric_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    # try coercing columns that look numeric
    coerced = X_full.apply(pd.to_numeric, errors='coerce')
    numeric_cols = coerced.select_dtypes(include=[np.number]).columns.tolist()
    X_full = coerced

print(f"Using numeric columns (count={len(numeric_cols)}): {numeric_cols[:30]}")

# Basic missing-value fill with median (we will test preprocessing variants on numeric features)
# 缺失值统计
X_num = X_full[numeric_cols].copy()
print("Missing per numeric col:\n", X_num.isna().sum().sort_values(ascending=False).head(10))

# train/test split stratified
X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("Train/test shapes:", X_train.shape, X_test.shape)

# helper functions
def winsorize_df(df_local, lower_q=0.01, upper_q=0.99):
    df2 = df_local.copy()
    for c in df2.columns:
        lo = df2[c].quantile(lower_q)
        hi = df2[c].quantile(upper_q)
        df2[c] = df2[c].clip(lo, hi)
    return df2

def log1p_safe(df_local):
    df2 = df_local.copy()
    for c in df2.columns:
        if (df2[c] >= 0).all():
            df2[c] = np.log1p(df2[c])
        else:
            # shift to make non-negative
            minv = df2[c].min()
            shift = -minv
            df2[c] = np.log1p(df2[c] + shift)
    return df2

# Preprocessing variants to compare
variants = {}
# Variant A: baseline - median impute + StandardScaler
X_train_fill = X_train.fillna(X_train.median())
X_test_fill = X_test.fillna(X_train.median())  # use train's medians
scaler = StandardScaler().fit(X_train_fill)
variants['baseline'] = (scaler.transform(X_train_fill), scaler.transform(X_test_fill))

# Variant B: Winsorize 1%-99%
X_train_win = winsorize_df(X_train.fillna(X_train.median()), 0.01, 0.99)
X_test_win = X_test.fillna(X_train.median())
# apply same clipping bounds from train
for c in X_train.columns:
    lo = X_train[c].quantile(0.01)
    hi = X_train[c].quantile(0.99)
    X_test_win[c] = X_test_win[c].clip(lo, hi)
scaler_win = StandardScaler().fit(X_train_win)
variants['winsorize_1_99'] = (scaler_win.transform(X_train_win), scaler_win.transform(X_test_win))

# Variant C: log1p on non-negative (or shifted) features
X_train_log = log1p_safe(X_train.fillna(X_train.median()))
X_test_log = X_test.fillna(X_train.median())
# ensure same transformation: if shifted during train, better to compute shift mapping - for quick script we do same operation on test (may differ minimally)
scaler_log = StandardScaler().fit(X_train_log)
variants['log1p'] = (scaler_log.transform(X_train_log), scaler_log.transform(X_test_log))

# Visualization: PCA 2D, t-SNE and UMAP (subsample for plotting if large)
def save_pca_plot(Xs, ys, name_prefix):
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca2.fit_transform(Xs)
    dfp = pd.DataFrame(Z, columns=['PC1','PC2'])
    dfp['y'] = ys
    plt.figure(figsize=(6,5))
    sns.scatterplot(x='PC1', y='PC2', hue='y', data=dfp, palette='coolwarm', s=10, alpha=0.7)
    plt.title(f"PCA 2D - {name_prefix}")
    plt.savefig(os.path.join(OUTDIR, f"pca2_{name_prefix}.png"), dpi=150)
    plt.close()

def save_tsne_plot(Xs, ys, name_prefix, perplexity=30):
    n = Xs.shape[0]
    sample_idx = np.arange(n)
    if n > 5000:
        sample_idx = np.random.choice(n, 5000, replace=False)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init='pca', perplexity=perplexity)
    Z = tsne.fit_transform(Xs[sample_idx])
    dfp = pd.DataFrame(Z, columns=['tSNE1','tSNE2'])
    dfp['y'] = ys[sample_idx]
    plt.figure(figsize=(6,5))
    sns.scatterplot(x='tSNE1', y='tSNE2', hue='y', data=dfp, palette='coolwarm', s=10, alpha=0.7)
    plt.title(f"t-SNE 2D - {name_prefix}")
    plt.savefig(os.path.join(OUTDIR, f"tsne2_{name_prefix}.png"), dpi=150)
    plt.close()

def save_umap_plot(Xs, ys, name_prefix, n_neighbors=15, min_dist=0.1):
    try:
        import umap
    except Exception:
        print("umap not installed; skipping UMAP plot.")
        return
    n = Xs.shape[0]
    idx = np.arange(n)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=RANDOM_STATE)
    Z = reducer.fit_transform(Xs[idx])
    dfp = pd.DataFrame(Z, columns=['UMAP1','UMAP2'])
    dfp['y'] = ys[idx]
    plt.figure(figsize=(6,5))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='y', data=dfp, palette='coolwarm', s=10, alpha=0.7)
    plt.title(f"UMAP 2D - {name_prefix}")
    plt.savefig(os.path.join(OUTDIR, f"umap2_{name_prefix}.png"), dpi=150)
    plt.close()

print("\nSaving visualizations (PCA, t-SNE, UMAP if available) for each preprocessing variant ...")
for name, (Xtr_scaled, Xte_scaled) in variants.items():
    Xvis = np.vstack([Xtr_scaled, Xte_scaled])
    yvis = np.concatenate([y_train.values, y_test.values])
    save_pca_plot(Xvis, yvis, name)
    save_tsne_plot(Xvis, yvis, name)
    save_umap_plot(Xvis, yvis, name)

# Modeling: Logistic (linear) vs Tree (xgboost if available else RF)
print("\nPreparing models for CV comparison ...")
log_clf = LogisticRegression(solver='saga', penalty='l2', max_iter=5000, class_weight='balanced', random_state=RANDOM_STATE)
try:
    from xgboost import XGBClassifier
    tree_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)
    tree_name = 'XGBoost'
except Exception:
    tree_clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    tree_name = 'RandomForest'

scoring = {'roc_auc': 'roc_auc', 'average_precision': 'average_precision'}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results_summary = {}

print("\nRunning CV for each preprocessing variant and each model (this may take some time)...")
for name, (Xtr_scaled, Xte_scaled) in variants.items():
    Xcv = np.vstack([Xtr_scaled, Xte_scaled])
    ycv = np.concatenate([y_train.values, y_test.values])
    res = {}
    for mdl_name, mdl in [('Logistic', log_clf), (tree_name, tree_clf)]:
        cvres = cross_validate(mdl, Xcv, ycv, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        roc_mean = np.mean(cvres['test_roc_auc']); roc_std = np.std(cvres['test_roc_auc'])
        ap_mean = np.mean(cvres['test_average_precision']); ap_std = np.std(cvres['test_average_precision'])
        res[mdl_name] = {'roc_mean': roc_mean, 'roc_std': roc_std, 'ap_mean': ap_mean, 'ap_std': ap_std}
        print(f"Variant={name:15s} Model={mdl_name:12s} ROC AUC={roc_mean:.4f}±{roc_std:.4f}  AP={ap_mean:.4f}±{ap_std:.4f}")
    results_summary[name] = res

# Save summary csv
rows = []
for variant, d in results_summary.items():
    for mname, metrics in d.items():
        rows.append({
            'variant': variant,
            'model': mname,
            'roc_mean': metrics['roc_mean'],
            'roc_std': metrics['roc_std'],
            'ap_mean': metrics['ap_mean'],
            'ap_std': metrics['ap_std']
        })
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "model_comparison_summary.csv"), index=False)
print("\nSummary saved to artifacts/model_comparison_summary.csv")

# Quick decision hint:
print("\nQuick decision hints:")
# Compare logistic vs tree on baseline
base = results_summary.get('baseline', {})
if base:
    if base.get('Logistic') and base.get(tree_name):
        diff = base[tree_name]['roc_mean'] - base['Logistic']['roc_mean']
        print(f"Baseline ROC difference ({tree_name} - Logistic): {diff:.4f}")
        if diff < 0.03:
            print(" -> 差距较小，线性模型（Logistic/LDA）可能是合适的选择（可优先可解释性）。")
        else:
            print(" -> 差距较大，数据可能存在非线性结构，建议使用树/GBM 或进行非线性特征工程。")

print("\n完成。请查看 artifacts/ 目录下的图片（PCA/t-SNE/UMAP）和 model_comparison_summary.csv，来判断是否采用线性模型与是否需要 Winsorize。")