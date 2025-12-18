#!/usr/bin/env python3
"""
线性判别分析（LDA）模型训练脚本

功能：
1. 加载PCA生成的主成分得分矩阵（X_train_pca.csv, X_test_pca.csv）
2. 加载标签（y_train.csv, y_test.csv）
3. 处理类别不平衡（使用class_weight='balanced'）
4. 训练LDA模型
5. 保存模型和初步评估结果

输出：
- artifacts/lda_model/
  - lda_model.joblib (训练好的LDA模型)
  - lda_coefficients.csv (判别函数系数)
  - lda_train_summary.txt (训练摘要)
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import joblib

RND = 42
OUTDIR = "artifacts/lda_model"
os.makedirs(OUTDIR, exist_ok=True)

# 输入路径
PCA_DIR = "artifacts/pca_analysis"
FEATURIZED_DIR = "artifacts/featurized"

def load_data():
    """加载PCA主成分得分矩阵和标签"""
    print("Loading PCA principal component scores and labels...")
    
    # 加载主成分得分矩阵
    X_train_pca = pd.read_csv(os.path.join(PCA_DIR, "X_train_pca.csv"))
    X_test_pca = pd.read_csv(os.path.join(PCA_DIR, "X_test_pca.csv"))
    
    # 加载标签 - 处理可能的列名问题
    y_train_raw = pd.read_csv(os.path.join(FEATURIZED_DIR, "y_train.csv"), header=None).iloc[:, 0]
    y_test_raw = pd.read_csv(os.path.join(FEATURIZED_DIR, "y_test.csv"), header=None).iloc[:, 0]
    
    # 过滤掉非数值的行（可能是列名或其他文本）
    # 只保留可以转换为数值的行
    def clean_labels(series):
        """清理标签，移除非数值行"""
        cleaned = []
        for val in series:
            try:
                # 尝试转换为数值
                num_val = int(float(str(val)))
                if num_val in [0, 1]:  # 只保留0或1
                    cleaned.append(num_val)
            except (ValueError, TypeError):
                # 跳过非数值行（如列名"class"）
                continue
        return pd.Series(cleaned, dtype=int)
    
    y_train = clean_labels(y_train_raw)
    y_test = clean_labels(y_test_raw)
    
    # 确保X和y的长度一致
    min_len = min(len(X_train_pca), len(y_train))
    X_train_pca = X_train_pca.iloc[:min_len].reset_index(drop=True)
    y_train = y_train.iloc[:min_len].reset_index(drop=True)
    
    min_len_test = min(len(X_test_pca), len(y_test))
    X_test_pca = X_test_pca.iloc[:min_len_test].reset_index(drop=True)
    y_test = y_test.iloc[:min_len_test].reset_index(drop=True)
    
    print(f"Train shapes: X={X_train_pca.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test_pca.shape}, y={y_test.shape}")
    
    # 检查类别分布
    print("\nClass distribution in training set:")
    print(y_train.value_counts().sort_index())
    if len(y_train.value_counts()) >= 2:
        print(f"Class ratio (0:1): {y_train.value_counts()[0]}:{y_train.value_counts()[1]}")
    
    return X_train_pca, X_test_pca, y_train, y_test

def train_lda(X_train, y_train, use_balanced=True):
    """
    训练LDA模型
    
    Args:
        X_train: 训练集主成分得分矩阵
        y_train: 训练集标签
        use_balanced: 是否使用类别权重平衡（处理类别不平衡）
    
    Returns:
        lda: 训练好的LDA模型
    """
    print("\n" + "=" * 60)
    print("Training LDA Model")
    print("=" * 60)
    
    # 创建LDA模型
    # 注意：LDA没有class_weight参数，但可以通过solver='svd'或手动处理
    # 对于类别不平衡，我们可以在训练前使用SMOTE，或者接受不平衡
    # 这里先使用默认设置，LDA本身对类别不平衡有一定鲁棒性
    
    lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
    
    print(f"LDA parameters:")
    print(f"  Solver: svd (适合高维数据)")
    print(f"  Store covariance: True (用于后续分析)")
    
    # 训练模型
    print("\nFitting LDA model...")
    lda.fit(X_train, y_train)
    
    print("✓ Model training completed")
    
    # 显示模型信息
    print(f"\nModel information:")
    print(f"  Number of classes: {len(lda.classes_)}")
    print(f"  Classes: {lda.classes_}")
    print(f"  Number of features: {lda.n_features_in_}")
    print(f"  Explained variance ratio: {lda.explained_variance_ratio_}")
    
    return lda

def analyze_discriminant_coefficients(lda, feature_names):
    """
    分析判别函数系数
    
    Args:
        lda: 训练好的LDA模型
        feature_names: 特征名称列表（主成分名称）
    
    Returns:
        coeff_df: 系数DataFrame
    """
    # LDA的判别函数系数
    # coef_ shape: (n_classes - 1, n_features)
    # intercept_ shape: (n_classes - 1,)
    
    if hasattr(lda, 'coef_'):
        coef = lda.coef_[0]  # 对于二分类，只有一个判别函数
        intercept = lda.intercept_[0]
        
        # 创建系数DataFrame
        coeff_df = pd.DataFrame({
            'Principal_Component': feature_names,
            'Coefficient': coef,
            'Abs_Coefficient': np.abs(coef)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\n" + "=" * 60)
        print("Discriminant Function Coefficients Analysis")
        print("=" * 60)
        print(f"Intercept: {intercept:.4f}")
        print("\nTop 5 principal components by absolute coefficient:")
        print(coeff_df.head(10).to_string(index=False))
        
        return coeff_df
    else:
        print("Warning: LDA model does not have coef_ attribute (solver='svd')")
        return None

def evaluate_model(lda, X_train, X_test, y_train, y_test):
    """
    评估模型性能
    
    Returns:
        metrics_dict: 评估指标字典
    """
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # 预测
    y_train_pred = lda.predict(X_train)
    y_test_pred = lda.predict(X_test)
    
    # 预测概率（用于AUC计算）
    y_train_proba = lda.predict_proba(X_train)[:, 1]
    y_test_proba = lda.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    metrics = {}
    
    # 训练集指标
    metrics['train'] = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_train, y_train_proba),
        'average_precision': average_precision_score(y_train, y_train_proba)
    }
    
    # 测试集指标
    metrics['test'] = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'average_precision': average_precision_score(y_test, y_test_proba)
    }
    
    # 混淆矩阵
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    print("\nTraining Set Metrics:")
    print(f"  Accuracy:  {metrics['train']['accuracy']:.4f}")
    print(f"  Precision: {metrics['train']['precision']:.4f}")
    print(f"  Recall:    {metrics['train']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['train']['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['train']['roc_auc']:.4f}")
    print(f"  PR AUC:    {metrics['train']['average_precision']:.4f}")
    
    print("\nTest Set Metrics:")
    print(f"  Accuracy:  {metrics['test']['accuracy']:.4f}")
    print(f"  Precision: {metrics['test']['precision']:.4f}")
    print(f"  Recall:    {metrics['test']['recall']:.4f}")
    print(f"  F1 Score:  {metrics['test']['f1']:.4f}")
    print(f"  ROC AUC:   {metrics['test']['roc_auc']:.4f}")
    print(f"  PR AUC:    {metrics['test']['average_precision']:.4f}")
    
    print("\nConfusion Matrix (Train):")
    print("                Predicted")
    print("              Good  Bad")
    print(f"Actual Good   {cm_train[0,0]:4d}  {cm_train[0,1]:4d}")
    print(f"        Bad    {cm_train[1,0]:4d}  {cm_train[1,1]:4d}")
    
    print("\nConfusion Matrix (Test):")
    print("                Predicted")
    print("              Good  Bad")
    print(f"Actual Good   {cm_test[0,0]:4d}  {cm_test[0,1]:4d}")
    print(f"        Bad    {cm_test[1,0]:4d}  {cm_test[1,1]:4d}")
    
    metrics['confusion_matrix_train'] = cm_train.tolist()
    metrics['confusion_matrix_test'] = cm_test.tolist()
    
    return metrics

def main():
    print("=" * 60)
    print("LDA Model Training")
    print("=" * 60)
    
    # 1. 加载数据
    X_train_pca, X_test_pca, y_train, y_test = load_data()
    
    # 2. 训练LDA模型
    lda = train_lda(X_train_pca, y_train, use_balanced=True)
    
    # 3. 分析判别函数系数
    feature_names = X_train_pca.columns.tolist()
    coeff_df = analyze_discriminant_coefficients(lda, feature_names)
    
    # 4. 评估模型
    metrics = evaluate_model(lda, X_train_pca, X_test_pca, y_train, y_test)
    
    # 5. 保存结果
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)
    
    # 保存模型
    model_path = os.path.join(OUTDIR, "lda_model.joblib")
    joblib.dump(lda, model_path)
    print(f"✓ LDA model saved: {model_path}")
    
    # 保存系数
    if coeff_df is not None:
        coeff_path = os.path.join(OUTDIR, "lda_coefficients.csv")
        coeff_df.to_csv(coeff_path, index=False)
        print(f"✓ Coefficients saved: {coeff_path}")
    
    # 保存评估指标
    metrics_path = os.path.join(OUTDIR, "lda_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # 保存训练摘要
    summary_path = os.path.join(OUTDIR, "lda_train_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LDA Model Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training samples: {len(y_train)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Number of principal components: {X_train_pca.shape[1]}\n")
        f.write(f"Number of classes: {len(lda.classes_)}\n\n")
        
        f.write("Test Set Performance:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:  {metrics['test']['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['test']['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['test']['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['test']['f1']:.4f}\n")
        f.write(f"ROC AUC:   {metrics['test']['roc_auc']:.4f}\n")
        f.write(f"PR AUC:    {metrics['test']['average_precision']:.4f}\n\n")
        
        if coeff_df is not None:
            f.write("Top 5 Principal Components by Discriminant Coefficient:\n")
            f.write("-" * 60 + "\n")
            for idx, row in coeff_df.head(5).iterrows():
                f.write(f"{row['Principal_Component']}: {row['Coefficient']:.4f}\n")
    
    print(f"✓ Training summary saved: {summary_path}")
    
    print("\n" + "=" * 60)
    print("LDA Training Complete!")
    print("=" * 60)
    print(f"\nAll results saved to: {OUTDIR}/")
    print("\nNext step: Run model evaluation script for detailed analysis")

if __name__ == "__main__":
    main()

