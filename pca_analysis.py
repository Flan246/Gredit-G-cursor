#!/usr/bin/env python3
"""
主成分分析（PCA）实施与解读脚本

功能：
1. 加载特征工程后的数据（artifacts/featurized/）
2. 分离数值特征和one-hot编码特征
3. 对数值特征进行PCA分析
4. 确定最佳主成分数量（碎石图、累计方差贡献率）
5. 分析主成分载荷矩阵，为每个主成分命名
6. 生成主成分得分矩阵（作为LDA模型的输入）
7. 保存所有结果和可视化图表

输出：
- artifacts/pca_analysis/
  - pca_transformer.joblib (PCA转换器)
  - X_train_pca.csv (训练集主成分得分)
  - X_test_pca.csv (测试集主成分得分)
  - pca_loadings.csv (载荷矩阵)
  - pca_variance_explained.csv (方差解释表)
  - scree_plot.png (碎石图)
  - cumulative_variance_plot.png (累计方差贡献率图)
  - pca_readme.txt (分析报告)
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
import matplotlib.font_manager as fm

# 设置中文字体 - 自动检测可用字体（增强版）
def setup_chinese_font(force_refresh=False):
    """
    设置中文字体，如果找不到则使用英文标签
    
    Args:
        force_refresh: 是否强制刷新字体缓存
    """
    # 如果需要，清除字体缓存
    if force_refresh:
        try:
            import matplotlib
            matplotlib.font_manager._rebuild()
            print("✓ 已刷新matplotlib字体缓存")
        except:
            pass
    
    # Windows系统常见中文字体（按优先级排序）
    chinese_fonts = [
        'Microsoft YaHei',      # 微软雅黑（Windows 7+）
        'SimHei',              # 黑体
        'SimSun',              # 宋体
        'KaiTi',               # 楷体
        'FangSong',            # 仿宋
        'Microsoft JhengHei',  # 微软正黑体
    ]
    # Mac系统常见中文字体
    chinese_fonts.extend(['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Hiragino Sans GB'])
    # Linux系统常见中文字体
    chinese_fonts.extend(['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback'])
    
    # 获取系统所有可用字体（包括完整路径信息）
    available_fonts = {}
    for font in fm.fontManager.ttflist:
        font_name = font.name
        if font_name not in available_fonts:
            available_fonts[font_name] = font.fname  # 保存字体文件路径
    
    # 查找第一个可用的中文字体
    font_found = None
    font_path = None
    
    for font in chinese_fonts:
        # 精确匹配
        if font in available_fonts:
            font_found = font
            font_path = available_fonts[font]
            break
        # 模糊匹配（处理字体名称变体）
        for available_name in available_fonts.keys():
            if font.lower() in available_name.lower() or available_name.lower() in font.lower():
                font_found = available_name
                font_path = available_fonts[available_name]
                break
        if font_found:
            break
    
    if font_found:
        # 更强制性的字体设置
        plt.rcParams['font.sans-serif'] = [font_found]
        plt.rcParams['axes.unicode_minus'] = False
        # 尝试直接设置字体属性
        try:
            from matplotlib import font_manager
            prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = 'sans-serif'
            print(f"✓ 使用中文字体: {font_found} (路径: {font_path})")
        except:
            print(f"✓ 使用中文字体: {font_found}")
        return True
    else:
        print("⚠ 未找到中文字体，将使用英文标签")
        print(f"   可用字体示例（前10个）: {list(available_fonts.keys())[:10]}")
        return False

# 执行字体设置
# 方案选择：
# 1. USE_CHINESE = False: 完全使用英文（最保险，推荐）
# 2. USE_CHINESE = True: 尝试使用中文（如果仍有方框，建议用方案1）

USE_CHINESE = False  # 改为 True 可以尝试中文，但可能仍有方框问题

if USE_CHINESE:
    # 尝试设置中文字体（带强制刷新）
    USE_CHINESE = setup_chinese_font(force_refresh=True)
    if not USE_CHINESE:
        print("⚠ 字体设置失败，自动切换到英文标签")
else:
    print("ℹ 使用英文标签以确保图表正常显示（无方框问题）")

RND = 42
OUTDIR = "artifacts/pca_analysis"
os.makedirs(OUTDIR, exist_ok=True)
sns.set_style("whitegrid")

# 输入路径
FEATURIZED_DIR = "artifacts/featurized"

def load_label_series(csv_path: str) -> pd.Series:
    """
    Robustly load a 1D label CSV.
    Supports both formats:
    - with header (e.g., column name 'class')
    - without header (single-column, pure values)
    """
    # First try default header=0
    df = pd.read_csv(csv_path)
    if df.shape[1] == 1:
        col0 = str(df.columns[0]).strip()
        # If the "header" looks like a real label name, keep it.
        # If it looks like a digit (common when the file actually has no header),
        # fall back to header=None.
        if col0.lower() in {"class", "label", "target", "y"} or not col0.isdigit():
            return df.iloc[:, 0]
    # Fallback: no header
    df2 = pd.read_csv(csv_path, header=None)
    return df2.iloc[:, 0]

def load_featurized_data():
    """加载特征工程后的数据"""
    print("Loading featurized data...")
    X_train = pd.read_csv(os.path.join(FEATURIZED_DIR, "X_train_feat.csv"))
    X_test = pd.read_csv(os.path.join(FEATURIZED_DIR, "X_test_feat.csv"))
    y_train = load_label_series(os.path.join(FEATURIZED_DIR, "y_train.csv"))
    y_test = load_label_series(os.path.join(FEATURIZED_DIR, "y_test.csv"))
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def separate_numeric_and_onehot(X_train, X_test):
    """
    分离数值特征和one-hot编码特征
    
    One-hot特征通常包含下划线分隔的列名（如 'purpose_business'）
    数值特征包括：原始数值特征、工程特征、序数编码特征
    """
    # 识别one-hot特征：包含下划线且值只有0/1的特征
    onehot_cols = []
    numeric_cols = []
    
    for col in X_train.columns:
        # 检查是否为one-hot特征（值只有0和1，且列名包含下划线分隔）
        unique_vals = set(X_train[col].dropna().unique())
        if unique_vals <= {0, 1, 0.0, 1.0} and '_' in col:
            # 进一步检查：one-hot特征通常格式为 "category_value"
            parts = col.split('_')
            if len(parts) >= 2:
                onehot_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            numeric_cols.append(col)
    
    print(f"\n分离结果：")
    print(f"  数值特征数量: {len(numeric_cols)}")
    print(f"  One-hot特征数量: {len(onehot_cols)}")
    
    X_train_numeric = X_train[numeric_cols].copy()
    X_test_numeric = X_test[numeric_cols].copy()
    X_train_onehot = X_train[onehot_cols].copy() if onehot_cols else pd.DataFrame()
    X_test_onehot = X_test[onehot_cols].copy() if onehot_cols else pd.DataFrame()
    
    return X_train_numeric, X_test_numeric, X_train_onehot, X_test_onehot, numeric_cols, onehot_cols

def determine_optimal_components(pca, variance_threshold=0.95, min_components=2, max_components=None):
    """
    确定最佳主成分数量
    
    策略：
    1. 累计方差贡献率 >= variance_threshold (默认95%)
    2. 至少保留min_components个主成分
    3. 不超过max_components（如果指定）
    """
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # 方法1：累计方差贡献率 >= threshold
    n_components_variance = np.argmax(cumsum_variance >= variance_threshold) + 1
    
    # 方法2：至少保留min_components
    n_components = max(min_components, n_components_variance)
    
    # 方法3：如果指定了max_components，不超过它
    if max_components is not None:
        n_components = min(n_components, max_components)
    
    return n_components, cumsum_variance

def plot_scree_plot(pca, n_components_to_show=20, save_path=None):
    """绘制碎石图（Scree Plot）"""
    n_show = min(n_components_to_show, len(pca.explained_variance_ratio_))
    components = range(1, n_show + 1)
    variance = pca.explained_variance_ratio_[:n_show]
    
    # 根据字体支持情况选择标签
    if USE_CHINESE:
        xlabel = '主成分编号'
        ylabel = '方差解释比例'
        title = 'PCA 碎石图 (Scree Plot)'
    else:
        xlabel = 'Principal Component Number'
        ylabel = 'Variance Explained Ratio'
        title = 'PCA Scree Plot'
    
    plt.figure(figsize=(10, 6))
    plt.plot(components, variance, 'bo-', linewidth=2, markersize=8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(components)
    
    # 添加数值标签
    for i, v in enumerate(variance):
        plt.text(i+1, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scree plot to {save_path}")
    plt.close()

def plot_cumulative_variance(cumsum_variance, n_components_optimal=None, save_path=None):
    """绘制累计方差贡献率图"""
    n_show = len(cumsum_variance)
    components = range(1, n_show + 1)
    
    # 根据字体支持情况选择标签
    if USE_CHINESE:
        threshold_label = '95% 阈值'
        optimal_label = f'最优主成分数: {n_components_optimal}'
        xlabel = '主成分数量'
        ylabel = '累计方差解释比例 (%)'
        title = 'PCA 累计方差贡献率'
    else:
        threshold_label = '95% Threshold'
        optimal_label = f'Optimal Components: {n_components_optimal}'
        xlabel = 'Number of Principal Components'
        ylabel = 'Cumulative Variance Explained (%)'
        title = 'PCA Cumulative Variance Explained'
    
    plt.figure(figsize=(10, 6))
    plt.plot(components, cumsum_variance * 100, 'ro-', linewidth=2, markersize=6)
    
    # 标记95%线
    plt.axhline(y=95, color='g', linestyle='--', linewidth=1.5, label=threshold_label)
    
    # 标记最优主成分数量
    if n_components_optimal:
        plt.axvline(x=n_components_optimal, color='orange', linestyle='--', 
                   linewidth=1.5, label=optimal_label)
        variance_pct = cumsum_variance[n_components_optimal-1]*100
        plt.plot(n_components_optimal, variance_pct, 
                'o', color='orange', markersize=12, label=f'{variance_pct:.1f}%')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cumulative variance plot to {save_path}")
    plt.close()

def interpret_principal_components(pca, feature_names, n_components, top_n=5):
    """
    解释主成分：分析载荷矩阵，为每个主成分命名
    
    返回：
    - component_names: 主成分名称列表
    - loadings_df: 载荷矩阵DataFrame
    - interpretations: 每个主成分的解释文本
    """
    loadings = pca.components_[:n_components]  # shape: (n_components, n_features)
    loadings_df = pd.DataFrame(loadings, 
                              columns=feature_names,
                              index=[f'PC{i+1}' for i in range(n_components)])
    
    interpretations = []
    component_names = []
    
    for i in range(n_components):
        pc_loadings = loadings[i]
        # 获取绝对值最大的top_n个特征
        top_indices = np.argsort(np.abs(pc_loadings))[-top_n:][::-1]
        top_features = [feature_names[idx] for idx in top_indices]
        top_loadings = pc_loadings[top_indices]
        
        # 分析正负载荷
        positive_features = [f for f, l in zip(top_features, top_loadings) if l > 0]
        negative_features = [f for f, l in zip(top_features, top_loadings) if l < 0]
        
        # 根据载荷特征命名主成分
        # 这里使用简单的启发式规则，你可以根据实际业务含义调整
        if 'credit_amount' in ' '.join(top_features) or 'monthly_payment' in ' '.join(top_features):
            name = "综合消费力"
        elif 'age' in ' '.join(top_features) or 'duration' in ' '.join(top_features):
            name = "客户稳定性"
        elif 'existing_credits' in ' '.join(top_features) or 'installment' in ' '.join(top_features):
            name = "还款能力"
        elif 'savings' in ' '.join(top_features) or 'employment' in ' '.join(top_features):
            name = "经济状况"
        else:
            name = f"综合因子{i+1}"
        
        component_names.append(name)
        
        # 生成解释文本
        expl_text = f"PC{i+1} ({name}):\n"
        expl_text += f"  方差解释比例: {pca.explained_variance_ratio_[i]:.2%}\n"
        expl_text += f"  主要正载荷特征: {', '.join(positive_features[:3])}\n"
        if negative_features:
            expl_text += f"  主要负载荷特征: {', '.join(negative_features[:3])}\n"
        
        interpretations.append(expl_text)
    
    return component_names, loadings_df, interpretations

def main():
    print("=" * 60)
    print("PCA 主成分分析实施")
    print("=" * 60)
    
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_featurized_data()
    
    # 2. 分离数值特征和one-hot特征
    X_train_num, X_test_num, X_train_ohe, X_test_ohe, numeric_cols, onehot_cols = \
        separate_numeric_and_onehot(X_train, X_test)
    
    print(f"\n数值特征列表（前10个）: {numeric_cols[:10]}")
    if onehot_cols:
        print(f"One-hot特征列表（前5个）: {onehot_cols[:5]}")
    
    # 3. 对数值特征进行PCA（先拟合所有主成分，再选择最优数量）
    print("\n" + "=" * 60)
    print("步骤1: 拟合PCA模型（保留所有主成分）")
    print("=" * 60)
    
    # 先拟合所有可能的主成分（最多不超过特征数）
    max_components = min(X_train_num.shape[1], X_train_num.shape[0])
    pca_full = PCA(n_components=max_components, random_state=RND)
    pca_full.fit(X_train_num)
    
    print(f"拟合完成，共 {max_components} 个主成分")
    
    # 4. 确定最佳主成分数量
    print("\n" + "=" * 60)
    print("步骤2: 确定最佳主成分数量")
    print("=" * 60)
    
    n_components_optimal, cumsum_variance = determine_optimal_components(
        pca_full, variance_threshold=0.95, min_components=2, max_components=20
    )
    
    print(f"最优主成分数量: {n_components_optimal}")
    print(f"累计方差解释比例: {cumsum_variance[n_components_optimal-1]:.2%}")
    
    # 5. 重新拟合PCA（使用最优主成分数量）
    print("\n" + "=" * 60)
    print("步骤3: 使用最优主成分数量重新拟合PCA")
    print("=" * 60)
    
    pca = PCA(n_components=n_components_optimal, random_state=RND)
    X_train_pca = pca.fit_transform(X_train_num)
    X_test_pca = pca.transform(X_test_num)
    
    print(f"训练集主成分得分形状: {X_train_pca.shape}")
    print(f"测试集主成分得分形状: {X_test_pca.shape}")
    
    # 6. 绘制可视化图表
    print("\n" + "=" * 60)
    print("步骤4: 生成可视化图表")
    print("=" * 60)
    
    plot_scree_plot(pca_full, n_components_to_show=min(20, max_components),
                   save_path=os.path.join(OUTDIR, "scree_plot.png"))
    
    plot_cumulative_variance(cumsum_variance, n_components_optimal=n_components_optimal,
                           save_path=os.path.join(OUTDIR, "cumulative_variance_plot.png"))
    
    # 7. 分析主成分载荷矩阵
    print("\n" + "=" * 60)
    print("步骤5: 分析主成分载荷矩阵并命名")
    print("=" * 60)
    
    component_names, loadings_df, interpretations = interpret_principal_components(
        pca, numeric_cols, n_components_optimal, top_n=5
    )
    
    # 打印解释
    for interp in interpretations:
        print(interp)
    
    # 8. 保存结果
    print("\n" + "=" * 60)
    print("步骤6: 保存结果")
    print("=" * 60)
    
    # 保存PCA转换器
    joblib.dump(pca, os.path.join(OUTDIR, "pca_transformer.joblib"))
    print(f"✓ PCA转换器已保存: {OUTDIR}/pca_transformer.joblib")
    
    # 保存主成分得分矩阵
    pca_train_df = pd.DataFrame(X_train_pca, 
                                columns=[f'PC{i+1}_{name}' for i, name in enumerate(component_names)])
    pca_test_df = pd.DataFrame(X_test_pca,
                               columns=[f'PC{i+1}_{name}' for i, name in enumerate(component_names)])
    
    pca_train_df.to_csv(os.path.join(OUTDIR, "X_train_pca.csv"), index=False)
    pca_test_df.to_csv(os.path.join(OUTDIR, "X_test_pca.csv"), index=False)
    print(f"✓ 主成分得分矩阵已保存: X_train_pca.csv, X_test_pca.csv")
    
    # 保存载荷矩阵
    loadings_df.to_csv(os.path.join(OUTDIR, "pca_loadings.csv"))
    print(f"✓ 载荷矩阵已保存: pca_loadings.csv")
    
    # 保存方差解释表
    variance_df = pd.DataFrame({
        '主成分': [f'PC{i+1}' for i in range(n_components_optimal)],
        '主成分名称': component_names,
        '方差解释比例': pca.explained_variance_ratio_,
        '累计方差解释比例': cumsum_variance[:n_components_optimal]
    })
    variance_df.to_csv(os.path.join(OUTDIR, "pca_variance_explained.csv"), index=False)
    print(f"✓ 方差解释表已保存: pca_variance_explained.csv")
    
    # 保存分析报告
    with open(os.path.join(OUTDIR, "pca_readme.txt"), "w", encoding="utf-8") as f:
        f.write("PCA 主成分分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"原始特征数量: {len(numeric_cols)}\n")
        f.write(f"最优主成分数量: {n_components_optimal}\n")
        f.write(f"累计方差解释比例: {cumsum_variance[n_components_optimal-1]:.2%}\n\n")
        f.write("主成分解释:\n")
        f.write("-" * 60 + "\n")
        for interp in interpretations:
            f.write(interp + "\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("输出文件说明:\n")
        f.write("- pca_transformer.joblib: PCA转换器（用于后续数据转换）\n")
        f.write("- X_train_pca.csv: 训练集主成分得分矩阵（LDA模型输入）\n")
        f.write("- X_test_pca.csv: 测试集主成分得分矩阵（LDA模型输入）\n")
        f.write("- pca_loadings.csv: 主成分载荷矩阵（用于解释主成分含义）\n")
        f.write("- pca_variance_explained.csv: 方差解释表\n")
        f.write("- scree_plot.png: 碎石图\n")
        f.write("- cumulative_variance_plot.png: 累计方差贡献率图\n")
    
    print(f"✓ 分析报告已保存: pca_readme.txt")
    
    print("\n" + "=" * 60)
    print("PCA分析完成！")
    print("=" * 60)
    print(f"\n所有结果已保存到: {OUTDIR}/")
    print(f"\n下一步: 使用 X_train_pca.csv 和 X_test_pca.csv 进行LDA模型训练")

if __name__ == "__main__":
    main()

