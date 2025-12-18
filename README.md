### This is a collegue's final homework.
I search dataset on the Internet to do multi-analysis and I choose the credit-G database in the openML. 
I can use it to do PCA and LCA and I try my best to finish it.
Sorry about mistakes I made and pride I take.
Hope I can do better next time.

---

### 复现说明（简要）

#### 处理顺序（建议）
- winsorize → featurize → scale → PCA →（可选：LDA/其它模型）

#### 关键产物（均在 `artifacts/` 下）
- **Featurization**：`artifacts/featurized/`（含 `X_train_feat.csv`、`X_test_feat.csv`、`y_train.csv`、`y_test.csv` 以及 `encoders/`）
- **PCA**：`artifacts/pca_analysis/`（含 `X_train_pca.csv`、`X_test_pca.csv`、`pca_transformer.joblib`、载荷/方差解释与图）
- **描述性统计/质量**：`artifacts/summary_stats.csv`、`artifacts/categorical_counts.csv`、`artifacts/missing_summary.csv`

#### 运行脚本（示例）
- 安装依赖：`pip install -r requirements.txt`
- 特征工程：`python3 featurize.py`
- PCA：`python3 pca_analysis.py`
- （可选）LDA：`python3 lda_train.py`
