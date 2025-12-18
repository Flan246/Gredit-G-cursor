PCA 主成分分析报告
============================================================

原始特征数量: 17
最优主成分数量: 11
累计方差解释比例: 97.54%

主成分解释:
------------------------------------------------------------
PC1 (综合消费力):
  方差解释比例: 29.80%
  主要正载荷特征: credit_amount, credit_amount_wins, credit_amount_log

PC2 (客户稳定性):
  方差解释比例: 13.90%
  主要正载荷特征: many_dependents_flag, num_dependents, installment_rel
  主要负载荷特征: duration, installment_commitment

PC3 (还款能力):
  方差解释比例: 11.51%
  主要正载荷特征: many_dependents_flag, num_dependents, installment_commitment
  主要负载荷特征: installment_rel

PC4 (客户稳定性):
  方差解释比例: 8.56%
  主要正载荷特征: age, residence_since, employment_ord
  主要负载荷特征: many_dependents_flag, num_dependents

PC5 (还款能力):
  方差解释比例: 6.16%
  主要正载荷特征: existing_credits
  主要负载荷特征: savings_status_ord, credit_per_existing, job_ord

PC6 (还款能力):
  方差解释比例: 5.71%
  主要正载荷特征: savings_status_ord, existing_credits, high_amount_flag
  主要负载荷特征: job_ord, installment_commitment

PC7 (还款能力):
  方差解释比例: 5.43%
  主要正载荷特征: job_ord, savings_status_ord, existing_credits
  主要负载荷特征: high_amount_flag, residence_since

PC8 (客户稳定性):
  方差解释比例: 5.04%
  主要正载荷特征: high_amount_flag, job_ord, installment_commitment
  主要负载荷特征: residence_since, duration

PC9 (客户稳定性):
  方差解释比例: 4.40%
  主要正载荷特征: age
  主要负载荷特征: residence_since, high_amount_flag, employment_ord

PC10 (还款能力):
  方差解释比例: 4.16%
  主要正载荷特征: employment_ord
  主要负载荷特征: residence_since, job_ord, installment_commitment

PC11 (综合消费力):
  方差解释比例: 2.87%
  主要正载荷特征: installment_commitment, monthly_payment
  主要负载荷特征: job_ord, age, duration


============================================================
输出文件说明:
- pca_transformer.joblib: PCA转换器（用于后续数据转换）
- X_train_pca.csv: 训练集主成分得分矩阵（LDA模型输入）
- X_test_pca.csv: 测试集主成分得分矩阵（LDA模型输入）
- pca_loadings.csv: 主成分载荷矩阵（用于解释主成分含义）
- pca_variance_explained.csv: 方差解释表
- scree_plot.png: 碎石图
- cumulative_variance_plot.png: 累计方差贡献率图
