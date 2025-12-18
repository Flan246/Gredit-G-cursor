EDA Summary
===========

Data shape: (1000, 21)
Target column (normalized): class_norm
Target mapping (inferred): {'good': 0, 'bad': 1}

Numeric columns count: 7
Categorical columns count: 13

Top numeric features by variance:
credit_amount             7.967843e+06
duration                  1.454150e+02
age                       1.294013e+02
installment_commitment    1.251523e+00
residence_since           1.218193e+00
existing_credits          3.336847e-01
num_dependents            1.311061e-01

Missingness (top 20 features):
               feature  missing_count  missing_pct
       checking_status              0          0.0
              duration              0          0.0
        credit_history              0          0.0
               purpose              0          0.0
         credit_amount              0          0.0
        savings_status              0          0.0
            employment              0          0.0
installment_commitment              0          0.0
       personal_status              0          0.0
         other_parties              0          0.0
       residence_since              0          0.0
    property_magnitude              0          0.0
                   age              0          0.0
   other_payment_plans              0          0.0
               housing              0          0.0
      existing_credits              0          0.0
                   job              0          0.0
        num_dependents              0          0.0
         own_telephone              0          0.0
        foreign_worker              0          0.0

Files saved in artifacts/: summary_stats.csv, missing_summary.csv, categorical_counts.csv, hist_features_*.png, box_by_target_topvars.png, corr_numeric.png, pairplot*.png (if generated), target_dist.png
