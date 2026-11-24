# Intelligent Financial Fraud Detection

An end-to-end machine learning project that detects potentially fraudulent credit card transactions. The project covers data cleaning, SQL-based analysis, exploratory data analysis, imbalanced classification, model comparison, threshold selection, hyperparameter tuning, and model explainability with SHAP.

> **Project status:** Model development and final holdout testing are complete. API and dashboard deployment are planned future work and are not yet implemented.

## 1. Project objectives

- Explore transaction data and understand the imbalance between legitimate and fraudulent transactions.
- Clean the dataset and store the processed records in SQLite.
- Use SQL and Python for data analysis.
- Train and compare Logistic Regression, Random Forest, and XGBoost models.
- Investigate class weighting and SMOTE for imbalanced classification.
- Tune XGBoost hyperparameters using chronological cross-validation.
- Explain model predictions using SHAP.
- Evaluate the selected model on a separate chronological test set.

## 2. Dataset

**Credit Card Fraud Detection** — the dataset published by the Machine Learning Group at Université Libre de Bruxelles (ULB).

Dataset page: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

The dataset contains transactions from European cardholders over two days in September 2013. It has 284,807 rows and 31 columns before cleaning. Most input variables (`V1` to `V28`) are anonymised PCA-transformed features. `Amount` contains the transaction amount, `Time` records elapsed seconds, and `Class` is the target label.

- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction

### Cleaning results

The initial inspection found 1,081 duplicate copies. Removing exact duplicate rows reduced the dataset to 283,726 records.

| Class | Records after cleaning |
|---|---:|
| Legitimate | 283,253 |
| Fraudulent | 473 |
| **Total** | **283,726** |

Fraudulent records represent approximately 0.167% of the cleaned dataset, so accuracy alone is not a suitable measure of model quality.

## 3. Project workflow

1. **Data inspection and cleaning:** inspect column types, missing values, duplicate rows, and class distribution.
2. **SQL integration:** save the cleaned records in a SQLite database and query transaction counts, class percentages, average amounts, and high-value fraud records.
3. **Exploratory data analysis:** examine transaction amounts, elapsed-time patterns, feature correlations, and feature distributions.
4. **Data preparation:** sort transactions by time and create chronological training, validation, and test partitions.
5. **Feature scaling:** fit `StandardScaler` on training data only. Use scaled data for Logistic Regression and SMOTE; use the original feature values for tree-based models.
6. **Model development:** train Logistic Regression, weighted Logistic Regression, Random Forest, and XGBoost variants.
7. **Imbalance experiments:** compare class weighting and SMOTE. Resampling is applied to the training data only.
8. **Hyperparameter tuning:** compare six XGBoost configurations using three expanding-window `TimeSeriesSplit` folds and Average Precision.
9. **Explainability:** use SHAP for overall feature analysis, fraud-only analysis, and an individual transaction explanation.
10. **Final evaluation:** evaluate the selected tuned XGBoost model once on the held-out chronological test set.

## 4. Chronological data split

The cleaned dataset was sorted by `Time` before splitting. This keeps earlier transactions in training and later transactions in validation and testing.

| Partition | Records | Legitimate | Fraudulent | Fraud rate |
|---|---:|---:|---:|---:|
| Training | 198,609 | 198,243 | 366 | 0.184% |
| Validation | 42,560 | 42,505 | 55 | 0.129% |
| Testing | 42,557 | 42,505 | 52 | 0.122% |
| **Total** | **283,726** | **283,253** | **473** | **0.167%** |

The validation set was used for model comparison and threshold selection. The test set was kept separate until final evaluation.

## 5. Models and validation results

The following are the best observed validation F1-scores for the configurations tested. Thresholds were selected using the validation data.

| Model | Selected threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.10 | 72.31% | 85.45% | 78.33% |
| Weighted Logistic Regression | 0.90 | 21.93% | 90.91% | 35.34% |
| Random Forest | 0.30 | 97.67% | 76.36% | 85.71% |
| Original XGBoost | 0.70 | 100.00% | 76.36% | 86.60% |
| Weighted XGBoost | 0.95 | 97.67% | 76.36% | 85.71% |
| XGBoost with SMOTE | 0.90 | 93.48% | 78.18% | 85.15% |
| **Tuned XGBoost** | **0.50** | **100.00%** | **76.36%** | **86.60%** |

The tuned XGBoost model was selected as the final candidate. Its tuned configuration was:

- `n_estimators = 300`
- `max_depth = 4`
- `learning_rate = 0.05`
- `min_child_weight = 3`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `reg_lambda = 5`

The best cross-validation configuration achieved mean Average Precision of 0.7678 across three chronological folds. This cross-validation score is calculated on different folds and should not be directly compared with validation-set Average Precision.

## 6. Final test results

The tuned XGBoost model was evaluated on the held-out test set using the threshold of 0.50 selected before testing.

| Metric | Test result |
|---|---:|
| Accuracy | 99.96% |
| Precision | 95.00% |
| Recall | 73.08% |
| F1-score | 82.61% |
| Average Precision | 0.7543 |
| ROC-AUC | 0.9817 |

Confusion matrix (rows = actual class; columns = predicted class):

```text
[[42503     2]
 [   14    38]]
```

The model correctly detected 38 of the 52 fraudulent test transactions and produced 2 false positives. It missed 14 fraudulent transactions.

The test results are lower than the validation results, particularly for Average Precision. This indicates that ranking performance was weaker in the later test period. The dataset covers only two days and contains 52 fraud cases in the test partition, so results have sampling uncertainty and do not establish performance on future real-world banking transactions.

## 7. SHAP explainability findings

SHAP analysis was performed using the tuned XGBoost model.

### Overall sample

Mean absolute SHAP values from a random sample of 1,000 validation transactions ranked these features highest:

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | V4 | 1.3597 |
| 2 | V14 | 1.0908 |
| 3 | V12 | 0.7105 |
| 4 | V8 | 0.3584 |
| 5 | Amount | 0.3296 |
| 6 | V10 | 0.2988 |

### Fraud-only sample

Among the 55 fraudulent validation transactions, the highest mean absolute SHAP values were:

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | V14 | 3.3859 |
| 2 | V10 | 1.5134 |
| 3 | V4 | 1.1026 |
| 4 | V12 | 0.7576 |
| 5 | V17 | 0.6510 |

These are model explanations, not causal relationships. Since `V1` to `V28` are anonymised PCA features, their original financial meanings are not available in this dataset. The fraud-only ranking is based on 55 cases and should be interpreted cautiously.

## 8. Technology stack

- **Python** — project programming language
- **Pandas and NumPy** — data preparation and numerical operations
- **SQLite and sqlite3** — database storage and SQL queries
- **Matplotlib and Seaborn** — data visualisation
- **Scikit-learn** — preprocessing, baseline models, metrics, and time-series splitting
- **XGBoost** — gradient-boosted decision trees
- **imbalanced-learn** — SMOTE oversampling
- **SHAP** — model explainability
- **VS Code** — development environment

## 9. Repository structure

```text
intelligent-fraud-detection/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       ├── fraud_detection.db
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── scaled/
│
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── final_test_metrics.csv
│   ├── final_test_predictions.csv
│   └── shap_feature_importance.csv
│
├── src/
│   ├── inspect_data.py
│   ├── check_duplicates.py
│   ├── create_database.py
│   ├── run_sql_queries.py
│   ├── eda.py
│   ├── prepare_data.py
│   ├── scale_features.py
│   ├── train_logistic_regression.py
│   ├── train_weighted_logistic.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_xgboost_weighted.py
│   ├── train_xgboost_smote.py
│   ├── tune_xgboost.py
│   ├── explain_shap.py
│   └── final_evaluation.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

The repository structure is a guide; include the files that exist in your final local project. The raw dataset, processed database, and generated model files can be kept outside Git if they make the repository too large. Provide instructions for obtaining the source dataset instead.

## 10. Setup and installation

1. Clone or download this repository.
2. Download `creditcard.csv` from the dataset page and place it in `data/raw/`.
3. Create and activate a virtual environment.
4. Install the Python dependencies.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run scripts from the project root directory. For example:

```bash
python src/inspect_data.py
python src/check_duplicates.py
python src/create_database.py
python src/run_sql_queries.py
python src/eda.py
python src/prepare_data.py
python src/scale_features.py
```

Run the model training, tuning, SHAP, and final evaluation scripts after the prepared datasets have been generated. The scripts should be executed in workflow order because later stages depend on files created by earlier stages.

## 11. Limitations and future work

- The data covers only two days in September 2013 and may not represent current transaction behaviour.
- The number of fraudulent observations is small, particularly in the validation and test partitions.
- Most features are anonymised PCA components, limiting direct business interpretation.
- The data does not include customer-level history, merchant details, or confirmed investigation outcomes.
- The temporal split is useful for testing later observations, but longer historical coverage is needed to evaluate real deployment stability.
- SMOTE creates synthetic training observations that may not represent real transactions.
- The model has not been deployed or tested in a live payment environment.

Future improvements include longer-term and newer transaction data, more robust temporal validation, probability calibration, cost-sensitive threshold selection, model monitoring, an inference API, and an interactive dashboard.

## 12. Disclaimer

This project is for educational and portfolio purposes. It is not a production-ready banking fraud detection service and should not be used to make real financial decisions.
