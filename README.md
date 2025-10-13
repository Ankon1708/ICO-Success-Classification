# ICO-Success-Classification

## Overview

Machine learning classification project to predict **startup success** (binary outcome) using a Random Forest and XGBoost ensemble. The project's highlight is the implementation of **Missing Indicator Imputation**, which dramatically improved model performance by leveraging the predictive power of missingness.

## Key Techniques & Workflow

### 1. Data Understanding & Preparation

| Category | Technique | Description |
| :--- | :--- | :--- |
| **Data Quality** | Missing Value Assessment | Calculated `NaN` counts/percentages to guide imputation strategy. |
| **Transformation** | Log Transformation | Applied $log(1+x)$ to features like `priceUSD` to address heavy skewness. |
| **Missing Data** | **Missing Indicator Imputation** | **(Core)** Created $\mathbf{F\_{na}}$ binary features to flag missingness (1 = missing, 0 = present). |
| **Missing Data** | Sentinel Value Imputation | Replaced `NaN` values with a distinct, out-of-range value (**-1**) for `priceUSD` and `teamSize`. |
| **Categorical** | One-Hot Encoding | Converted nominal features (`countryRegion`) into multiple binary columns. |
| **Scaling** | Standard Scaling | Normalized numerical features (mean 0, std 1) for consistent model input. |

---

### 2. Modeling & Training Pipeline

| Category | Technique | Description |
| :--- | :--- | :--- |
| **Splitting** | Train-Test Split | Separated data (e.g., 80/20) to ensure unbiased model testing. |
| **Imbalance** | **SMOTE Over-sampling** | Applied to the **training set** only to synthesize minority class samples and address class imbalance. |
| **Models** | Random Forest Classifier | Used for baseline performance and initial feature importance analysis. |
| **Models** | XGBoost Classifier | Utilized an optimized gradient boosting framework for final high-performance prediction. |
| **Optimization** | GridSearchCV + Cross-Validation | Performed systematic hyperparameter tuning with K-Fold cross-validation for robust model selection. |

---

### 3. Evaluation & Interpretation

| Category | Technique | Description |
| :--- | :--- | :--- |
| **Evaluation** | ROC AUC Score | Primary metric to assess the model's ability to distinguish between classes. |
| **Evaluation** | PRFC Metrics | Used Precision, Recall, and F1-Score for a comprehensive view of classification performance. |
| **Interpretation** | Feature Importance Analysis | Extracted scores from tree models to identify the most predictive features (e.g., $\mathbf{F\_{na}}$ features were top-ranked). |
| **Visualization** | Confusion Matrix | Visualized model errors (False Positives/Negatives) for deeper analysis. |

### Modelling Results

T


## Key Finding

The **Missing Indicator Imputation** technique for `priceUSD` and `teamSize` yielded a significant **7-8% increase in model accuracy**. This confirms that the act of a value being missing was a highly predictive signal in the dataset, which the $\mathbf{F\_{na}}$ features successfully captured.

## Project Files

| File Name | Purpose |
| :--- | :--- |
| `Data_Understanding.ipynb` | Initial EDA, visualization, and raw data quality checks. |
| `Modelling.ipynb` | Complete ML pipeline from feature engineering to final model training and evaluation. |
