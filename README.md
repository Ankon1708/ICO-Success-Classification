# ICO-Success-Classification
This project explores the application of supervised machine learning models to predict the success of Initial Coin Offerings (ICOs) using a real-world dataset of over 2,700 ICOs.
The goal is to model success based on a mix of financial, social, and campaign features extracted from public ICO data.

## Project Overview
ICOs are a popular form of blockchain-based crowdfunding, but their success is highly unpredictable due to decentralized and often noisy data. This project:

Cleans and transforms a large dataset containing categorical, numeric, datetime, and textual features.

Engineers new features (e.g., campaign duration).

Handles high-cardinality categorical variables using one-hot encoding.

Compares two classification models — Random Forest Classifier (RFC) and Support Vector Classifier (SVC) — to determine which performs better in predicting campaign outcomes.

## Key Features
Data Preprocessing: Outlier removal, normalization, handling missing data, category consolidation.

Feature Engineering: Derived variables such as campaign duration and encoded categorical fields.

### Model Training

Random Forest Classifier: 50 trees, tuned hyperparameters.

Support Vector Classifier: RBF kernel with regularization.

Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

Comparison & Interpretation: Trade-offs analyzed between precision-focused and recall-focused predictions.

## Tech Stack
Python (Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib, Seaborn)

Jupyter Notebooks
