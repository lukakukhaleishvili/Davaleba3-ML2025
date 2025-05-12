# Davaleba3-ML2025


# German Credit Risk Prediction using Logistic Regression

This project builds a logistic regression model to assess the credit risk of individuals using the [German Credit Risk dataset]([https://www.kaggle.com/datasets/uciml/german-credit](https://github.com/lukakukhaleishvili/Davaleba3-ML2025/blob/main/german_credit_data.csv)). The goal is to predict whether a loan applicant is likely to be a high-risk borrower. I took dataset from kaggle and imported csv file in my reposityro and link is given above for the csv file.


## My objectives where:

- Clean and preprocess the dataset
- Encode categorical variables
- Create a synthetic risk label
- Train a logistic regression model
- Evaluate the model with appropriate metrics

---

##  Dataset

The dataset includes 1000 entries with the following columns:

- **Age**
- **Sex**
- **Job**
- **Housing**
- **Saving accounts**
- **Checking account**
- **Credit amount**
- **Duration**
- **Purpose**

---

##  Steps Performed

All tasks were executed in a single Python file: `credit_risk_model.py`.

### 1. **Data Preprocessing**
- Loaded the dataset using `pandas`.
- Replaced missing values in `Saving accounts` and `Checking account` with `'unknown'`.
- Converted categorical columns (`Sex`, `Housing`, `Saving accounts`, `Checking account`, `Purpose`) to numeric using one-hot encoding.
- Created a synthetic binary target column `High_Risk`:
  - Label `1` if `Credit amount` and `Duration` are both above median.
  - Otherwise, label `0`.

### 2. **Train/Test Split**
- Used `train_test_split` with a 70/30 ratio.

### 3. **Model Training**
- Applied `LogisticRegression` from `sklearn.linear_model`.
- Trained the model on training data.

### 4. **Evaluation**
- Calculated:
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC AUC Score

### 🔍 Example Output:
Confusion Matrix:
[[195   9]
 [ 13  83]]

Classification Report:
              precision    recall  f1-score   support

           0       0.94      0.96      0.95       204
           1       0.90      0.86      0.88        96

    accuracy                           0.93       300
   macro avg       0.92      0.91      0.91       300
weighted avg       0.93      0.93      0.93       300

Accuracy: 0.9266666666666666
ROC AUC Score: 0.9704350490196079
                        Feature  Coefficient
17            Purpose_education    -3.614446
9    Saving accounts_quite rich     1.306448
20              Purpose_repairs    -0.986348
10         Saving accounts_rich     0.775752
6                   Housing_own    -0.539812
8      Saving accounts_moderate     0.527167
7                  Housing_rent    -0.485813
11      Saving accounts_unknown     0.459675
14     Checking account_unknown     0.420512
18  Purpose_furniture/equipment     0.346104
