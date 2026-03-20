# Titanic Survival Prediction: Data Science & ML Analysis

## Overview

This repository documents a complete machine learning workflow to predict the survival of passengers aboard the RMS Titanic.
This project addresses the classic **Kaggle Titanic competition** by performing in-depth **Exploratory Data Analysis (EDA)**, **feature engineering**, and **model comparison**.

The core objective is to identify key factors influencing survival and develop a highly accurate classification model.
The full analysis, visualizations, and modeling steps are detailed within the main Jupyter Notebook: `Titanic.ipynb`.

---

## Dataset and Features

The dataset is sourced from the Kaggle Titanic competition and consists of the standard `train.csv` and `test.csv` files, available in this repository's root directory.

### Key Features Used:

| Feature | Description | Type | Notes |
| :--- | :--- | :--- | :--- |
| **Survived** | Survival (0 = No, 1 = Yes) | **Target** | The variable we are predicting. |
| **Pclass** | Ticket class (1st, 2nd, 3rd) | Categorical | Proxy for socio-economic status. |
| **Sex** | Sex (male/female) | Categorical | A primary predictor of survival. |
| **Age** | Age in years | Numerical | Imputed during preprocessing. |
| **Fare** | Passenger fare | Numerical | Price paid for the ticket. |
| **SibSp** | # of siblings/spouses aboard | Numerical | Siblings/Spouses. |
| **Parch** | # of parents/children aboard | Numerical | Parents/Children. |

---

## Exploratory Data Analysis (EDA) Highlights

EDA is powered by **[ydata-profiling](https://github.com/ydataai/ydata-profiling)**, which auto-generates a rich interactive report covering distributions, missing values, correlations, duplicate detection, and train/test comparison — replacing manual summary steps.

### Key Insights from EDA:

* **Gender:** Female passengers showed a significantly higher survival rate (~74%) compared to males (~19%).
* **Class:** Survival rates dropped sharply from 1st to 3rd class, indicating a strong correlation between socio-economic status and survival.
* **Age Distribution:** Age was analyzed and binned into categories (Child, Youth, Adult) to improve predictive strength.
* **Embarked & Fare:** Passengers embarking from Cherbourg and paying higher fares showed better survival odds, largely due to their concentration in 1st class.

### Visualizations:

* Survival rate by **Sex**, **Title**, and **Passenger Function**
* Age distribution split by survival status and sex
* Effect of **Pclass** on survival rate
* Fare vs Pclass strip plot (log scale)
* Passenger distribution by embarkation port and class (squarify treemap)

---

## Data Preprocessing & Feature Engineering

The following steps were implemented to prepare the data for optimal model performance:

1. **Missing Value Handling:** `Age` imputed with KNN, `Fare` with median, `Embarked` with mode.
2. **Feature Transformation:**
   * The **`Title`** feature was extracted from the `Name` column and grouped into categories (e.g., Nobility, Military, Women, Men).
   * **`FamilySize`** was created by summing `SibSp` and `Parch`.
   * **`Function`** grouped passengers by title into broader social roles.
   * **Age Binning:** Continuous `Age` transformed into categorical bins.
3. **Encoding and Scaling:** Categorical features encoded with One-Hot Encoding; numerical features scaled with `StandardScaler`.
4. **Dropped Features:** `Name`, `Cabin` (too sparse), `PassengerId`, `Ticket` were dropped before modeling.

---

## Modeling and Evaluation

Several classical machine learning classification models were trained and compared using `GridSearchCV` with `roc_auc` scoring.

### Models Explored:

* **Logistic Regression**
* **Support Vector Classifier (SVC)**
* **BernoulliNB**
* **Random Forest Classifier**
* **XGBoost Classifier**
* **AdaBoost Classifier**
* **LightGBM Classifier**

### Evaluation Metrics:

Models were selected based on **roc_auc** and evaluated using **Accuracy**, **F1-Score**, **Confusion Matrix**, and **k-Fold Cross-Validation**.

### Best Performing Model:

**BernoulliNB** achieved the highest `roc_auc` score after hyperparameter tuning.

| Model | Cross-Validation Score | ROC-AUC |
| :--- | :--- | :--- |
| Logistic Regression | 0.862 | 0.960 |
| SVC | 0.843 | 0.955 |
| **BernoulliNB** | 0.804 | **1.000** |
| Random Forest | 0.853 | 0.943 |
| XGBoost | **0.865** | 0.963 |
| AdaBoost | 0.859 | 0.967 |
| LightGBM | 0.855 | 0.945 |

---

## Project Structure and Execution

### Repository Structure

```
Titanic/
├── LICENSE
├── README.md                     # This document
├── Titanic.ipynb                 # Main analysis: EDA, feature engineering, modeling
├── test.csv                      # Original test data
├── train.csv                     # Original training data
├── gender_submission.csv         # Sample submission file
└── requirements.txt              # Python dependencies
```

### Setup and Running

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ishaq-ML/Titanic.git
   cd Titanic
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Analysis:**
   ```bash
   jupyter notebook Titanic.ipynb
   ```

---

## Dependencies and License

### Dependencies

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `squarify`
* `scikit-learn`
* `xgboost`
* `lightgbm`
* `ydata-profiling`
* `jupyter`

### License

This project is licensed under the **MIT License**.

---
