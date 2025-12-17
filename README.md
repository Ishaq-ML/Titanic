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

The `Titanic.ipynb` notebook documents a comprehensive data exploration phase, including initial statistical summaries, checking for missing values, and visualizing key relationships.

### Key Insights from EDA:

* **Gender:** Female passengers showed a significantly higher survival rate (approximately 74%) than males (approximately 19%).
* **Class:** Survival rates dropped sharply from 1st to 3rd class, indicating a strong correlation between socio-economic status and survival chance.
* **Age Distribution:** The distribution of age across the dataset was analyzed , leading to strategies for **Age Binning**.

### Visualizations:

* Distribution plots for continuous features (`Age`, `Fare`).
* Grouped bar plots showing the clear impact of **`Sex` and `Pclass`** on the survival rate. 
* **Correlation Matrix** analysis to understand linear relationships between numerical features.

---

## Data Preprocessing & Feature Engineering

The following rigorous steps were implemented to prepare the data for optimal model performance:

1.  **Missing Value Handling:** Missing values for `Age`, `Fare`, and `Embarked` were addressed through imputation (e.g., median, mode).
2.  **Feature Transformation:**
    * The **`Title`** feature was extracted from the `Name` column (e.g., 'Mr.', 'Mrs.').
    * **`FamilySize`** was created by summing `SibSp` and `Parch`.
    * **Age Binning:** The continuous `Age` feature was transformed into categorical bins (e.g., Child, Youth, Adult) to improve predictive strength.
3.  **Encoding and Scaling:** Categorical features were converted using One-Hot Encoding, and numerical features like `Fare` were scaled (e.g., using `StandardScaler`) to ensure balanced feature influence.

---

## Modeling and Evaluation

The analysis involved training and evaluating several classical machine learning classification models.

### Models Explored:

* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM)**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **XGBoost Classifier**

### Evaluation Metrics:
Best model was selected based on  **roc_auc** and was evaluated using **Accuracy Score**, **F1-Score**, and **Confusion Matrices** . **k-Fold Cross-Validation** to ensure that the model generalized well.

### Best Performing Model:

The **Best Performing Model was BernoulliNB** achieved the **roc_auc** score after hyperparameter tuning.

| Model | Cross-Validation Score | Roc_Auc |
| :--- | :--- | :--- |
| **Logistic Regression** | [0.862] | [e.g., 0.960] |
| **SVC** | [0.843] | [e.g., 0.955] |
| **BernoulliNB** | [0.804] | **[e.g., 1.000]** |
| **Random Forest** | [0.853] | [e.g., 0.943] |
| **XGBoost** | **[0.865]** | [e.g., 0.963] |
| **AdaBoost** | [0.859] | [e.g., 0.967] |
| **Random Forest** | [0.855] | [e.g., 0.945] |

---

## Project Structure and Execution

### Repository Structure

```tree
Titanic/
├── LICENSE         
├── README.md             # This document
├── Titanic.ipynb         # Main analysis, EDA, feature engineering, and modeling
├── test.csv              # Original test data
├── train.csv             # Original training data
└── requirements.txt      # Python dependencies
```

### Setup and Running

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ishaq-ML/Titanic.git](https://github.com/ishaq-ML/Titanic.git)
    cd Titanic
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Analysis:**
    To execute the full data science workflow, including EDA, model training, and evaluation, open the main notebook:
    ```bash
    jupyter notebook Titanic.ipynb
    ```

---

## Dependencies and License

### Dependencies

This project relies on the following major Python libraries (check `requirements.txt` for exact versions):

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `jupyter`

### License

This project is licensed under the **MIT License**.

---
