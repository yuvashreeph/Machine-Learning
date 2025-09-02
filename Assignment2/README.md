# Loan Sanction Amount Prediction using Multiple Regression Models

This repository contains the code and report for a machine learning project focused on predicting the sanctioned loan amount based on customer and property data. The project compares the performance of eleven different regression algorithms to identify the most accurate model.

## 📝 Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23-project-overview)
  - [Dataset](https://www.google.com/search?q=%23-dataset)
  - [Methodology](https://www.google.com/search?q=%23-methodology)
  - [Models Implemented](https://www.google.com/search?q=%23-models-implemented)
  - [Results](https://www.google.com/search?q=%23-results)
  - [Files in this Repository](https://www.google.com/search?q=%23-files-in-this-repository)
  - [How to Run](https://www.google.com/search?q=%23-how-to-run)
  - [Libraries Used](https://www.google.com/search?q=%23-libraries-used)

## 📖 Project Overview

This project aims to build, evaluate, and compare multiple regression models to predict loan sanction amounts. The results conclusively show that the **XGBoost Regressor** is the superior model, achieving an outstanding **R² score of 0.96**. This high performance indicates that the underlying relationships in the financial data are complex and non-linear, making ensemble methods the most effective tool for this predictive task.

## 📊 Dataset

The dataset used for this project is `loan_sanction.csv`, containing various customer attributes.

**Key Features:**

  - Customer Demographics (`Gender`, `Dependents`)
  - Financial Information (`Income (USD)`, `Credit Score`, `Current Loan Expenses (USD)`)
  - Employment Details (`Type of Employment`, `Income Stability`)
  - Property Information (`Property Location`, `Property Age`)

**Target Variable:**

  - `Loan Sanction Amount (USD)`

## ⚙️ Methodology

The project follows a standard machine learning pipeline:

1.  **Data Loading:** The dataset is loaded using pandas.
2.  **Data Preprocessing:**
      - **Missing Value Imputation:** Null values are handled using mean/median for numerical columns and mode/default values for categorical ones.
      - **Outlier Capping:** Outliers are managed by capping them at the 5th and 95th percentiles to limit the influence of extreme values.
      - **Categorical Encoding:** Non-numeric features are converted into a machine-readable format using `LabelEncoder`.
      - **Feature Standardization:** Numerical features are scaled using `StandardScaler` to have a mean of 0 and a standard deviation of 1.
3.  **Exploratory Data Analysis (EDA):** Visualizations like distribution plots, scatter plots, and a correlation heatmap are used to understand the data's characteristics.
4.  **Train/Test Split:** The preprocessed data is split into training, validation, and test sets.
5.  **Model Training & Hyperparameter Tuning:** Eleven different regression models are trained on the data. `GridSearchCV` is used for hyperparameter tuning to find the optimal configuration for each model.
6.  **Model Evaluation:** Models are evaluated based on MAE, MSE, RMSE, and R² Score.

## 🤖 Models Implemented

A total of eleven regression models were trained and compared:

1.  Linear Regression
2.  Ridge Regression
3.  Lasso Regression
4.  ElasticNet Regression
5.  Polynomial Regression (Degree 2)
6.  Decision Tree Regressor
7.  Random Forest Regressor
8.  AdaBoost Regressor
9.  Gradient Boosting Regressor
10. **XGBoost Regressor (Best Performing Model)**
11. Support Vector Regressor (SVR) with Linear, Polynomial, RBF, and Sigmoid kernels.

## 🏆 Results

The comprehensive comparison revealed a clear performance hierarchy, with tree-based ensemble models significantly outperforming all others.

  - **Best Model:** **XGBoost Regressor**
  - **Test Set R² Score:** **0.96**
  - **Test Set RMSE:** **8825.72**

This demonstrates XGBoost's superior ability to model the complex, non-linear patterns present in the financial data.

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| Linear Regression | 19022.77 | 27266.71 | 0.60 |
| Ridge Regression | 19022.77 | 27266.71 | 0.60 |
| Decision Tree | 1211.33 | 10977.72 | 0.94 |
| Random Forest | 928.32 | 9599.34 | 0.95 |
| Gradient Boosting | 950.60 | 9746.79 | 0.95 |
| **XGBoost Regressor** | **808.56** | **8825.72** | **0.96** |

## 📂 Files in this Repository

  - `ml_LoanPrediction.ipynb`: The Jupyter Notebook containing all the Python code for data preprocessing, EDA, model training, and evaluation.
  - `ml_LoanPrediction_Report.pdf`: A detailed PDF report summarizing the project's aim, methodology, results, and conclusion.
  - `loan_sanction.csv`: The dataset used for the project (You should add this file to your repository).
  - `README.md`: This file, providing a comprehensive overview of the project.

## 🚀 How to Run

To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

    *(You can create a `requirements.txt` file with the content from the section below).*

4.  **Launch Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

5.  **Run the notebook:** Open the `ml_LoanPrediction.ipynb` file and execute the cells.

## 🛠️ Libraries Used

Create a file named `requirements.txt` and add the following libraries to it:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```
