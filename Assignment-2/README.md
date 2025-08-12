# Comprehensive Analysis of Regression Models for Loan Amount Prediction


A detailed Machine Learning project focused on predicting loan sanction amounts. This repository contains the code and analysis for a comparative study of eleven different regression algorithms, from simple linear models to advanced ensemble methods, to determine the most effective approach for this financial forecasting task.

---

## 📖 Table of Contents

* [1. Project Motivation](#1-project-motivation)
* [2. Dataset](#2-dataset)
* [3. Project Pipeline](#3-project-pipeline)
* [4. Installation & Usage](#4-installation--usage)
* [5. Models Evaluated](#5-models-evaluated)
* [6. Results and Key Findings](#6-results-and-key-findings)
* [7. Conclusion](#7-conclusion)
* [8. Future Improvements](#8-future-improvements)

---

### 1. Project Motivation

Accurately predicting the loan amount that can be sanctioned for an applicant is a critical task for financial institutions. It helps in managing risk, ensuring regulatory compliance, and improving customer satisfaction. This project was undertaken to explore how various machine learning techniques can be applied to this problem and to identify which models provide the most reliable predictions based on applicant data.

### 2. Dataset

The analysis is performed on the `loan_sanction.csv` dataset. This dataset contains a variety of applicant attributes, including:
* **Demographic Information:** Gender, Dependents
* **Financial Status:** Income, Income Stability, Type of Employment
* **Credit History:** Credit Score, Active Credit Cards, Current Loan Expenses
* **Property Details:** Property Age, Property Location
* **Target Variable:** `Loan Sanction Amount (USD)`

### 3. Project Pipeline

The project follows a systematic and robust machine learning workflow, ensuring reproducibility and reliability.

1.  **Data Cleaning & Preprocessing:**
    * **Missing Value Imputation:** Categorical features with missing values were filled using the mode, while numerical features were imputed with the mean or median. Rows with a missing target variable were dropped.
    * **Outlier Capping:** To prevent extreme values from skewing model performance, outliers in numerical columns were capped at the 5th and 95th percentiles.
    * **Type Conversion:** Ensured all data types were appropriate for modeling.

2.  **Feature Engineering:**
    * **Categorical Encoding:** `LabelEncoder` was used to convert all categorical string values into a numerical format suitable for machine learning algorithms.
    * **Standardization:** All numerical features were scaled using `StandardScaler` to ensure that features with larger ranges did not disproportionately influence model training.

3.  **Exploratory Data Analysis (EDA):**
    * Conducted to uncover insights and understand the underlying structure of the data. This involved visualizing feature distributions, analyzing relationships between variables using scatter plots, and examining multicollinearity with a correlation heatmap.

4.  **Model Training & Evaluation:**
    * The dataset was split into **Training (70%)**, **Validation (15%)**, and **Test (15%)** sets.
    * Eleven distinct regression models were trained and fine-tuned.
    * **Hyperparameter Tuning** was performed using `GridSearchCV` to optimize the more complex models (e.g., Ridge, Lasso, Decision Tree, XGBoost).
    * **5-Fold Cross-Validation** was employed on the training set to ensure the model's performance is stable and generalizable, providing a robust estimate of its effectiveness.

### 4. Installation & Usage

To set up and run this project locally, please follow these steps. It is highly recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yuvashreeph/Machine-Learning.git](https://github.com/yuvashreeph/Machine-Learning.git)
    cd Machine-Learning/Assignment2
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file should be created with all the necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```
    *If a `requirements.txt` file is not available, install the primary libraries manually:*
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```

4.  **Run the experiment:**
    Open the Jupyter Notebook `Loan_Amount_Prediction.ipynb` and run the cells sequentially to execute the entire pipeline from data loading to model evaluation.

### 5. Models Evaluated

The following eleven models were trained and compared:
1.  Linear Regression
2.  Ridge Regression
3.  Lasso Regression
4.  ElasticNet Regression
5.  Polynomial Regression
6.  Decision Tree Regressor
7.  Support Vector Regressor (SVR)
8. AdaBoost Regressor
9. Gradient Boosting Regressor
10. **XGBoost Regressor**
11.  Random Forest Regressor

### 6. Results and Key Findings

The models showed a wide range of performance, with a clear distinction between linear models and more complex, non-linear ensemble methods. The ensemble models, particularly those based on gradient boosting, significantly outperformed all other types.

The **XGBoost Regressor** emerged as the top-performing model.

**Final Test Set Performance of XGBoost Regressor:**

| Metric | Score |
| :--- | :--- |
| **R-squared ($R^2$)** | **0.96** |
| Mean Absolute Error (MAE) | 808.56 |
| Root Mean Squared Error (RMSE) | 8825.72 |

The high $R^2$ score indicates that the XGBoost model can explain approximately 96% of the variance in the loan sanction amount, demonstrating its powerful predictive capability for this dataset. The superiority of tree-based ensembles suggests that the relationships between applicant features and the final loan amount are intricate and non-linear.

### 7. Conclusion

This project successfully demonstrates the application of a rigorous machine learning pipeline to a real-world financial problem. By comparing eleven different models, it was conclusively determined that the **XGBoost Regressor is the most suitable model** for predicting loan sanction amounts for this dataset, owing to its ability to capture complex patterns effectively.

### 8. Future Improvements

Potential next steps to enhance this project could include:
* **Advanced Feature Engineering:** Creating new features from existing ones to potentially improve model accuracy.
* **Alternative Encoding:** Exploring other encoding techniques like One-Hot Encoding for categorical variables with low cardinality.
* **Model Deployment:** Packaging the final XGBoost model into a REST API using a framework like Flask or FastAPI for real-world application.
* **Deeper Error Analysis:** Investigating the specific cases where the model makes the largest errors to understand its limitations.
