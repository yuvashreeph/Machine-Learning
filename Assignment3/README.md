# Spam Mail Prediction using Multiple Classification Models

## 📖 Project Overview

This project focuses on building and evaluating a suite of machine learning models to classify emails as "spam" or "not spam". The primary goal is to compare various classification algorithms, from simple baselines to powerful ensemble methods, and identify the most effective model for this task. The project follows a structured machine learning workflow, including data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and robust evaluation using K-fold cross-validation.

-----

## 📊 Dataset

The experiment uses the "Spam Mail" dataset, which contains pre-extracted features from emails. The dataset consists of 57 features, primarily related to word and character frequencies, and a target variable indicating whether the email is spam.

  - **File:** `mail.csv`
  - **Initial Shape:** (4601, 58)
  - **Shape after Preprocessing:** (2185, 58)

-----

## 🤖 Models Implemented

A comprehensive set of 13 classification models were trained and evaluated to provide a thorough comparison:

1.  **Logistic Regression** (Baseline)
2.  **Naive Bayes**
      - GaussianNB
      - MultinomialNB
      - BernoulliNB
3.  **K-Nearest Neighbors (KNN)**
      - Tested with k-values: [1, 3, 5, 7, 9]
      - Algorithms: KDTree, BallTree
4.  **Support Vector Machine (SVM)**
      - Linear Kernel
      - Polynomial Kernel
      - RBF Kernel
      - Sigmoid Kernel
5.  **Tree-Based Ensemble Models**
      - Decision Tree Classifier
      - Random Forest Classifier
      - AdaBoost Classifier
      - Gradient Boosting Classifier
      - **XGBoost Classifier**

-----

## ⚙️ Methodology

The project follows a standard machine learning pipeline:

1.  **Data Preprocessing & EDA:**

      - Loaded the dataset using `pandas`.
      - Handled missing values by dropping rows with more than 50% missing data and filling the rest with the median.
      - Removed outliers using the Z-score method (threshold=3).
      - Standardized numerical features using `StandardScaler`.

2.  **Data Splitting:**

      - The dataset was split into training (70%), validation (15%), and test (15%) sets to ensure robust and unbiased evaluation.

3.  **Model Training & Hyperparameter Tuning:**

      - Each model was trained on the training data.
      - `GridSearchCV` and `RandomizedSearchCV` were used to find the optimal hyperparameters for key models like Decision Tree and Random Forest.

4.  **Evaluation:**

      - Models were evaluated using a comprehensive set of metrics: **Accuracy, Precision, Recall, and F1-Score**.
      - **5-Fold Cross-Validation** was performed on the training set to get a stable measure of performance.
      - Final performance was reported on the held-out test set.
      - Visualizations like **Confusion Matrices** and **ROC Curves** were generated for each model to analyze its performance in detail.

-----

## 📈 Results

The ensemble methods demonstrated clear superiority over simpler models. The **XGBoost Classifier** emerged as the top-performing model across all evaluation metrics.

### Key Findings:

  - **Best Model:** **XGBoost Classifier**
  - **Test Set F1-Score:** **0.9499**
  - **Test Set Accuracy:** **0.9481**
  - The performance of XGBoost on the 5-fold cross-validation (Avg. F1-Score: 0.947) was very close to its performance on the final test set, indicating that the model is robust and generalizes well to unseen data.

| Model | Test Accuracy | Test F1-Score |
| :--- | :--- | :--- |
| **XGBoost Classifier** | **0.9481** | **0.9499** |
| Random Forest | 0.9451 | 0.9458 |
| Gradient Boosting | 0.9420 | 0.9424 |
| SVC (Linear) | 0.9237 | 0.9210 |
| Logistic Regression | 0.9207 | 0.9167 |

-----

## 🚀 How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yuvashreeph/Machine-Learning.git
    cd Machine-Learning/Assignment3
    ```

2.  **Install the required libraries:**

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```

3.  **Run the Jupyter Notebook:**
    Open and run the `ml_Mail.ipynb` notebook in a Jupyter environment. Ensure the `mail.csv` dataset is in the correct path as specified in the notebook.

-----

## 🛠️ Libraries Used

  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `xgboost`

-----


