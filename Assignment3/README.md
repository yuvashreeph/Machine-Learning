# Spam Mail Prediction using Multiple Classification Models

## 📖 Project Overview

This project provides a comprehensive analysis of various classification algorithms for the task of spam email detection. The core of the project is a Jupyter Notebook (`ml_Mail.ipynb`) that implements a full machine learning pipeline, from data preprocessing to model evaluation. A detailed report (`ml_Mail_Report.pdf`) accompanies the code, documenting the experiment's aim, methodology, results, and conclusions.

The primary goal is to compare the performance of several models—including Naïve Bayes variants, K-Nearest Neighbors, and Support Vector Machines—to identify the most effective and robust classifier for this dataset.

-----

## 📊 Dataset

The experiment utilizes a dataset containing pre-extracted features from emails, making it suitable for a feature-based classification task.

  - **Source File:** `mail.csv` (not included in the repository, assumed to be available)
  - **Features:** The dataset consists of 57 numerical features, likely related to word frequencies, character frequencies, and other email metadata.
  - **Target Variable:** A binary variable indicating whether an email is spam (1) or not spam (0).
  - **Preprocessing:** The raw data is cleaned by handling missing values, removing outliers using the Z-score method, and scaling features with `StandardScaler` to ensure optimal model performance.

-----

## 🤖 Models Implemented

A diverse set of classification models were trained and evaluated to provide a thorough comparison:

1.  **Logistic Regression** (as a baseline)
2.  **Naïve Bayes Variants**
      - Gaussian Naïve Bayes
      - Multinomial Naïve Bayes
      - Bernoulli Naïve Bayes
3.  **K-Nearest Neighbors (KNN)**
      - Evaluated with multiple `k` values (1, 3, 5, 7, 9).
      - Compared `KDTree` and `BallTree` algorithms.
4.  **Support Vector Machine (SVM)**
      - Linear Kernel
      - Polynomial Kernel
      - RBF Kernel
      - Sigmoid Kernel

-----

## ⚙️ Methodology

The project follows a structured and robust machine learning workflow:

1.  **Data Preprocessing:** The dataset is loaded, cleaned by handling missing values and outliers, and features are standardized.
2.  **Data Splitting:** The data is partitioned into training (70%), validation (15%), and test (15%) sets to ensure an unbiased evaluation.
3.  **Model Training:** Each model is trained on the preprocessed training data.
4.  **Evaluation:**
      - Performance is measured using **Accuracy, Precision, Recall, and F1-Score**.
      - **5-Fold Cross-Validation** is used on the training data to assess the stability and generalization of each model.
      - The final performance is reported on the held-out test set.
      - Visualizations, including **Confusion Matrices** and **ROC Curves**, are generated for each model to provide a deeper insight into its performance.

-----

## 📈 Results

After a comprehensive evaluation, a clear performance hierarchy emerged among the tested models.

### Key Findings:

  - **Champion Model:** The **Support Vector Classifier (SVC) with a Linear Kernel** was the best-performing model.
  - **Top Performance Metrics (Test Set):**
      - **Accuracy:** **0.9237**
      - **F1-Score:** **0.9210**
  - **Conclusion:** The superior performance of the linear SVC and Logistic Regression suggests that the features of this dataset are largely linearly separable. Non-linear models like KNN and SVMs with non-linear kernels were less effective. The consistency between the cross-validation and test set results confirms the robustness of the SVC (Linear) model.

| Model | Test Accuracy | Test F1-Score | CV Accuracy (Avg) |
| :--- | :--- | :--- | :--- |
| **SVC (Linear)** | **0.9237** | **0.9210** | **0.921** |
| Logistic Regression | 0.9207 | 0.9167 | 0.918 |
| Bernoulli NB | 0.8810 | 0.8830 | 0.879 |
| KNN (k=5) | 0.7622 | 0.7508 | 0.791 |

-----

## 🚀 How to Run

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    Ensure you have Python installed, then install the required libraries.

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```

3.  **Launch Jupyter Notebook:**
    Open the `ml_Mail.ipynb` file in a Jupyter environment to view, run, and interact with the code.

    ```bash
    jupyter notebook ml_Mail.ipynb
    ```

4.  **View the Report:**
    The detailed analysis and findings are documented in the `ml_Mail_Report.pdf` file.

-----

## 🛠️ Libraries Used

  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
