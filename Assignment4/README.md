# Ensemble Model Evaluation for Breast Cancer Diagnosis

📖 **Project Overview**

This project provides a comprehensive analysis and comparison of a single Decision Tree classifier against various advanced ensemble methods for the task of breast cancer diagnosis. The core of the project is a single Python script designed for Google Colab that implements a full machine learning pipeline, from data preprocessing to model evaluation. A detailed LaTeX report accompanies the code, documenting the experiment's aim, methodology, results, and conclusions.

The primary goal is to determine if ensemble techniques (bagging, boosting, and stacking) offer a significant performance improvement over a single decision tree for this classification problem.

📊 **Dataset**

The experiment utilizes the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

-   **Source File:** `wdbc.data`
-   **Features:** The dataset consists of 30 real-valued features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.
-   **Target Variable:** A binary variable `Diagnosis` indicating whether a tumor is malignant (M) or benign (B).
-   **Preprocessing:** The raw data is cleaned by dropping the non-predictive ID column, encoding the target variable, and scaling all features using `StandardScaler` to ensure optimal model performance.

🤖 **Models Implemented**

A diverse set of classification models were trained and evaluated to provide a thorough comparison:

-   **Decision Tree** (as a baseline)
-   **Boosting Ensembles**
    -   AdaBoost
    -   Gradient Boosting
    -   XGBoost
-   **Bagging Ensemble**
    -   Random Forest
-   **Stacking Ensemble**
    -   Three different configurations were tested, combining base models like SVM, Naive Bayes, Decision Tree, and KNN with a final meta-learner.

⚙️ **Methodology**

The project follows a structured and robust machine learning workflow:

-   **Data Preprocessing:** The dataset is loaded, cleaned, and all features are standardized.
-   **Data Splitting:** The data is partitioned into a training (80%) and test (20%) set using a **stratified split** to maintain the class distribution.
-   **Model Training and Tuning:** Each model's hyperparameters are systematically tuned using **`GridSearchCV`**.
-   **Evaluation:**
    -   Performance is measured using **Accuracy**, **F1-Score**, **Confusion Matrix**, and the **ROC Curve**.
    -   **5-Fold Cross-Validation** is used on the training data to assess the stability and generalization of each model.
    -   The final performance is reported on the held-out test set.
    -   **Feature Importance** plots are generated for each model to provide insight into its decision-making process.

📈 **Results**

After a comprehensive evaluation, the ensemble methods demonstrated a clear and consistent performance advantage over the single Decision Tree.

**Key Findings:**

-   **Champion Models:** **AdaBoost**, **XGBoost**, and the primary **Stacked Model** emerged as the top-performing models, all achieving an identical average cross-validation accuracy of **96.48%**.
-   **Excellent Generalization:** The XGBoost and Stacked Models showed the best generalization, with almost no difference between their cross-validation and test set scores, indicating they are very robust and not overfitted.
-   **Conclusion:** The superior performance of the ensemble methods confirms their effectiveness for this classification task. They successfully combine the predictive power of multiple models to create a more accurate and reliable classifier than a single Decision Tree.

| Model | CV Accuracy (Avg) | Test Accuracy | Test F1-Score (Weighted) |
| :--- | :--- | :--- | :--- |
| **AdaBoost** | **0.9648** | **0.9737** | **0.9737** |
| **XGBoost** | **0.9648** | **0.9649** | **0.9647** |
| **Stacked Model** | **0.9648** | **0.9649** | **0.9645** |
| Random Forest | 0.9626 | 0.9737 | 0.9737 |
| Decision Tree | 0.9451 | 0.9474 | 0.9474 |

🚀 **How to Run**

1.  **Set up Google Drive:**
    -   In your Google "My Drive", create a folder named `ML_LAB`.
    -   Inside `ML_LAB`, create another folder named `BreastCancer_WinconsinDiagnostic`.
    -   Upload the `wdbc.data` file into this folder.

2.  **Run in Google Colab:**
    -   Open a new Google Colab notebook.
    -   Copy the entire content of the final Python script into a single cell.
    -   Run the cell. The script will automatically mount your Google Drive, load the data, train all models, and display all results, tables, and plots.

3.  **View the Report:** The detailed analysis and findings are documented in the accompanying LaTeX report PDF.

🛠️ **Libraries Used**

-   `pandas`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `xgboost`
-   `google-colab`
