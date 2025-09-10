# Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

This repository contains the code and report for a machine learning project that compares a basic Perceptron Learning Algorithm (PLA) with a Multilayer Perceptron (MLP) for character recognition. The project also includes hyperparameter tuning to find the optimal configuration for the MLP.

📝 **Table of Contents**

  * [Project Overview](https://www.google.com/search?q=%23project-overview)
  * [Dataset](https://www.google.com/search?q=%23dataset)
  * [Methodology](https://www.google.com/search?q=%23methodology)
  * [Models Implemented](https://www.google.com/search?q=%23models-implemented)
  * [Results](https://www.google.com/search?q=%23results)
  * [Files in this Repository](https://www.google.com/search?q=%23files-in-this-repository)
  * [How to Run](https://www.google.com/search?q=%23how-to-run)
  * [Libraries Used](https://www.google.com/search?q=%23libraries-used)

📖 **Project Overview**

The primary goal of this experiment is to evaluate the practical differences in performance between two computational models for character recognition:

  * **Model A:** A basic Perceptron Learning Algorithm (PLA).
  * **Model B:** An advanced Multilayer Perceptron (MLP).

The experiment also systematically determines an optimal configuration for the MLP by tuning its hyperparameters to select the most effective combination of network architecture, activation function, learning algorithm, and batch processing size.

📊 **Dataset**

The dataset consists of images of handwritten characters. The images are processed and flattened into feature vectors for the models.

⚙️ **Methodology**

The project follows a standard machine learning pipeline:

1.  **Data Loading and Preprocessing:** The character image data is loaded, and labels are encoded.
2.  **Train/Test Split:** The data is split into training and testing sets.
3.  **Model Training:**
      * The PLA model is trained on the training data.
      * The MLP model is trained with various hyperparameter combinations.
4.  **Hyperparameter Tuning:** A grid search is performed to find the best hyperparameters for the MLP model.
5.  **Model Evaluation:** The models are evaluated based on accuracy, precision, recall, and F1-score.

🤖 **Models Implemented**

  * **Perceptron Learning Algorithm (PLA):** A single-layer neural network.
  * **Multilayer Perceptron (MLP):** A neural network with one or more hidden layers. Different configurations were tested with varying activation functions, optimizers, learning rates, and batch sizes.

🏆 **Results**

The MLP significantly outperformed the PLA, demonstrating the advantage of hidden layers and non-linear activation functions for complex tasks like character recognition. The best-performing MLP model was achieved with the `relu` activation function, `adam` optimizer, a learning rate of `0.001`, and a batch size of `32`.

| Model | Accuracy | Precision | Recall | F1-score |
| :--- | :--- | :--- | :--- | :--- |
| PLA | 0.1774 | 0.2708 | 0.1774 | 0.1576 |
| MLP | 0.2977 | 0.3207 | 0.2977 | 0.2752 |

📂 **Files in this Repository**

  * `ml_Perceptron.ipynb`: The primary Jupyter Notebook containing all Python code for the analysis.
  * `ml_Perceptron_Report.pdf`: A detailed PDF report summarizing the project's aim, methodology, results, and conclusions.
  * `README.md`: This file, providing a comprehensive overview of the project.

🚀 **How to Run**

To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
4.  **Run the notebook:** Open the `ml_Perceptron.ipynb` file and execute the cells.

🛠️ **Libraries Used**

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn`
  * `tensorflow`
  * `PIL` (Pillow)
