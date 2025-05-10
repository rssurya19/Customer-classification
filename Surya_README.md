# 9417 Group Project - README

## Overview

This project implements a machine learning pipeline to address a classification problem, with a focus on handling imbalanced data. The pipeline includes data loading, exploratory data analysis (EDA), preprocessing, model training, selection, prediction generation, and distribution shift analysis.

## Prerequisites

Before running the code, ensure you have the following installed:

* **Python 3.x**
* **Required Python Libraries:** You can install these using pip:

    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm
    ```

## Data

The code expects the following data files to be present in the same directory as the script:

* `X_train.csv`: Training features.
* `y_train.csv`: Training labels.
* `X_test_1.csv`: Test set 1 features.
* `X_test_2.csv`: Test set 2 features.
* `y_test_2_reduced.csv`: Reduced labels for test set 2 (used in distribution shift analysis).

    **Note:** Ensure the data files are in CSV format and contain the expected columns.

## How to Run the Code

1.  **Clone the Repository (Optional):** If you have the code in a repository, clone it to your local machine.
2.  **Install Dependencies:** If you haven't already, install the required Python libraries using pip (see Prerequisites).
3.  **Place Data Files:** Make sure all the necessary CSV data files are in the same directory as the Python script (`9417_group_project.py`).
4.  **Run the Script:** Execute the Python script from your terminal:

    ```bash
    python final_deeplearn.py
    ```

## Output

The script will perform the following actions:

* **Data Loading and Preparation:** Loads the training and test datasets.
* **Exploratory Data Analysis (EDA):** Prints data shapes, class distributions (and saves a plot: `class_distribution.png`), missing value information, and feature statistics.
* **Data Preprocessing:** Standardizes features and encodes labels.
* **Handle Class Imbalance:** Handles class imbalance using class weights.
* **Model Development:** Trains several machine learning models:
    * Random Forest
    * Balanced Random Forest
    * XGBoost
    * LightGBM
    * Logistic Regression
    * It evaluates these models using weighted F1-score and log loss, and prints a classification report for each.
* **Model Selection and Final Training:** Selects the best-performing model based on validation F1-score and retrains it on the entire training dataset.
* **Generate Predictions:** Generates prediction probabilities for `X_test_1` and `X_test_2`.
* **Distribution Shift Analysis:** Compares feature and class distributions between the training and test sets and performs PCA for visualization (saves plots: `feature_distribution_comparison.png`, `class_distribution_comparison.png`, and `pca_comparison.png`).
* **Save Predictions:** Saves the prediction probabilities to `preds_1.npy` and `preds_2.npy`, and creates a zip archive (`GROUP_9417.zip`) containing these files.

## Files Generated

After running the script, you should find the following files in the same directory:

* `class_distribution.png`: Plot of the class distribution in the training data.
* `feature_distribution_comparison.png`: Plot comparing feature distributions.
* `class_distribution_comparison.png`: Plot comparing class distributions.
* `pca_comparison.png`: PCA visualization of train vs. test data.
* `preds_1.npy`: NumPy array containing prediction probabilities for test set 1.
* `preds_2.npy`: NumPy array containing prediction probabilities for (reduced) test set 2.
* `GROUP_9417.zip`: Zip archive containing `preds_1.npy` and `preds_2.npy`.

## Notes

* The script uses a fixed random seed (9417) for reproducibility.
* You may need to adjust file paths if your data files are located elsewhere.
* The script assumes a specific format for the input CSV files. Ensure your data matches this format.
* The group name used for the zip file can be changed by modifying the `group_name` variable in the `save_predictions` function.