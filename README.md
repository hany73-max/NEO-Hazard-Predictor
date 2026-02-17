# NEO Hazard Predictor: Asteroid Classification Engine ☄️

## Overview
A machine learning pipeline designed to classify Near-Earth Objects (NEOs) as hazardous or safe based on NASA orbital telemetry and physical characteristics data. 

This project demonstrates a complete end-to-end classification architecture, focusing heavily on diagnosing and curing class imbalance and algorithmic overfitting in a production context.

## The Engineering Challenge
Planetary defense datasets are inherently imbalanced (~90% Safe, ~10% Hazardous). A naive model will achieve 90% accuracy by simply ignoring the minority class. In this domain, the cost of a **False Negative** (missing a hazardous asteroid) is catastrophic, while a **False Positive** (a false alarm) is merely an administrative cost. 

**Objective:** Engineer a model that violently prioritizes **Recall** for the hazardous class without entirely sacrificing overall precision.

## Pipeline Architecture

### 1. Data Engineering & Preprocessing
* **Feature Engineering:** Consolidated redundant min/max diameter fields into a single `est_diameter_avg` to reduce dimensionality.
* **Scaling:** Applied transformations to normalize massive numerical disparities between orbital distances and physical diameters.
* **Imbalance Handling:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to inject synthetic hazardous samples into the training space, forcing the baseline algorithm to recognize minority class boundaries.

### 2. Modeling & Hyperparameter Tuning
* **Baseline:** Logistic Regression (established a baseline Recall vs. Precision tradeoff).
* **Champion Engine:** `RandomForestClassifier`
* **Diagnostic Tuning:** The un-tuned Random Forest suffered from terminal overfitting (1.00 Training Recall vs 0.61 Testing Recall). A `GridSearchCV` was deployed to apply physical constraints to the algorithm:
  * `max_depth`: Applied branches-pruning to stop training data memorization and force generalized learning.
  * `class_weight='balanced'`: Hacked the Gini Impurity calculation to severely penalize the engine for misclassifying the minority (Hazardous) class.
  * `n_estimators`: Scaled the engine to ensure stable, averaged-out predictions.

### 3. Engine Interpretability (Black Box Extraction)
Extracted the `feature_importances_` to visualize the AI's physical decision-making process. The model independently learned that **Absolute Magnitude** (light reflection) and **Estimated Diameter** are inversely correlated and act as the primary physical indicators of an asteroid's threat level.

## Tech Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn (Ensemble methods, Metrics, Model Selection)
* **Data Processing:** Pandas, NumPy, Imbalanced-Learn (SMOTE)
* **Visualization:** Matplotlib, Seaborn

## How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed: `pip install -r requirements.txt`
3. Run the Jupyter Notebook `modeling_and_evaluating.ipynb` to execute the pipeline from raw data ingestion to feature importance visualization.