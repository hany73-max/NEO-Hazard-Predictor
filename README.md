# ☄️ NEO Hazard Predictor (NASA NeoWs)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green.svg)](https://imbalanced-learn.org/)

## 📝 Project Overview
The **NEO Hazard Predictor** is a binary classification machine learning pipeline designed to predict whether a Near-Earth Object (NEO) poses a hazardous threat to Earth. 

Utilizing telemetry and sensor data from the **NASA Near Earth Object Web Service (NeoWs)**, this project demonstrates a robust, production-ready data science pipeline with a strong emphasis on preventing data leakage, handling severe class imbalances, and mathematically driven feature selection.

## 📊 The Dataset
The dataset contains over 90,000 recorded asteroids. Features include physical dimensions, relative velocity, and orbital proximity.
* **Target Variable:** `hazardous` (Boolean: True/False)
* **Challenge:** The dataset presents a severe **10:1 class imbalance** (~90% Safe, ~10% Hazardous), requiring advanced data augmentation techniques to prevent the model from defaulting to a majority-class prediction.

---

## 🏗️ Pipeline Architecture & Engineering Decisions

### 1. Exploratory Data Analysis & Data Cleaning (Row-Wise Operations)
* **Identifier Removal:** Dropped non-predictive features (`id`, `name`) to prevent model noise.
* **Dimensionality Reduction (Row-wise):** Synthesized highly correlated features (`est_diameter_min` and `est_diameter_max`) into a single `est_diameter_avg` feature to retain 100% of the variance while reducing dimensionality.
* **Zero-Variance Pruning:** Identified and removed the `sentry_object` feature after discovering it held zero variance across the dataset.

### 2. Data Splitting (Leakage Prevention)
To ensure the integrity of the test environment, the dataset was split into an 80/20 Train-Test configuration **before** any column-wise operations were performed. 
* Utilized `stratify=y` to guarantee the 10:1 hazard ratio was preserved in both the training and testing environments.

### 3. Feature Selection & Transformation (Column-Wise Operations)
* **Multicollinearity Hunt:** Generated a Pearson Correlation Matrix strictly on the training data. Verified that all remaining features scored well below the 0.85 correlation threshold, confirming unique sensor inputs.
* **Standardization:** Applied `StandardScaler` to normalize features with vastly different magnitudes (e.g., millions of kilometers vs. decimal magnitudes). *Strictly fitted on training data and applied to testing data to prevent data leakage.*

### 4. Synthetic Data Augmentation (Imbalance Handling)
* Applied **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively to the training data.
* Successfully transformed the heavily skewed 65,596/7,072 target distribution into a perfectly balanced 65,596/65,596 training set, forcing the algorithms to weigh hazardous and safe asteroids equally.

### 5. Modeling (In Progress)
* **Baseline Model:** Logistic Regression
* **Advanced Ensemble Model:** Random Forest Classifier

### 6. Evaluation Metrics (Upcoming)
Due to the critical nature of the minority class (Hazardous NEOs), standard accuracy is an invalid metric. The models will be evaluated primarily on:
* **Recall (Sensitivity):** Minimizing False Negatives (Predicting an asteroid is safe when it is actually hazardous).
* **Precision-Recall Curve Area:** Evaluating performance across different classification thresholds.

---

## 💻 Repository Structure
```text
NEO-Hazard-Predictor/
│
├── data/
│   ├── raw/neo.csv                 # Original NASA dataset
│   └── processed/                  # Cleaned and split datasets
│
├── notebooks/
│   ├── 01_EDA_and_Audit.ipynb      # Data exploration and visualizations
│   ├── 02_PreModeling.ipynb        # Engineering, SMOTE, and Scaling
│   └── 03_Modeling_and_Eval.ipynb  # Algorithm training and diagnostics
│
├── src/                            # Modular python scripts (Future)
└── README.md