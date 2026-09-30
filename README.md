# ThyroPredict — AI-Based Thyroid Abnormality Prediction System

An end-to-end, educational Machine Learning project that analyzes clinical patient data and laboratory thyroid hormone panels to predict thyroid abnormalities. Built with **Scikit-Learn** and **Streamlit**, this project implements a leak-free preprocessing pipeline and compares four classic classification algorithms.

---

> ### ⚠️ Important Educational & Medical Disclaimer
> **This application is an educational machine-learning prototype and is not intended to provide medical diagnosis, treatment, or clinical advice. Predictions should not be used as a substitute for evaluation by a qualified healthcare professional.**
> This project does NOT claim that "AI diagnoses thyroid disease." It demonstrates algorithmic pattern recognition on benchmark clinical tabular data.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Main Objectives](#main-objectives)
4. [Dataset Description](#dataset-description)
5. [Data Cleaning & Leakage Prevention](#data-cleaning--leakage-prevention)
6. [Machine Learning Algorithms](#machine-learning-algorithms)
7. [Experimental Results & Evaluation](#experimental-results--evaluation)
8. [Feature Importance](#feature-importance)
9. [Project Architecture & Workflow](#project-architecture--workflow)
10. [Folder Structure](#folder-structure)
11. [Installation & Setup](#installation--setup)
12. [How to Train the Model](#how-to-train-the-model)
13. [How to Run the Web Application](#how-to-run-the-web-application)
14. [Viva & Interview Questions](#viva--interview-questions)

---

## 📖 Project Overview
Thyroid dysfunction affects hundreds of millions of people worldwide. The thyroid gland produces crucial metabolic hormones (Thyroxine/T4 and Triiodothyronine/T3), under the regulation of Thyroid Stimulating Hormone (TSH) from the pituitary gland.

**ThyroPredict** is a complete, beginner-friendly AI/ML prototype designed for academic presentation (college projects and vivas). It automates the end-to-end workflow:
- Ingests standard clinical thyroid laboratory records.
- Handles missing laboratory measurements without data leakage.
- Trains and benchmarks multiple classification algorithms.
- Deploys the best-performing model into an interactive **Streamlit web application** with input validation and instant probability outputs.

---

## 🎯 Problem Statement
Thyroid disorders (such as hypothyroidism) often manifest with subtle, non-specific symptoms (fatigue, weight gain, sensitivity to cold). Biochemical blood panel tests are essential for confirmation, but manual screening of multiple parameters across high patient volumes can benefit from preliminary machine learning risk estimation. 

The goal is to construct an explainable, reliable binary classifier that separates **Normal** patients from those exhibiting **Thyroid Abnormalities** while maintaining high Recall to minimize missed cases.

---

## 🚀 Main Objectives
1. **Data Ingestion**: Load the canonical UCI Thyroid Disease (Hypothyroid) benchmark dataset.
2. **Exploratory Data Analysis (EDA)**: Understand class distributions, feature correlations, and hormone ranges through clean visualizations.
3. **Robust Preprocessing**: Build Scikit-Learn `ColumnTransformer` and `Pipeline` objects to prevent data leakage between training and testing sets.
4. **Multi-Model Training**: Benchmark Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors.
5. **Rigorous Evaluation**: Evaluate models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
6. **Model Persistence**: Serialize the best pipeline using `joblib`.
7. **Interactive Web Interface**: Deploy a Streamlit web app allowing users to input patient vitals and test results.

---

## 📊 Dataset Description
The project utilizes the **UCI Thyroid Disease (Hypothyroid) Dataset**, a standard medical machine learning benchmark originally collected at Sydney Hospitals and the Garvan Institute.

- **Total Records**: 3,772 rows
- **Raw Columns**: 30 (7 numerical, 23 categorical/indicator)
- **Target Variable**: `Class`
  - `negative` (3,481 samples): Normal thyroid function.
  - `compensated_hypothyroid` (194 samples): Mild/subclinical hypothyroidism where elevated TSH compensates to normalize T4.
  - `primary_hypothyroid` (95 samples): Overt hypothyroidism (high TSH, low T4).
  - `secondary_hypothyroid` (2 samples): Rare central pituitary failure.
- **Problem Formulation**:
  Binary classification:
  - **`0 = Normal`**: (`negative`)
  - **`1 = Thyroid Abnormality`**: (`compensated_hypothyroid`, `primary_hypothyroid`, `secondary_hypothyroid`)

### Key Clinical Features Used
| Feature | Type | Description / Normal Reference Range |
|---|---|---|
| `age` | Numerical | Patient age in years |
| `sex` | Categorical | Biological sex (`F` / `M`) |
| `TSH` | Numerical | Thyroid Stimulating Hormone (`0.4 – 4.0 μIU/mL`) |
| `T3` | Numerical | Triiodothyronine (`1.2 – 3.1 nmol/L`) |
| `TT4` | Numerical | Total Thyroxine (`60 – 150 nmol/L`) |
| `T4U` | Numerical | Thyroxine Uptake ratio (`0.8 – 1.3`) |
| `FTI` | Numerical | Free Thyroxine Index (`70 – 140`) |
| Clinical History Flags | Categorical | `on_thyroxine`, `query_hypothyroid`, `thyroid_surgery`, `sick`, `pregnant`, `tumor`, `goitre`, etc. (`f` / `t`) |

---

## 🧹 Data Cleaning & Leakage Prevention

### Cleaning Steps Taken:
1. **Dropped `TBG` and `TBG_measured`**: The `TBG` column contains 100% missing values (0 non-null values) and was safely removed.
2. **Dropped `referral_source`**: Administrative routing codes (`SVHC`, `SVI`, etc.) have no physiological relevance and cause hospital-specific bias.
3. **Dropped `*_measured` indicators**: Redundant flags (`TSH_measured`, `T3_measured`, etc.) that merely duplicated whether the test was performed.
4. **Handled Erroneous Age Outlier**: A known transcription typo in the UCI dataset (`age = 455`) was detected and converted to `NaN`, then imputed via the median.
5. **Removed Duplicate Records**: 61 exact duplicate rows were removed, resulting in 3,711 unique records.

### Preventing Data Leakage:
- **Never fit imputers or scalers on the entire dataset.**
- A Scikit-Learn `ColumnTransformer` combines:
  - **Numerical Pipeline**: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - **Categorical Pipeline**: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
- All transformations are **fitted strictly on `X_train`** during pipeline fitting and applied blindly to `X_test` and production user inputs.

---

## 🤖 Machine Learning Algorithms

Four classic machine learning algorithms were trained and compared:

1. **Logistic Regression**:
   - Linear probabilistic model using the sigmoid function. Serves as our baseline classifier.
   - Configured with `class_weight='balanced'` to accommodate the ~7.8% minority class.
2. **Decision Tree Classifier**:
   - Non-parametric rule-based model using recursive binary splitting.
   - Configured with `max_depth=5` to prevent overfitting and ensure easy human explainability.
3. **Random Forest Classifier**:
   - Ensemble bagging algorithm creating 100 decorrelated decision trees with random feature subspacing.
   - Combines multiple weak learners to drastically reduce variance.
4. **K-Nearest Neighbors (KNN)**:
   - Instance-based, distance-weighted classifier ($k=5$) identifying similar past patient vectors.

---

## 📈 Experimental Results & Evaluation

All models were evaluated on the **same stratified 20% holdout test set (743 unseen samples: 685 Normal, 58 Abnormal)**.

> Note: All numbers below are **real, calculated test-set metrics** from actual script execution.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 97.98% | 83.08% | 93.10% | 87.80% | 0.9955 |
| **Decision Tree** | 99.60% | 96.61% | 98.28% | 97.44% | 0.9893 |
| **Random Forest (Selected)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** |
| **K-Nearest Neighbors** | 94.62% | 87.50% | 36.21% | 51.22% | 0.8028 |

### Confusion Matrix for Selected Model (Random Forest)
```text
                  Predicted Normal    Predicted Abnormal
Actual Normal           685                   0
Actual Abnormal           0                  58
```
- **True Negatives (TN)**: 685 (Normal patients correctly identified)
- **False Positives (FP)**: 0 (No false alarms)
- **False Negatives (FN)**: 0 (No abnormal cases missed)
- **True Positives (TP)**: 58 (All abnormal cases detected)

### Model Selection Rationale
Random Forest was selected as the final deployed model because it achieved the highest F1-Score and ROC-AUC. In clinical screening, **Recall** is critical: missing an abnormality (False Negative) can delay care, while **Precision** prevents unnecessary emotional distress and secondary testing costs. Random Forest delivered optimal performance across both dimensions.

---

## 🔍 Feature Importance

Feature importance extracted from the trained Random Forest ensemble reveals which variables contributed most to the splitting decisions:

```text
Feature                Importance (%)
--------------------------------------
TSH                    ████████████████████████████  58.95%
FTI                    ███████                       14.56%
TT4                    █████                         10.61%
T3                     ██                             4.69%
T4U                    █                              1.99%
age                    █                              1.95%
on_thyroxine           █                              1.77%
sex                    ░                              0.60%
query_hypothyroid      ░                              0.58%
```

> **Statistical vs. Medical Note**: Feature importance measures mathematical entropy reduction across decision trees. While aligning strongly with clinical endocrinology (TSH, FTI, and TT4 are the hallmark diagnostic markers for thyroid disease), importance reflects algorithmic association, not medical causation.

---

## 🏗️ Project Architecture & Workflow

```text
                 ┌─────────────────────────────────┐
                 │       UCI Thyroid Dataset       │
                 │      (data/thyroid.csv)         │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │    Data Cleaning & Validation   │
                 │  (Drop TBG, clean age outliers) │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │    Stratified Train/Test Split  │
                 │       (80% Train, 20% Test)     │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │     Preprocessing Pipeline      │
                 │ (SimpleImputer + StandardScaler │
                 │      + OneHotEncoder)           │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │    Model Training & Benchmark   │
                 │  (LR, Decision Tree, RF, KNN)   │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │     Model Evaluation & Choice   │
                 │  (F1-Score, ROC-AUC, CM matrix) │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │  Joblib Pipeline Serialization  │
                 │   (models/thyroid_model.pkl)    │
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │     Streamlit Web Interface     │
                 │   (User Input -> Prediction)    │
                 └─────────────────────────────────┘
```

---

## 📁 Folder Structure

```text
ThyroPredict/
│
├── data/
│   └── thyroid.csv              # Benchmark UCI Thyroid dataset
│
├── notebooks/
│   └── thyroid_analysis.ipynb   # Executed analysis notebook with 5 EDA plots
│
├── models/
│   ├── thyroid_model.pkl        # Serialized preprocessor + Random Forest pipeline
│   └── model_metadata.json      # Saved evaluation metrics & feature metadata
│
├── app.py                       # Streamlit web application
├── train_model.py               # Reproducible end-to-end training script
├── requirements.txt             # Python project dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore configurations
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.9 or higher (tested on Python 3.13)
- `pip` package manager

### 1. Clone or Open the Project
```bash
cd ThyroPredict
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ How to Train the Model

To execute the data cleaning, train all candidate models, display the real comparison table, and save the fitted pipeline:

```bash
python train_model.py
```

Expected output:
```text
======================================================================
      THYROPREDICT — MODEL TRAINING & EVALUATION PIPELINE
======================================================================

[Step 1/6] Loading and cleaning dataset...
 -> Cleaned dataset: 3711 rows, 21 features
 -> Target distribution: 3420 Normal, 291 Thyroid Abnormality (7.84%)
[Step 2/6] Performing Stratified 80/20 Train-Test Split...
[Step 3/6] Training and evaluating candidate classification models...
...
[Step 4/6] Best Model Selected: Random Forest
...
[Step 6/6] Saving trained pipeline and project metadata...
 -> Saved complete Scikit-Learn pipeline to: models/thyroid_model.pkl
```

---

## 🌐 How to Run the Web Application

To launch the interactive web interface:

```bash
streamlit run app.py
```

The browser will open automatically at: `http://localhost:8501`

### Features in the Web App:
- **One-Click Test Presets**: Test normal or abnormal patient samples with single clicks.
- **Reference Range Helper Text**: Displays standard biochemical reference intervals for all blood tests.
- **Validation**: Enforces positive age and non-negative hormone values.
- **Probability Output**: Displays model confidence percentages alongside clear class classifications.
- **Prominent Medical Disclaimer**: Clear notice that the app is an educational prototype.

---

## 🎓 Viva & Interview Questions & Answers

### 1. What is the main objective of the ThyroPredict project?
**Answer**: To build an educational, end-to-end machine learning system that analyzes patient clinical and biochemical thyroid hormone markers (such as TSH, T3, TT4, FTI) to predict whether a patient has normal thyroid function or a thyroid abnormality.

### 2. Which dataset did you use and where did it come from?
**Answer**: We used the UCI Thyroid Disease (Hypothyroid) benchmark dataset, originally collected by Ross Quinlan at the Garvan Institute and Sydney Hospitals, containing 3,772 patient records.

### 3. Why did you convert the problem into binary classification?
**Answer**: The original dataset has 4 classes: `negative` (3,481), `compensated_hypothyroid` (194), `primary_hypothyroid` (95), and `secondary_hypothyroid` (2). Because `secondary_hypothyroid` has only 2 samples, multiclass stratification is unstable. Grouping them into `Normal` vs. `Thyroid Abnormality` creates a robust screening tool suitable for clinical triage.

### 4. What is data leakage and how did you prevent it?
**Answer**: Data leakage occurs when information from the test dataset leaks into the training process (e.g., calculating median or scaling parameters across the entire dataset before splitting). We prevented this by using Scikit-Learn's `Pipeline` and `ColumnTransformer`, which fit imputers and scalers exclusively on `X_train` and apply those learned parameters to `X_test`.

### 5. Why did you drop the `TBG` column?
**Answer**: In this dataset, `TBG` (Thyroxine-Binding Globulin) had 0 non-null values (100% missing). A feature with zero data points cannot provide predictive value and must be removed.

### 6. What was the outlier in the `age` column and how did you treat it?
**Answer**: There was a record with `age = 455`, which is a well-known typographical error in the UCI repository. We converted this value to `NaN` and allowed the median imputer inside the training pipeline to impute a realistic value.

### 7. Why did you use stratified train/test splitting?
**Answer**: The dataset is imbalanced (~7.8% abnormalities vs. ~92.2% normal). Regular random splitting could lead to unequal class proportions in test and train sets. Stratified splitting preserves the exact 7.8% proportion across both splits.

### 8. What features were most important to the model?
**Answer**: TSH (Thyroid Stimulating Hormone) was by far the most influential feature (~58.95%), followed by FTI (Free Thyroxine Index, ~14.56%) and TT4 (Total Thyroxine, ~10.61%).

### 9. Does high feature importance imply medical causation?
**Answer**: No. Feature importance measures how often and how effectively a feature was used to split nodes in decision trees (information gain / Gini impurity reduction). It indicates statistical association within the dataset, not clinical causation.

### 10. Why is Accuracy alone not a reliable metric for this dataset?
**Answer**: Because 92.2% of the patients are normal, a trivial classifier predicting "Normal" for everyone would achieve 92.2% accuracy while completely failing to detect any diseased patients. Metrics like Recall, Precision, and F1-Score are far more informative.

### 11. What is the difference between Precision and Recall in this context?
**Answer**: 
- **Recall**: Out of all patients who genuinely have a thyroid abnormality, what percentage did the model catch? High recall minimizes missed diagnoses (False Negatives).
- **Precision**: Out of all patients the model flagged as abnormal, what percentage actually have an abnormality? High precision minimizes false alarms (False Positives).

### 12. What is the F1-Score?
**Answer**: The harmonic mean of Precision and Recall ($F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$). It provides a single balanced metric when both false positives and false negatives carry significant costs.

### 13. What is ROC-AUC?
**Answer**: Receiver Operating Characteristic - Area Under the Curve measures the model's ability to rank positive instances higher than negative instances across all possible classification probability thresholds. An AUC of 1.0 represents perfect discrimination.

### 14. How does Logistic Regression work?
**Answer**: It models the log-odds of the positive class as a linear combination of input features, transforming the output between 0 and 1 using the Sigmoid (logistic) function: $\sigma(z) = \frac{1}{1 + e^{-z}}$.

### 15. How does a Decision Tree work?
**Answer**: It recursively partitions the feature space into regions by selecting thresholds that maximize purity (using Gini Impurity or Entropy) at each split.

### 16. How does Random Forest improve upon a single Decision Tree?
**Answer**: A single decision tree has high variance and is prone to overfitting. Random Forest builds an ensemble of many decision trees using **bagging** (bootstrap aggregation) and **random feature subspaces**, then averages their predictions. This decorrelates the trees and drastically reduces variance.

### 17. How does K-Nearest Neighbors work?
**Answer**: KNN is a non-parametric, lazy-learning algorithm that classifies a new patient based on the majority class among the $k$ closest data points in the normalized feature space using Euclidean distance.

### 18. Why is StandardScaler used for numerical features?
**Answer**: Features like TSH and TT4 have vastly different scales (TSH ranges 0–500, TT4 ranges 0–450, while T4U is around 1.0). StandardScaler transforms them to mean=0 and variance=1, which is essential for distance-based models (KNN) and gradient-based models (Logistic Regression).

### 19. Why did you use Joblib instead of Python's built-in Pickle?
**Answer**: `joblib` is optimized for serializing large NumPy arrays and Scikit-Learn pipelines containing multiple transformers and estimators more efficiently than standard `pickle`.

### 20. What is the role of Streamlit in this project?
**Answer**: Streamlit provides a Python-native web interface that takes patient parameters via widgets, packages them into a DataFrame, passes them through the pre-fitted Scikit-Learn pipeline, and renders user-friendly prediction cards with probability scores.

---

## 📜 License & Acknowledgments
- **Dataset**: UCI Machine Learning Repository (Thyroid Disease Database).
- **Educational Prototype**: Created for academic demonstration and machine learning coursework.
