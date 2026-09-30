"""
train_model.py
==============
ThyroPredict — AI-Based Thyroid Abnormality Prediction System

This standalone, reproducible script:
1. Loads the UCI Thyroid Disease dataset (data/thyroid.csv).
2. Cleans the data (handles outliers, drops 100% missing columns, removes duplicates).
3. Constructs a Scikit-Learn ColumnTransformer and Pipeline to prevent data leakage.
4. Splits data into 80% train and 20% test with stratification.
5. Trains and evaluates multiple ML classification models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - K-Nearest Neighbors Classifier
6. Compares actual metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
7. Selects the best performing model (Random Forest) and saves the complete
   pipeline to models/thyroid_model.pkl.
8. Saves metadata and evaluation results to models/model_metadata.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'thyroid.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'thyroid_model.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')


def load_and_preprocess_data(file_path):
    """
    Loads dataset, handles missing data, cleans anomalies, and maps target.
    
    Returns:
        X (DataFrame): Features
        y (Series): Binary target (0: Normal, 1: Thyroid Abnormality)
        num_cols (list): List of numerical column names
        cat_cols (list): List of categorical column names
        cleaning_stats (dict): Dictionary summarizing cleaning actions
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please place thyroid.csv in data/")
    
    df = pd.read_csv(file_path)
    initial_rows, initial_cols = df.shape
    
    # 1. Drop TBG and TBG_measured because TBG is 100% missing (0 non-null values)
    # 2. Drop referral_source (administrative routing metadata, not clinical)
    # 3. Drop redundant measurement flags (*_measured) since the numeric value presence indicates measurement
    cols_to_drop = [
        'TBG', 'TBG_measured', 'referral_source',
        'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 4. Handle biological anomaly in age: in UCI dataset, age 455 is a known transcription typo
    # Replace age > 100 with NaN so it is imputed mathematically during pipeline execution
    invalid_age_count = (df['age'] > 100).sum()
    df.loc[df['age'] > 100, 'age'] = np.nan
    
    # 5. Remove duplicate records
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 6. Map Target:
    # 'negative' -> Normal (0)
    # 'compensated_hypothyroid', 'primary_hypothyroid', 'secondary_hypothyroid' -> Thyroid Abnormality (1)
    df['target'] = (df['Class'] != 'negative').astype(int)
    
    y = df['target']
    X = df.drop(columns=['Class', 'target'])
    
    # Define feature groups
    num_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    cleaning_stats = {
        'initial_shape': [initial_rows, initial_cols],
        'final_shape': list(df.shape),
        'duplicates_removed': int(dup_count),
        'invalid_ages_handled': int(invalid_age_count),
        'normal_count': int((y == 0).sum()),
        'abnormal_count': int((y == 1).sum()),
        'abnormal_percentage': float(round((y == 1).mean() * 100, 2))
    }
    
    return X, y, num_cols, cat_cols, cleaning_stats


def create_preprocessor(num_cols, cat_cols):
    """
    Creates a ColumnTransformer to prevent data leakage.
    Numerical: Median imputation + StandardScaler
    Categorical: Most-frequent imputation + OneHotEncoder
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    return preprocessor


def train_and_evaluate_models():
    """
    Main training workflow. Trains models, evaluates them, prints real metrics,
    and saves the best pipeline.
    """
    print("=" * 70)
    print("      THYROPREDICT — MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)
    
    # 1. Load and clean data
    print("\n[Step 1/6] Loading and cleaning dataset...")
    X, y, num_cols, cat_cols, stats = load_and_preprocess_data(DATA_PATH)
    print(f" -> Initial shape: {stats['initial_shape'][0]} rows, {stats['initial_shape'][1]} columns")
    print(f" -> Duplicates removed: {stats['duplicates_removed']}")
    print(f" -> Invalid age entries (>100) corrected: {stats['invalid_ages_handled']}")
    print(f" -> Cleaned dataset: {len(X)} rows, {len(X.columns)} features")
    print(f" -> Target distribution: {stats['normal_count']} Normal, {stats['abnormal_count']} Thyroid Abnormality ({stats['abnormal_percentage']}%)")
    
    # 2. Train-test split (80% train, 20% test, stratified)
    print("\n[Step 2/6] Performing Stratified 80/20 Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f" -> Training set size: {len(X_train)} samples")
    print(f" -> Testing set size:  {len(X_test)} samples (Abnormal: {y_test.sum()}, Normal: {(y_test == 0).sum()})")
    
    # 3. Create Preprocessing Pipeline
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    # 4. Define candidate models
    candidate_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, class_weight='balanced'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5
        )
    }
    
    # 5. Train and evaluate all models
    print("\n[Step 3/6] Training and evaluating candidate classification models...")
    results = {}
    fitted_pipelines = {}
    
    for name, clf in candidate_models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Fit ONLY on training data to prevent data leakage
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Predict on unseen test set
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        
        # Calculate real metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1_Score': float(f1),
            'ROC_AUC': float(auc),
            'Confusion_Matrix': cm
        }
    
    # Display comparison table
    print("\n" + "=" * 70)
    print("                    MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 74)
    for name, m in results.items():
        print(f"{name:<22} | {m['Accuracy']*100:6.2f}%   | {m['Precision']*100:6.2f}%   | {m['Recall']*100:6.2f}%  | {m['F1_Score']*100:6.2f}%  | {m['ROC_AUC']:6.4f}")
    print("=" * 74)
    
    # 6. Select Best Model
    # Selection criteria: Highest F1-Score & ROC-AUC on imbalanced clinical data
    best_model_name = max(results, key=lambda k: (results[k]['F1_Score'], results[k]['ROC_AUC']))
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n[Step 4/6] Best Model Selected: {best_model_name}")
    print(f" -> Reason: Achieved highest F1-Score ({best_metrics['F1_Score']*100:.2f}%) and ROC-AUC ({best_metrics['ROC_AUC']:.4f}),")
    print("    providing optimal balance between detecting abnormalities (high recall) and avoiding false alarms (high precision).")
    
    # Confusion Matrix breakdown for best model
    cm = best_metrics['Confusion_Matrix']
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"\n -> Confusion Matrix for {best_model_name}:")
    print(f"    True Negatives (Normal correctly identified):        {tn}")
    print(f"    False Positives (Normal predicted as Abnormality):   {fp}")
    print(f"    False Negatives (Abnormality missed):               {fn}")
    print(f"    True Positives (Abnormality correctly detected):    {tp}")
    
    # 7. Extract Feature Importances (if applicable)
    print("\n[Step 5/6] Extracting Feature Importance from Best Model...")
    feature_importances = {}
    if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
        cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        all_features = num_cols + cat_feature_names
        importances = best_pipeline.named_steps['classifier'].feature_importances_
        
        feat_series = pd.Series(importances, index=all_features).sort_values(ascending=False)
        feature_importances = {k: float(v) for k, v in feat_series.items()}
        
        print("\nTop 10 Most Important Features:")
        for feat, imp in list(feature_importances.items())[:10]:
            print(f"  - {feat:<20}: {imp*100:5.2f}%")
        print("\nNote: Feature importance reflects statistical utility for the decision tree splits, NOT medical causation.")
    
    # 8. Save Pipeline and Metadata
    print("\n[Step 6/6] Saving trained pipeline and project metadata...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f" -> Saved complete Scikit-Learn pipeline to: {MODEL_PATH}")
    
    metadata = {
        'project_title': 'ThyroPredict — AI-Based Thyroid Abnormality Prediction System',
        'best_model': best_model_name,
        'metrics': best_metrics,
        'all_models_evaluated': results,
        'feature_columns': list(X.columns),
        'numerical_columns': num_cols,
        'categorical_columns': cat_cols,
        'target_classes': {0: 'Normal', 1: 'Thyroid Abnormality'},
        'top_features': list(feature_importances.items())[:10],
        'cleaning_statistics': stats
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f" -> Saved model metadata and real metrics to: {METADATA_PATH}")
    
    print("\n" + "=" * 70)
    print("               MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    train_and_evaluate_models()
