
from mostlyai import qa
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
import json
import warnings

"""Evaluation helpers for similarity, utility, and report persistence."""

'''
Resources:
- https://arxiv.org/html/2501.03941v1
- https://www.nature.com/articles/s41746-023-00771-5 uses median for NNDR
'''

def get_metrics(data_train, data_gen, data_holdout):
    # Run MOSTLY AI QA report and keep only structured metrics output.
    _, metrics = qa.report(
        syn_tgt_data=data_gen,
        trn_tgt_data=data_train,
        hol_tgt_data=data_holdout,
    )
    # Drop the HTML side artifact to keep runs clean/reproducible.
    os.remove('model-report.html')
    return metrics

def run_tstr_evaluation(
    real_data: pd.DataFrame, 
    synthetic_data: pd.DataFrame, 
    target_column: str, 
    model, 
    metric_func,
    test_size: float = 0.3, 
    random_state: int = 42
):
    """
    Performs the "Train on Synthetic, Test on Real" (TSTR) evaluation
    with automatic pre-processing for string-based columns.
    """
    
    # --- 1. Split the real data into train and test (R_test) ---
    try:
        real_train, real_test = train_test_split(
            real_data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=real_data[target_column]
        )
    except ValueError:
        warnings.warn(f"Could not stratify on '{target_column}'. Proceeding without stratification.")
        real_train, real_test = train_test_split(
            real_data, 
            test_size=test_size, 
            random_state=random_state
        )

    # --- 2. Prepare data for ML (Separation) ---
    X_real_train = real_train.drop(columns=[target_column])
    y_real_train = real_train[target_column]
    X_real_test = real_test.drop(columns=[target_column])
    y_real_test = real_test[target_column]
    
    X_synthetic_train = synthetic_data.drop(columns=[target_column])
    y_synthetic_train = synthetic_data[target_column]

    # --- 3. NEW: Pre-processing (Encoding) ---
    
    # Find categorical (string) features
    categorical_features = X_real_train.select_dtypes(include=['object', 'category']).columns
    numeric_features = X_real_train.select_dtypes(include=np.number).columns

    # Create encoders
    # OrdinalEncoder for features
    feature_preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # LabelEncoder for the target column
    target_encoder = LabelEncoder()

    # Fit encoders on REAL training data
    X_real_train = feature_preprocessor.fit_transform(X_real_train)
    y_real_train = target_encoder.fit_transform(y_real_train)
    
    # Transform test and synthetic data using the *fitted* encoders
    X_real_test = feature_preprocessor.transform(X_real_test)
    y_real_test = target_encoder.transform(y_real_test)
    
    X_synthetic_train = feature_preprocessor.transform(X_synthetic_train)
    y_synthetic_train = target_encoder.transform(y_synthetic_train)
    
    # --- 4. Train M_R (Model on Real) ---
    model_real = clone(model)
    model_real.fit(X_real_train, y_real_train)
    
    # --- 5. Train M_S (Model on Synthetic) ---
    model_synthetic = clone(model)
    model_synthetic.fit(X_synthetic_train, y_synthetic_train)
    
    # --- 6. Evaluate both on R_test ---
    # Compare reference (real-trained) vs synthetic-trained model behavior.
    preds_real = model_real.predict(X_real_test)
    score_real = metric_func(y_real_test, preds_real)
    
    preds_synthetic = model_synthetic.predict(X_real_test)
    score_synthetic = metric_func(y_real_test, preds_synthetic)
    
    # --- 7. Calculate performance gap ---
    if score_real == 0:
        performance_gap_pct = float('inf') if score_synthetic != 0 else 0.0
    else:
        performance_gap = (score_real - score_synthetic) / score_real
        performance_gap_pct = performance_gap * 100
        
    return score_real, score_synthetic, performance_gap_pct

def evaluate_and_save_reports(original_real_data, synthetic_data, metadata, report_path, metrics_qa, training_time, evaluation_time, tstr_results):
    """Generates reports, saves them, and returns the scores as a dictionary."""
    print(f"--- Evaluating and saving reports to '{report_path}' ---")
    diagnostic_report = run_diagnostic(original_real_data, synthetic_data, metadata)
    quality_report = evaluate_quality(original_real_data, synthetic_data, metadata)
    
    combined_report_data = {
        'diagnostic_report': {'properties': diagnostic_report.get_properties().to_dict('records')},
        'quality_report': {
            'overall_score': quality_report.get_score(),
            'properties': quality_report.get_properties().to_dict('records')
        },
        'metrics_qa': { "overall_accuracy": metrics_qa.accuracy.overall,
            "univariate_accuracy": metrics_qa.accuracy.univariate,
            "bivariate_accuracy": metrics_qa.accuracy.bivariate,
            "discriminator_auc": metrics_qa.similarity.discriminator_auc_training_synthetic,
            "identical_matches": metrics_qa.distances.ims_training,
            "dcr_training": metrics_qa.distances.dcr_training,
            "dcr_holdout": metrics_qa.distances.dcr_holdout,
            "dcr_share": metrics_qa.distances.dcr_share
        },
        "times":{"training_time": training_time,
                 "evaluation_time": evaluation_time},
        
        # --- ADDED THIS LINE ---
        "tstr_evaluation": tstr_results
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(combined_report_data, f, indent=4)
    print(f"Combined report saved successfully.")
    return combined_report_data
