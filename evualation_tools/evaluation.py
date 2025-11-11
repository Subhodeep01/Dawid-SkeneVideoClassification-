"""
Evaluation Script for Video Classification Models

This script calculates and compares performance metrics for:
- Individual classifier models
- Ensemble predictions (majority voting)
- Dawid-Skene predictions

Metrics calculated: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import os
from typing import Dict, Tuple
import ast


def load_ground_truth(filepath: str = 'classifiers/sampled_labels.csv') -> pd.DataFrame:
    """Load ground truth labels."""
    df = pd.read_csv(filepath)
    # Ensure consistent naming
    if 'filename' in df.columns:
        df = df.rename(columns={'filename': 'video_name', 'label': 'true_label'})
    return df[['video_name', 'true_label']]


def load_predictions(filepath: str, model_name: str) -> pd.DataFrame:
    """Load predictions from a CSV file."""
    df = pd.read_csv(filepath)
    
    # Handle different column names
    if 'predicted_class' not in df.columns and 'ensemble_prediction' in df.columns:
        df = df.rename(columns={'ensemble_prediction': 'predicted_class'})
    
    # Handle Dawid-Skene format with dictionary strings
    if 'predicted_class' in df.columns:
        # Check if first entry is a dictionary string
        first_pred = str(df['predicted_class'].iloc[0])
        if first_pred.startswith('{'):
            # Parse dictionary and extract class name
            def extract_class(pred_str):
                try:
                    pred_dict = ast.literal_eval(pred_str)
                    return list(pred_dict.keys())[0]
                except:
                    return pred_str
            
            df['predicted_class'] = df['predicted_class'].apply(extract_class)
    
    df['model'] = model_name
    return df[['video_name', 'predicted_class', 'model']]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     labels: list) -> Dict[str, float]:
    """
    Calculate performance metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of all possible labels
        
    Returns:
        Dictionary of metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Use macro averaging for multi-class classification
    precision = precision_score(y_true, y_pred, labels=labels, 
                               average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, 
                         average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, 
                  average='macro', zero_division=0)
    
    # Also calculate weighted averages
    precision_weighted = precision_score(y_true, y_pred, labels=labels,
                                        average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=labels,
                                  average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels,
                          average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: list, model_name: str, 
                         output_dir: str = 'evaluation_results'):
    """
    Save confusion matrix to CSV.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of all possible labels
        model_name: Name of the model
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create DataFrame with row and column labels
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    # Save to CSV
    output_file = os.path.join(output_dir, f'{model_name}_confusion_matrix.csv')
    cm_df.to_csv(output_file)
    print(f"  Saved confusion matrix to: {output_file}")
    
    return cm


def evaluate_all_models(ground_truth_file: str = 'classifiers/sampled_labels.csv',
                       predictions_dir: str = 'predictions',
                       output_dir: str = 'evaluation_results') -> pd.DataFrame:
    """
    Evaluate all models and generate comparison table.
    
    Args:
        ground_truth_file: Path to ground truth CSV
        predictions_dir: Directory containing prediction CSVs
        output_dir: Directory to save evaluation results
        
    Returns:
        DataFrame with all model metrics
    """
    print("=" * 80)
    print("VIDEO CLASSIFICATION MODEL EVALUATION")
    print("=" * 80)
    
    # Load ground truth
    print(f"\nLoading ground truth from: {ground_truth_file}")
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"  Found {len(ground_truth)} labeled videos")
    
    # Get all unique labels for metrics calculation
    all_labels = sorted(ground_truth['true_label'].unique())
    print(f"  Number of classes: {len(all_labels)}")
    
    # Dictionary to store all predictions
    all_predictions = {}
    results = []
    
    # Individual model predictions
    print(f"\n{'=' * 80}")
    print("INDIVIDUAL MODELS")
    print("=" * 80)
    
    model_files = {
        'GPT-4o-mini': 'predictions/gpt4o_predictions.csv',
        'GPT-5-mini': 'predictions/gpt-5-mini_predictions.csv',
        'Gemini': 'predictions/gemini_predictions.csv',
        'Replicate': 'predictions/replicate_predictions.csv',
        'MoonDream2': 'predictions/moondream_predictions.csv',
        'Qwen-VL': 'predictions/qwen_predictions.csv',
        'TwelveLabs': 'predictions/twelvelabs_predictions.csv',
    }
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            print(f"\nEvaluating: {model_name}")
            print(f"  File: {filepath}")
            
            # Load predictions
            preds = load_predictions(filepath, model_name)
            
            # Merge with ground truth
            merged = ground_truth.merge(preds, on='video_name', how='inner')
            
            if len(merged) == 0:
                print(f"  WARNING: No matching videos found!")
                continue
            
            print(f"  Evaluating {len(merged)} videos")
            
            # Calculate metrics
            metrics = calculate_metrics(
                merged['true_label'].values,
                merged['predicted_class'].values,
                all_labels
            )
            
            # Save confusion matrix
            save_confusion_matrix(
                merged['true_label'].values,
                merged['predicted_class'].values,
                all_labels,
                model_name.lower().replace('-', '_'),
                output_dir
            )
            
            # Add to results
            result_row = {'Model': model_name, **metrics}
            results.append(result_row)
            
            # Store predictions for later analysis
            all_predictions[model_name] = merged
            
            # Print summary
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            
        else:
            print(f"\nSkipping {model_name}: File not found - {filepath}")
    
    # Ensemble predictions
    print(f"\n{'=' * 80}")
    print("ENSEMBLE (Majority Voting)")
    print("=" * 80)
    
    ensemble_file = 'ensemble_predictions.csv'
    if os.path.exists(ensemble_file):
        print(f"\nEvaluating: Ensemble")
        print(f"  File: {ensemble_file}")
        
        preds = load_predictions(ensemble_file, 'Ensemble')
        merged = ground_truth.merge(preds, on='video_name', how='inner')
        
        print(f"  Evaluating {len(merged)} videos")
        
        metrics = calculate_metrics(
            merged['true_label'].values,
            merged['predicted_class'].values,
            all_labels
        )
        
        save_confusion_matrix(
            merged['true_label'].values,
            merged['predicted_class'].values,
            all_labels,
            'ensemble',
            output_dir
        )
        
        result_row = {'Model': 'Ensemble', **metrics}
        results.append(result_row)
        all_predictions['Ensemble'] = merged
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    
    # Dawid-Skene predictions
    print(f"\n{'=' * 80}")
    print("DAWID-SKENE")
    print("=" * 80)
    
    dawid_skene_file = 'dawid_skene_predictions.csv'
    if os.path.exists(dawid_skene_file):
        print(f"\nEvaluating: Dawid-Skene")
        print(f"  File: {dawid_skene_file}")
        
        preds = load_predictions(dawid_skene_file, 'Dawid-Skene')
        merged = ground_truth.merge(preds, on='video_name', how='inner')
        
        print(f"  Evaluating {len(merged)} videos")
        
        metrics = calculate_metrics(
            merged['true_label'].values,
            merged['predicted_class'].values,
            all_labels
        )
        
        save_confusion_matrix(
            merged['true_label'].values,
            merged['predicted_class'].values,
            all_labels,
            'dawid_skene',
            output_dir
        )
        
        result_row = {'Model': 'Dawid-Skene', **metrics}
        results.append(result_row)
        all_predictions['Dawid-Skene'] = merged
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by accuracy (descending)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'model_comparison.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nComparison table saved to: {output_file}")
    
    # Display formatted table
    print("\nModel Performance Comparison:")
    print("-" * 120)
    
    # Format for display
    display_df = results_df.copy()
    for col in display_df.columns:
        if col != 'Model':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    print("-" * 120)
    
    # Create detailed comparison table
    detailed_output = os.path.join(output_dir, 'detailed_comparison.csv')
    results_df.to_csv(detailed_output, index=False, float_format='%.6f')
    print(f"\nDetailed metrics saved to: {detailed_output}")
    
    # Generate per-class performance report
    print(f"\n{'=' * 80}")
    print("PER-CLASS PERFORMANCE")
    print("=" * 80)
    
    per_class_reports = []
    
    for model_name, merged_df in all_predictions.items():
        print(f"\n{model_name}:")
        report = classification_report(
            merged_df['true_label'].values,
            merged_df['predicted_class'].values,
            labels=all_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Save detailed report
        report_file = os.path.join(output_dir, f'{model_name.lower().replace("-", "_").replace(" ", "_")}_classification_report.csv')
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(report_file)
        print(f"  Detailed report saved to: {report_file}")
    
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"  - model_comparison.csv: Summary metrics for all models")
    print(f"  - detailed_comparison.csv: Detailed metrics with more precision")
    print(f"  - *_confusion_matrix.csv: Confusion matrices for each model")
    print(f"  - *_classification_report.csv: Per-class metrics for each model")
    
    return results_df


def generate_summary_statistics(results_df: pd.DataFrame, 
                               output_dir: str = 'evaluation_results'):
    """
    Generate additional summary statistics and insights.
    
    Args:
        results_df: DataFrame with model comparison results
        output_dir: Directory to save results
    """
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Best model for each metric
    print("\nBest Models by Metric:")
    print("-" * 60)
    
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    best_models = {}
    
    for metric in metrics:
        if metric in results_df.columns:
            best_idx = results_df[metric].idxmax()
            best_model = results_df.loc[best_idx, 'Model']
            best_value = results_df.loc[best_idx, metric]
            best_models[metric] = (best_model, best_value)
            print(f"  {metric.replace('_', ' ').title()}: {best_model} ({best_value:.4f})")
    
    # Save best models summary
    best_models_df = pd.DataFrame([
        {'Metric': metric.replace('_', ' ').title(), 
         'Best Model': model, 
         'Score': f"{score:.6f}"}
        for metric, (model, score) in best_models.items()
    ])
    
    best_models_file = os.path.join(output_dir, 'best_models_by_metric.csv')
    best_models_df.to_csv(best_models_file, index=False)
    print(f"\nBest models summary saved to: {best_models_file}")
    
    # Performance gaps
    print("\nPerformance Analysis:")
    print("-" * 60)
    
    if 'accuracy' in results_df.columns:
        max_acc = results_df['accuracy'].max()
        min_acc = results_df['accuracy'].min()
        mean_acc = results_df['accuracy'].mean()
        std_acc = results_df['accuracy'].std()
        
        print(f"  Accuracy - Max: {max_acc:.4f}, Min: {min_acc:.4f}, "
              f"Mean: {mean_acc:.4f}, Std: {std_acc:.4f}")
        print(f"  Accuracy Range: {max_acc - min_acc:.4f}")
        
        # Check if ensemble/Dawid-Skene outperform individual models
        individual_models = results_df[~results_df['Model'].isin(['Ensemble', 'Dawid-Skene'])]
        if len(individual_models) > 0:
            best_individual_acc = individual_models['accuracy'].max()
            
            ensemble_row = results_df[results_df['Model'] == 'Ensemble']
            if len(ensemble_row) > 0:
                ensemble_acc = ensemble_row['accuracy'].values[0]
                improvement = ensemble_acc - best_individual_acc
                print(f"\n  Ensemble vs Best Individual:")
                print(f"    Ensemble: {ensemble_acc:.4f}")
                print(f"    Best Individual: {best_individual_acc:.4f}")
                print(f"    Improvement: {improvement:+.4f} ({improvement/best_individual_acc*100:+.2f}%)")
            
            ds_row = results_df[results_df['Model'] == 'Dawid-Skene']
            if len(ds_row) > 0:
                ds_acc = ds_row['accuracy'].values[0]
                improvement = ds_acc - best_individual_acc
                print(f"\n  Dawid-Skene vs Best Individual:")
                print(f"    Dawid-Skene: {ds_acc:.4f}")
                print(f"    Best Individual: {best_individual_acc:.4f}")
                print(f"    Improvement: {improvement:+.4f} ({improvement/best_individual_acc*100:+.2f}%)")


if __name__ == "__main__":
    # Run evaluation
    results = evaluate_all_models()
    
    # Generate summary statistics
    generate_summary_statistics(results)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
