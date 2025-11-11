"""
Ablation Study: Impact of Removing Top Classifiers

This script performs an ablation study to analyze how removing top-performing
classifiers affects the performance of Ensemble (majority voting) vs Dawid-Skene
aggregation methods.

The study systematically removes the top classifiers and tests various configurations,
measuring accuracy, precision, recall, F1-score, classifier accuracies, and execution time.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time
from typing import Dict, List, Tuple
from collections import Counter
from dawid_skene import DawidSkene
import json


def load_ground_truth(filepath: str = 'classifiers/sampled_labels.csv') -> pd.DataFrame:
    """Load ground truth labels."""
    df = pd.read_csv(filepath)
    if 'filename' in df.columns:
        df = df.rename(columns={'filename': 'video_name', 'label': 'true_label'})
    return df[['video_name', 'true_label']]


def load_all_predictions(predictions_dir: str = 'predictions') -> Dict[str, pd.DataFrame]:
    """Load all individual model predictions."""
    model_files = {
        'GPT-4o-mini': 'gpt4o_predictions.csv',
        'GPT-5-mini': 'gpt-5-mini_predictions.csv',
        'Gemini': 'gemini_predictions.csv',
        'Replicate': 'replicate_predictions.csv',
        'MoonDream2': 'moondream_predictions.csv',
        'Qwen-VL': 'qwen_predictions.csv',
        'TwelveLabs': 'twelvelabs_predictions.csv',
    }
    
    predictions = {}
    for model_name, filename in model_files.items():
        filepath = os.path.join(predictions_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            predictions[model_name] = df[['video_name', 'predicted_class']].copy()
            predictions[model_name] = predictions[model_name].rename(
                columns={'predicted_class': model_name}
            )
        else:
            print(f"⚠️  Warning: {filepath} not found, skipping {model_name}")
    
    return predictions


def ensemble_predict(predictions: Dict[str, pd.DataFrame], 
                     classifier_subset: List[str],
                     ground_truth: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform majority voting ensemble on a subset of classifiers.
    
    Args:
        predictions: Dictionary of all predictions
        classifier_subset: List of classifier names to include
        ground_truth: Ground truth DataFrame
        
    Returns:
        Tuple of (y_true, y_pred, execution_time)
    """
    start_time = time.time()
    
    # Merge all predictions
    merged = ground_truth.copy()
    for clf_name in classifier_subset:
        if clf_name in predictions:
            merged = merged.merge(predictions[clf_name], on='video_name', how='inner')
    
    # Majority voting
    ensemble_preds = []
    for idx, row in merged.iterrows():
        votes = [row[clf] for clf in classifier_subset if clf in row.index]
        if votes:
            # Get most common prediction
            vote_counts = Counter(votes)
            # Check if there's a clear majority
            most_common = vote_counts.most_common(2)
            if len(most_common) == 1:
                # Only one unique vote - clear majority
                majority_vote = most_common[0][0]
            elif most_common[0][1] > most_common[1][1]:
                # First place has more votes than second place - clear majority
                majority_vote = most_common[0][0]
            else:
                # Tie - no clear majority
                majority_vote = "ERROR"
            ensemble_preds.append(majority_vote)
        else:
            ensemble_preds.append(None)
    
    merged['ensemble_pred'] = ensemble_preds
    
    # Remove rows with None predictions
    merged = merged.dropna(subset=['ensemble_pred'])
    
    execution_time = time.time() - start_time
    
    return merged['true_label'].values, merged['ensemble_pred'].values, execution_time


def dawid_skene_predict(predictions: Dict[str, pd.DataFrame],
                       classifier_subset: List[str],
                       ground_truth: pd.DataFrame,
                       all_classes: List[str]) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    """
    Perform Dawid-Skene aggregation on a subset of classifiers.
    
    Args:
        predictions: Dictionary of all predictions
        classifier_subset: List of classifier names to include
        ground_truth: Ground truth DataFrame
        all_classes: List of all possible classes
        
    Returns:
        Tuple of (y_true, y_pred, execution_time, classifier_accuracies)
    """
    start_time = time.time()
    
    # Merge all predictions
    merged = ground_truth.copy()
    for clf_name in classifier_subset:
        if clf_name in predictions:
            merged = merged.merge(predictions[clf_name], on='video_name', how='inner')
    
    # Prepare annotations in Dawid-Skene format
    annotations = {}
    for idx, row in merged.iterrows():
        video_name = row['video_name']
        annotations[video_name] = {}
        for clf_name in classifier_subset:
            if clf_name in row.index and pd.notna(row[clf_name]):
                annotations[video_name][clf_name] = row[clf_name]
    
    # Fit Dawid-Skene model
    model = DawidSkene(max_iterations=100, tolerance=1e-6)
    model.fit(annotations, all_classes, classifier_subset)
    
    # Get predictions
    ds_predictions = model.predict(return_probabilities=False)
    
    # Get classifier accuracies
    classifier_accuracies = model.get_annotator_accuracy()
    
    # Align with ground truth
    y_true = []
    y_pred = []
    for video_name in ds_predictions.keys():
        if video_name in merged['video_name'].values:
            true_label = merged[merged['video_name'] == video_name]['true_label'].values[0]
            y_true.append(true_label)
            y_pred.append(ds_predictions[video_name])
    
    execution_time = time.time() - start_time
    
    return np.array(y_true), np.array(y_pred), execution_time, classifier_accuracies


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     labels: List[str]) -> Dict[str, float]:
    """Calculate performance metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, labels=labels, 
                                    average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, labels=labels, 
                              average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, labels=labels, 
                      average='macro', zero_division=0)
    }


def run_ablation_study(output_dir: str = 'ablation_results'):
    """
    Run complete ablation study.
    
    Systematically removes top classifiers and compares Ensemble vs Dawid-Skene.
    """
    print("=" * 100)
    print("ABLATION STUDY: Impact of Removing Top Classifiers")
    print("=" * 100)
    
    # Load data
    print("\n📂 Loading data...")
    ground_truth = load_ground_truth()
    predictions = load_all_predictions()
    all_classes = sorted(ground_truth['true_label'].unique())
    
    print(f"   Ground truth: {len(ground_truth)} videos")
    print(f"   Classes: {len(all_classes)}")
    print(f"   Classifiers loaded: {len(predictions)}")
    
    # Define classifier rankings (from evaluation results)
    # Top 3 individual classifiers based on accuracy
    classifier_ranking = [
        'Gemini',       # 93.60%
        'GPT-5-mini',   # 93.30%
        'TwelveLabs',   # 86.10%
        'GPT-4o-mini',  # 84.70%
        'Qwen-VL',      # 71.80%
        'Replicate',    # 64.20%
        'MoonDream2',   # 5.80%
    ]
    
    # Filter to only available classifiers
    classifier_ranking = [clf for clf in classifier_ranking if clf in predictions]
    
    print(f"\n📊 Classifier ranking (by individual accuracy):")
    for i, clf in enumerate(classifier_ranking, 1):
        print(f"   {i}. {clf}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results
    results = []
    
    print("\n" + "=" * 100)
    print("ABLATION EXPERIMENTS")
    print("=" * 100)
    
    # Experiment 1: All classifiers (baseline)
    print("\n" + "-" * 100)
    print("EXPERIMENT 1: Baseline (All Classifiers)")
    print("-" * 100)
    
    active_classifiers = classifier_ranking.copy()
    print(f"Active classifiers ({len(active_classifiers)}): {', '.join(active_classifiers)}")
    
    # Ensemble
    print("\n  🔹 Ensemble (Majority Voting)...")
    y_true_ens, y_pred_ens, ens_time = ensemble_predict(predictions, active_classifiers, ground_truth)
    metrics_ens = calculate_metrics(y_true_ens, y_pred_ens, all_classes)
    print(f"     Accuracy: {metrics_ens['accuracy']:.4f} | F1: {metrics_ens['f1']:.4f} | "
          f"Precision: {metrics_ens['precision']:.4f} | Recall: {metrics_ens['recall']:.4f}")
    
    # Dawid-Skene
    print("  🔹 Dawid-Skene...")
    y_true_ds, y_pred_ds, ds_time, ds_accuracies = dawid_skene_predict(predictions, active_classifiers, 
                                                ground_truth, all_classes)
    metrics_ds = calculate_metrics(y_true_ds, y_pred_ds, all_classes)
    print(f"     Accuracy: {metrics_ds['accuracy']:.4f} | F1: {metrics_ds['f1']:.4f} | "
          f"Precision: {metrics_ds['precision']:.4f} | Recall: {metrics_ds['recall']:.4f}")
    
    results.append({
        'Experiment': 'Baseline (All)',
        'Removed_Classifier': 'None',
        'Active_Classifiers': ', '.join(active_classifiers),
        'Num_Classifiers': len(active_classifiers),
        'Ensemble_Accuracy': metrics_ens['accuracy'],
        'Ensemble_Precision': metrics_ens['precision'],
        'Ensemble_Recall': metrics_ens['recall'],
        'Ensemble_F1': metrics_ens['f1'],
        'Ensemble_Time': ens_time,
        'DS_Accuracy': metrics_ds['accuracy'],
        'DS_Precision': metrics_ds['precision'],
        'DS_Recall': metrics_ds['recall'],
        'DS_F1': metrics_ds['f1'],
        'DS_Time': ds_time,
        'Accuracy_Diff': metrics_ds['accuracy'] - metrics_ens['accuracy'],
        'F1_Diff': metrics_ds['f1'] - metrics_ens['f1'],
        'Time_Diff': ds_time - ens_time
    })
    
    # Experiments 2-6: Remove top 5 classifiers one by one
    for exp_num in range(1, 6):
        print("\n" + "-" * 100)
        print(f"EXPERIMENT {exp_num + 1}: Remove Top-{exp_num} Classifier(s)")
        print("-" * 100)
        
        # Remove the top classifier
        removed_classifier = classifier_ranking[exp_num - 1]
        active_classifiers = [clf for clf in classifier_ranking if clf != removed_classifier 
                            and clf in [c for i, c in enumerate(classifier_ranking) 
                                      if i >= exp_num - 1]]
        
        # Actually, we want to remove classifiers cumulatively
        active_classifiers = classifier_ranking[exp_num:]  # Skip top exp_num classifiers
        
        print(f"❌ Removed: {removed_classifier}")
        print(f"Active classifiers ({len(active_classifiers)}): {', '.join(active_classifiers)}")
        
        if len(active_classifiers) < 2:
            print("  ⚠️  Warning: Less than 2 classifiers remaining, skipping...")
            continue
        
        # Ensemble
        print("\n  🔹 Ensemble (Majority Voting)...")
        y_true_ens, y_pred_ens, ens_time = ensemble_predict(predictions, active_classifiers, ground_truth)
        metrics_ens = calculate_metrics(y_true_ens, y_pred_ens, all_classes)
        print(f"     Accuracy: {metrics_ens['accuracy']:.4f} | F1: {metrics_ens['f1']:.4f} | "
              f"Precision: {metrics_ens['precision']:.4f} | Recall: {metrics_ens['recall']:.4f}")
        
        # Dawid-Skene
        print("  🔹 Dawid-Skene...")
        y_true_ds, y_pred_ds, ds_time, ds_accuracies = dawid_skene_predict(predictions, active_classifiers, 
                                                    ground_truth, all_classes)
        metrics_ds = calculate_metrics(y_true_ds, y_pred_ds, all_classes)
        print(f"     Accuracy: {metrics_ds['accuracy']:.4f} | F1: {metrics_ds['f1']:.4f} | "
              f"Precision: {metrics_ds['precision']:.4f} | Recall: {metrics_ds['recall']:.4f}")
        
        # Performance drop from baseline
        baseline_ens_acc = results[0]['Ensemble_Accuracy']
        baseline_ds_acc = results[0]['DS_Accuracy']
        
        print(f"\n  📉 Performance drop from baseline:")
        print(f"     Ensemble: {(metrics_ens['accuracy'] - baseline_ens_acc) * 100:+.2f}%")
        print(f"     Dawid-Skene: {(metrics_ds['accuracy'] - baseline_ds_acc) * 100:+.2f}%")
        
        results.append({
            'Experiment': f'Remove Top-{exp_num}',
            'Removed_Classifier': removed_classifier,
            'Active_Classifiers': ', '.join(active_classifiers),
            'Num_Classifiers': len(active_classifiers),
            'Ensemble_Accuracy': metrics_ens['accuracy'],
            'Ensemble_Precision': metrics_ens['precision'],
            'Ensemble_Recall': metrics_ens['recall'],
            'Ensemble_F1': metrics_ens['f1'],
            'Ensemble_Time': ens_time,
            'DS_Accuracy': metrics_ds['accuracy'],
            'DS_Precision': metrics_ds['precision'],
            'DS_Recall': metrics_ds['recall'],
            'DS_F1': metrics_ds['f1'],
            'DS_Time': ds_time,
            'Accuracy_Diff': metrics_ds['accuracy'] - metrics_ens['accuracy'],
            'F1_Diff': metrics_ds['f1'] - metrics_ens['f1'],
            'Time_Diff': ds_time - ens_time
        })
    
    # Additional experiments: Remove each top-5 classifier individually
    print("\n" + "=" * 100)
    print("INDIVIDUAL REMOVAL EXPERIMENTS (Remove one top-5 classifier at a time)")
    print("=" * 100)
    
    for i in range(5):
        removed_classifier = classifier_ranking[i]
        
        print("\n" + "-" * 100)
        print(f"EXPERIMENT: Remove {removed_classifier} only")
        print("-" * 100)
        
        active_classifiers = [clf for clf in classifier_ranking if clf != removed_classifier]
        
        print(f"❌ Removed: {removed_classifier}")
        print(f"Active classifiers ({len(active_classifiers)}): {', '.join(active_classifiers)}")
        
        # Ensemble
        print("\n  🔹 Ensemble (Majority Voting)...")
        y_true_ens, y_pred_ens, ens_time = ensemble_predict(predictions, active_classifiers, ground_truth)
        metrics_ens = calculate_metrics(y_true_ens, y_pred_ens, all_classes)
        print(f"     Accuracy: {metrics_ens['accuracy']:.4f} | F1: {metrics_ens['f1']:.4f} | "
              f"Precision: {metrics_ens['precision']:.4f} | Recall: {metrics_ens['recall']:.4f}")
        
        # Dawid-Skene
        print("  🔹 Dawid-Skene...")
        y_true_ds, y_pred_ds, ds_time, ds_accuracies = dawid_skene_predict(predictions, active_classifiers, 
                                                    ground_truth, all_classes)
        metrics_ds = calculate_metrics(y_true_ds, y_pred_ds, all_classes)
        print(f"     Accuracy: {metrics_ds['accuracy']:.4f} | F1: {metrics_ds['f1']:.4f} | "
              f"Precision: {metrics_ds['precision']:.4f} | Recall: {metrics_ds['recall']:.4f}")
        
        # Performance drop from baseline
        baseline_ens_acc = results[0]['Ensemble_Accuracy']
        baseline_ds_acc = results[0]['DS_Accuracy']
        
        print(f"\n  📉 Performance drop from baseline:")
        print(f"     Ensemble: {(metrics_ens['accuracy'] - baseline_ens_acc) * 100:+.2f}%")
        print(f"     Dawid-Skene: {(metrics_ds['accuracy'] - baseline_ds_acc) * 100:+.2f}%")
        
        results.append({
            'Experiment': f'Remove {removed_classifier}',
            'Removed_Classifier': removed_classifier,
            'Active_Classifiers': ', '.join(active_classifiers),
            'Num_Classifiers': len(active_classifiers),
            'Ensemble_Accuracy': metrics_ens['accuracy'],
            'Ensemble_Precision': metrics_ens['precision'],
            'Ensemble_Recall': metrics_ens['recall'],
            'Ensemble_F1': metrics_ens['f1'],
            'Ensemble_Time': ens_time,
            'DS_Accuracy': metrics_ds['accuracy'],
            'DS_Precision': metrics_ds['precision'],
            'DS_Recall': metrics_ds['recall'],
            'DS_F1': metrics_ds['f1'],
            'DS_Time': ds_time,
            'Accuracy_Diff': metrics_ds['accuracy'] - metrics_ens['accuracy'],
            'F1_Diff': metrics_ds['f1'] - metrics_ens['f1'],
            'Time_Diff': ds_time - ens_time
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    
    # Display summary table
    print("\n📊 Ablation Study Results:")
    print("-" * 100)
    
    summary_cols = ['Experiment', 'Num_Classifiers', 'Ensemble_Accuracy', 'DS_Accuracy', 
                   'Accuracy_Diff', 'Ensemble_F1', 'DS_F1', 'F1_Diff']
    summary_df = results_df[summary_cols].copy()
    
    # Format for display
    summary_df['Ensemble_Accuracy'] = summary_df['Ensemble_Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    summary_df['DS_Accuracy'] = summary_df['DS_Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    summary_df['Ensemble_F1'] = summary_df['Ensemble_F1'].apply(lambda x: f"{x*100:.2f}%")
    summary_df['DS_F1'] = summary_df['DS_F1'].apply(lambda x: f"{x*100:.2f}%")
    summary_df['Accuracy_Diff'] = summary_df['Accuracy_Diff'].apply(lambda x: f"{x*100:+.2f}%")
    summary_df['F1_Diff'] = summary_df['F1_Diff'].apply(lambda x: f"{x*100:+.2f}%")
    
    print(summary_df.to_string(index=False))
    print("-" * 100)
    
    # Save detailed results
    output_file = os.path.join(output_dir, 'ablation_study_results.csv')
    results_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Analysis
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    
    # Which method is more robust?
    baseline_ens_acc = results_df.iloc[0]['Ensemble_Accuracy']
    baseline_ds_acc = results_df.iloc[0]['DS_Accuracy']
    
    # Calculate average performance drop for cumulative removal
    cumulative_experiments = results_df[results_df['Experiment'].str.contains('Remove Top-')]
    
    if len(cumulative_experiments) > 0:
        avg_ens_drop = (cumulative_experiments['Ensemble_Accuracy'] - baseline_ens_acc).mean() * 100
        avg_ds_drop = (cumulative_experiments['DS_Accuracy'] - baseline_ds_acc).mean() * 100
        
        print(f"\n1️⃣  Average performance drop (cumulative removal):")
        print(f"   Ensemble: {avg_ens_drop:.2f}%")
        print(f"   Dawid-Skene: {avg_ds_drop:.2f}%")
        
        if abs(avg_ds_drop) < abs(avg_ens_drop):
            print(f"   ✅ Dawid-Skene is more robust ({abs(avg_ens_drop - avg_ds_drop):.2f}% better)")
        else:
            print(f"   ✅ Ensemble is more robust ({abs(avg_ds_drop - avg_ens_drop):.2f}% better)")
    
    # Which method wins more often?
    ds_wins = (results_df['Accuracy_Diff'] > 0).sum()
    ens_wins = (results_df['Accuracy_Diff'] < 0).sum()
    ties = (results_df['Accuracy_Diff'] == 0).sum()
    
    print(f"\n2️⃣  Head-to-head comparison ({len(results_df)} experiments):")
    print(f"   Dawid-Skene wins: {ds_wins}")
    print(f"   Ensemble wins: {ens_wins}")
    print(f"   Ties: {ties}")
    
    # Most impactful classifier to remove
    individual_removals = results_df[results_df['Experiment'].str.contains('Remove') & 
                                    ~results_df['Experiment'].str.contains('Top-')]
    
    if len(individual_removals) > 0:
        print(f"\n3️⃣  Impact of removing individual top-3 classifiers:")
        for _, row in individual_removals.iterrows():
            removed = row['Removed_Classifier']
            ens_drop = (row['Ensemble_Accuracy'] - baseline_ens_acc) * 100
            ds_drop = (row['DS_Accuracy'] - baseline_ds_acc) * 100
            
            print(f"\n   {removed}:")
            print(f"     Ensemble drop: {ens_drop:+.2f}%")
            print(f"     Dawid-Skene drop: {ds_drop:+.2f}%")
            
            if abs(ds_drop) < abs(ens_drop):
                print(f"     → Dawid-Skene handles removal better")
            elif abs(ens_drop) < abs(ds_drop):
                print(f"     → Ensemble handles removal better")
            else:
                print(f"     → Equal impact")
    
    # Best configuration
    best_idx = results_df['DS_Accuracy'].idxmax()
    best_config = results_df.iloc[best_idx]
    
    print(f"\n4️⃣  Best configuration found:")
    print(f"   Experiment: {best_config['Experiment']}")
    print(f"   Classifiers: {best_config['Active_Classifiers']}")
    print(f"   Dawid-Skene Accuracy: {best_config['DS_Accuracy']*100:.2f}%")
    print(f"   Ensemble Accuracy: {best_config['Ensemble_Accuracy']*100:.2f}%")
    
    print("\n" + "=" * 100)
    print("ABLATION STUDY COMPLETE")
    print("=" * 100)
    print(f"\n📁 Results saved to: {output_dir}/")
    print(f"   - ablation_study_results.csv")
    print("\n")
    
    return results_df


if __name__ == "__main__":
    results = run_ablation_study()
