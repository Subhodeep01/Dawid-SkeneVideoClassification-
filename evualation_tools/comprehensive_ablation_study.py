"""
Comprehensive Ablation Study: Enhanced Analysis with Timing and Classifier Accuracies

This script performs an enhanced ablation study with:
1. Remove top-1, top-2, top-3, top-4 classifiers
2. Gemini with bottom-1, bottom-2, bottom-3 classifiers
3. Performance timing for both Ensemble and Dawid-Skene
4. Classifier accuracy estimates from Dawid-Skene for each configuration
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import time
from typing import Dict, List, Tuple
from collections import Counter
from dawid_skene import DawidSkene


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
        votes = [row[clf] for clf in classifier_subset if clf in row.index and pd.notna(row[clf])]
        if votes:
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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, method_name: str) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    return {
        'Method': method_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (Macro)': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Precision (Weighted)': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall (Macro)': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall (Weighted)': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 (Macro)': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'F1 (Weighted)': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def run_comprehensive_ablation_study():
    """
    Run comprehensive ablation study with all requested configurations.
    """
    print("=" * 100)
    print("COMPREHENSIVE ABLATION STUDY")
    print("=" * 100)
    
    # Load ground truth and predictions
    ground_truth = pd.read_csv('classifiers/sampled_labels.csv')
    # Rename columns to match expected format
    ground_truth = ground_truth.rename(columns={'filename': 'video_name', 'label': 'true_label'})
    ground_truth = ground_truth[['video_name', 'true_label']]
    
    # Load all classifier predictions
    classifier_files = {
        'Gemini': 'predictions/gemini_predictions.csv',
        'GPT-5-mini': 'predictions/gpt-5-mini_predictions.csv',
        'TwelveLabs': 'predictions/twelvelabs_predictions.csv',
        'GPT-4o-mini': 'predictions/gpt4o_predictions.csv',
        'Qwen-VL': 'predictions/qwen_predictions.csv',
        'Replicate': 'predictions/replicate_predictions.csv',
        'MoonDream2': 'predictions/moondream_predictions.csv'
    }
    
    predictions = {}
    for clf_name, file_path in classifier_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            predictions[clf_name] = df[['video_name', 'predicted_class']].copy()
            predictions[clf_name] = predictions[clf_name].rename(
                columns={'predicted_class': clf_name}
            )
            print(f"✓ Loaded {clf_name}: {len(df)} predictions")
        else:
            print(f"✗ {file_path} not found, skipping {clf_name}")
    
    # Get all unique classes
    all_classes = sorted(ground_truth['true_label'].unique())
    print(f"\n📊 Dataset: {len(ground_truth)} videos, {len(all_classes)} classes")
    
    # Define all test configurations
    # Rankings: Gemini(93.6%), GPT-5-mini(93.3%), TwelveLabs(86.1%), GPT-4o-mini(84.7%), 
    #           Qwen-VL(71.8%), Replicate(64.2%), MoonDream2(5.8%)
    
    all_classifiers = ['Gemini', 'GPT-5-mini', 'TwelveLabs', 'GPT-4o-mini', 
                      'Qwen-VL', 'Replicate', 'MoonDream2']
    
    test_configs = [
        # {
        #     'name': 'Remove_Top1_Gemini',
        #     'description': 'Remove top-1 classifier (Gemini)',
        #     'classifiers': ['GPT-5-mini', 'TwelveLabs', 'GPT-4o-mini', 'Qwen-VL', 'Replicate', 'MoonDream2']
        # },
        # {
        #     'name': 'Remove_Top2_Gemini_GPT5mini',
        #     'description': 'Remove top-2 classifiers (Gemini, GPT-5-mini)',
        #     'classifiers': ['TwelveLabs', 'GPT-4o-mini', 'Qwen-VL', 'Replicate', 'MoonDream2']
        # },
        # {
        #     'name': 'Remove_Top3',
        #     'description': 'Remove top-3 classifiers (Gemini, GPT-5-mini, TwelveLabs)',
        #     'classifiers': ['GPT-4o-mini', 'Qwen-VL', 'Replicate', 'MoonDream2']
        # },
        # {
        #     'name': 'Remove_Top4',
        #     'description': 'Remove top-4 classifiers',
        #     'classifiers': ['Qwen-VL', 'Replicate', 'MoonDream2']
        # },
        {
            'name': 'Gemini_Plus_Bottom1',
            'description': 'Only Gemini with bottom-1 (MoonDream2)',
            'classifiers': ['Gemini', 'MoonDream2']
        },
        {
            'name': 'Gemini_Plus_Bottom2',
            'description': 'Only Gemini with bottom-2 (Replicate, MoonDream2)',
            'classifiers': ['Gemini', 'Replicate', 'MoonDream2']
        },
        {
            'name': 'Gemini_Plus_Bottom3',
            'description': 'Only Gemini with bottom-3 (Qwen-VL, Replicate, MoonDream2)',
            'classifiers': ['Gemini', 'Qwen-VL', 'Replicate', 'MoonDream2']
        }
    ]
    
    # Results storage
    results = []
    performance_data = []
    
    # Create directories for results
    os.makedirs('ablation_results', exist_ok=True)
    os.makedirs('ablation_results/classifier_accuracies', exist_ok=True)
    
    print("\n" + "=" * 100)
    print("RUNNING EXPERIMENTS")
    print("=" * 100)
    
    for i, config in enumerate(test_configs, 1):
        config_name = config['name']
        description = config['description']
        classifier_subset = config['classifiers']
        
        print(f"\n{'='*100}")
        print(f"EXPERIMENT {i}/{len(test_configs)}: {config_name}")
        print(f"{'='*100}")
        print(f"Description: {description}")
        print(f"Classifiers ({len(classifier_subset)}): {', '.join(classifier_subset)}")
        print(f"{'-'*100}")
        
        # Ensemble prediction
        print("\n🔹 Running Ensemble aggregation...")
        y_true_ens, y_pred_ens, ensemble_time = ensemble_predict(
            predictions, classifier_subset, ground_truth
        )
        print(f"   ✓ Completed in {ensemble_time:.2f}s")
        
        # Dawid-Skene prediction
        print("🔹 Running Dawid-Skene aggregation...")
        y_true_ds, y_pred_ds, ds_time, classifier_accuracies = dawid_skene_predict(
            predictions, classifier_subset, ground_truth, all_classes
        )
        print(f"   ✓ Completed in {ds_time:.2f}s")
        
        # Calculate metrics for Ensemble
        ens_metrics = calculate_metrics(y_true_ens, y_pred_ens, "Ensemble")
        
        # Calculate metrics for Dawid-Skene
        ds_metrics = calculate_metrics(y_true_ds, y_pred_ds, "Dawid-Skene")
        
        # Store results
        result = {
            'Configuration': config_name,
            'Description': description,
            'Num_Classifiers': len(classifier_subset),
            'Classifiers': ', '.join(classifier_subset),
            
            # Ensemble metrics
            'Ensemble_Accuracy': ens_metrics['Accuracy'],
            'Ensemble_Precision_Macro': ens_metrics['Precision (Macro)'],
            'Ensemble_Precision_Weighted': ens_metrics['Precision (Weighted)'],
            'Ensemble_Recall_Macro': ens_metrics['Recall (Macro)'],
            'Ensemble_Recall_Weighted': ens_metrics['Recall (Weighted)'],
            'Ensemble_F1_Macro': ens_metrics['F1 (Macro)'],
            'Ensemble_F1_Weighted': ens_metrics['F1 (Weighted)'],
            'Ensemble_Time_Seconds': ensemble_time,
            
            # Dawid-Skene metrics
            'DS_Accuracy': ds_metrics['Accuracy'],
            'DS_Precision_Macro': ds_metrics['Precision (Macro)'],
            'DS_Precision_Weighted': ds_metrics['Precision (Weighted)'],
            'DS_Recall_Macro': ds_metrics['Recall (Macro)'],
            'DS_Recall_Weighted': ds_metrics['Recall (Weighted)'],
            'DS_F1_Macro': ds_metrics['F1 (Macro)'],
            'DS_F1_Weighted': ds_metrics['F1 (Weighted)'],
            'DS_Time_Seconds': ds_time,
            
            # Differences
            'Accuracy_Difference': ds_metrics['Accuracy'] - ens_metrics['Accuracy'],
            'F1_Macro_Difference': ds_metrics['F1 (Macro)'] - ens_metrics['F1 (Macro)']
        }
        
        results.append(result)
        
        # Store performance data
        performance_data.append({
            'Configuration': config_name,
            'Ensemble_Time_Seconds': ensemble_time,
            'DawidSkene_Time_Seconds': ds_time,
            'Time_Difference_Seconds': ds_time - ensemble_time
        })
        
        # Save classifier accuracies for Dawid-Skene
        accuracies_df = pd.DataFrame([
            {'Classifier': clf, 'Estimated_Accuracy': acc}
            for clf, acc in classifier_accuracies.items()
        ])
        accuracies_df = accuracies_df.sort_values('Estimated_Accuracy', ascending=False)
        accuracies_file = f'ablation_results/classifier_accuracies/{config_name}_dawid_skene_accuracies.csv'
        accuracies_df.to_csv(accuracies_file, index=False, float_format='%.6f')
        
        # Print summary
        print(f"\n📊 RESULTS:")
        print(f"\n   Ensemble:")
        print(f"      Accuracy:  {ens_metrics['Accuracy']:.4f} ({ens_metrics['Accuracy']*100:.2f}%)")
        print(f"      F1 Macro:  {ens_metrics['F1 (Macro)']:.4f} ({ens_metrics['F1 (Macro)']*100:.2f}%)")
        print(f"      Time:      {ensemble_time:.2f}s")
        
        print(f"\n   Dawid-Skene:")
        print(f"      Accuracy:  {ds_metrics['Accuracy']:.4f} ({ds_metrics['Accuracy']*100:.2f}%)")
        print(f"      F1 Macro:  {ds_metrics['F1 (Macro)']:.4f} ({ds_metrics['F1 (Macro)']*100:.2f}%)")
        print(f"      Time:      {ds_time:.2f}s")
        
        print(f"\n   Performance Comparison:")
        acc_diff = ds_metrics['Accuracy'] - ens_metrics['Accuracy']
        f1_diff = ds_metrics['F1 (Macro)'] - ens_metrics['F1 (Macro)']
        time_diff = ds_time - ensemble_time
        
        print(f"      Accuracy Δ:  {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
        print(f"      F1 Macro Δ:  {f1_diff:+.4f} ({f1_diff*100:+.2f}%)")
        print(f"      Time Δ:      {time_diff:+.2f}s")
        
        if acc_diff > 0:
            print(f"      → Dawid-Skene wins by {acc_diff*100:.2f}%")
        elif acc_diff < 0:
            print(f"      → Ensemble wins by {abs(acc_diff)*100:.2f}%")
        else:
            print(f"      → Tie")
        
        print(f"\n   Classifier Accuracies (Dawid-Skene estimates):")
        for clf, acc in sorted(classifier_accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"      {clf:15s}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Save comprehensive results
    results_df = pd.DataFrame(results)
    results_df.to_csv('ablation_results/comprehensive_ablation_results.csv', index=False, float_format='%.6f')
    
    # Save performance data
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('ablation_results/performance.csv', index=False, float_format='%.2f')
    
    # Print final summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print(f"\n📁 Files saved:")
    print(f"   ✓ ablation_results/comprehensive_ablation_results.csv")
    print(f"   ✓ ablation_results/performance.csv")
    print(f"   ✓ ablation_results/classifier_accuracies/*.csv ({len(test_configs)} files)")
    
    print(f"\n📊 Overall Statistics:")
    
    # Best configurations
    best_ens_idx = results_df['Ensemble_Accuracy'].idxmax()
    best_ds_idx = results_df['DS_Accuracy'].idxmax()
    
    print(f"\n   Best Ensemble Configuration:")
    print(f"      {results_df.iloc[best_ens_idx]['Configuration']}")
    print(f"      Accuracy: {results_df.iloc[best_ens_idx]['Ensemble_Accuracy']:.4f} ({results_df.iloc[best_ens_idx]['Ensemble_Accuracy']*100:.2f}%)")
    
    print(f"\n   Best Dawid-Skene Configuration:")
    print(f"      {results_df.iloc[best_ds_idx]['Configuration']}")
    print(f"      Accuracy: {results_df.iloc[best_ds_idx]['DS_Accuracy']:.4f} ({results_df.iloc[best_ds_idx]['DS_Accuracy']*100:.2f}%)")
    
    # Win counts
    ds_wins = (results_df['Accuracy_Difference'] > 0).sum()
    ens_wins = (results_df['Accuracy_Difference'] < 0).sum()
    ties = (results_df['Accuracy_Difference'] == 0).sum()
    
    print(f"\n   Head-to-Head (Accuracy):")
    print(f"      Dawid-Skene wins: {ds_wins}/{len(test_configs)}")
    print(f"      Ensemble wins:    {ens_wins}/{len(test_configs)}")
    print(f"      Ties:             {ties}/{len(test_configs)}")
    
    # Average timing
    avg_ens_time = results_df['Ensemble_Time_Seconds'].mean()
    avg_ds_time = results_df['DS_Time_Seconds'].mean()
    
    print(f"\n   Average Execution Time:")
    print(f"      Ensemble:     {avg_ens_time:.2f}s")
    print(f"      Dawid-Skene:  {avg_ds_time:.2f}s")
    print(f"      Difference:   {avg_ds_time - avg_ens_time:+.2f}s")
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE ABLATION STUDY COMPLETE")
    print("=" * 100)
    
    return results_df, performance_df


if __name__ == "__main__":
    results_df, performance_df = run_comprehensive_ablation_study()
    
    print("\n✨ Study completed successfully!")
    print("\nTo view results:")
    print("   - Metrics: ablation_results/comprehensive_ablation_results.csv")
    print("   - Timing:  ablation_results/performance.csv")
    print("   - Accuracies: ablation_results/classifier_accuracies/")
