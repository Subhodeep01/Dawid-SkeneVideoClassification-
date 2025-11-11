"""
Comparison Test: Standard Dawid-Skene vs Complement Dawid-Skene

This script compares both implementations on real video classification data
to determine which performs better empirically.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

from dawid_skene import DawidSkene
from dawid_skene_complement import DawidSkeneComplement


def load_ground_truth(filepath: str = 'classifiers/sampled_labels.csv') -> pd.DataFrame:
    """Load ground truth labels."""
    df = pd.read_csv(filepath)
    if 'filename' in df.columns:
        df = df.rename(columns={'filename': 'video_name', 'label': 'true_label'})
    return df[['video_name', 'true_label']]


def load_predictions(predictions_dir: str = 'predictions'):
    """Load all individual model predictions."""
    import os
    
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
    
    return predictions


def prepare_annotations(predictions, classifier_subset, ground_truth):
    """Prepare annotations in Dawid-Skene format."""
    # Merge all predictions
    merged = ground_truth.copy()
    for clf_name in classifier_subset:
        if clf_name in predictions:
            merged = merged.merge(predictions[clf_name], on='video_name', how='inner')
    
    # Convert to annotations dict
    annotations = {}
    for idx, row in merged.iterrows():
        video_name = row['video_name']
        annotations[video_name] = {}
        for clf_name in classifier_subset:
            if clf_name in row.index and pd.notna(row[clf_name]):
                annotations[video_name][clf_name] = row[clf_name]
    
    return annotations, merged


def calculate_metrics(y_true, y_pred, labels):
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


def run_comparison():
    """Run comparison between Standard and Complement Dawid-Skene."""
    
    print("=" * 100)
    print("DAWID-SKENE COMPARISON: Standard vs Complement M-step")
    print("=" * 100)
    
    # Load data
    print("\n📂 Loading data...")
    ground_truth = load_ground_truth()
    predictions = load_predictions()
    all_classes = sorted(ground_truth['true_label'].unique())
    
    print(f"   Ground truth: {len(ground_truth)} videos")
    print(f"   Classes: {len(all_classes)}")
    print(f"   Classifiers loaded: {len(predictions)}")
    
    # Test configurations
    test_configs = [
        {
            'name': 'All 7 Classifiers',
            'classifiers': list(predictions.keys())
        },
        {
            'name': 'Top 4 Classifiers',
            'classifiers': ['Gemini', 'GPT-5-mini', 'TwelveLabs', 'GPT-4o-mini']
        },
        {
            'name': 'Top 3 Classifiers',
            'classifiers': ['Gemini', 'GPT-5-mini', 'TwelveLabs']
        },
        {
            'name': 'Bottom 3 Classifiers (Low Quality)',
            'classifiers': ['Qwen-VL', 'Replicate', 'MoonDream2']
        }
    ]
    
    results = []
    
    for config in test_configs:
        print("\n" + "=" * 100)
        print(f"TEST: {config['name']}")
        print("=" * 100)
        print(f"Classifiers: {', '.join(config['classifiers'])}")
        
        # Prepare annotations
        annotations, merged = prepare_annotations(
            predictions, config['classifiers'], ground_truth
        )
        
        print(f"Videos to classify: {len(annotations)}")
        
        # Test 1: Standard Dawid-Skene
        print("\n" + "-" * 100)
        print("1️⃣  STANDARD DAWID-SKENE")
        print("-" * 100)
        
        start_time = time.time()
        model_standard = DawidSkene(max_iterations=100, tolerance=1e-6)
        model_standard.fit(annotations, all_classes, config['classifiers'])
        time_standard = time.time() - start_time
        
        # Get predictions
        preds_standard = model_standard.predict(return_probabilities=False)
        
        # Calculate metrics
        y_true = []
        y_pred_standard = []
        for video_name in preds_standard.keys():
            if video_name in merged['video_name'].values:
                true_label = merged[merged['video_name'] == video_name]['true_label'].values[0]
                y_true.append(true_label)
                y_pred_standard.append(preds_standard[video_name])
        
        y_true = np.array(y_true)
        y_pred_standard = np.array(y_pred_standard)
        
        metrics_standard = calculate_metrics(y_true, y_pred_standard, all_classes)
        accuracies_standard = model_standard.get_annotator_accuracy()
        
        print(f"⏱️  Training time: {time_standard:.2f}s")
        print(f"📊 Performance:")
        print(f"   Accuracy:  {metrics_standard['accuracy']*100:.2f}%")
        print(f"   Precision: {metrics_standard['precision']*100:.2f}%")
        print(f"   Recall:    {metrics_standard['recall']*100:.2f}%")
        print(f"   F1 Score:  {metrics_standard['f1']*100:.2f}%")
        
        print(f"\n📈 Classifier Accuracies:")
        for clf, acc in sorted(accuracies_standard.items(), key=lambda x: x[1], reverse=True):
            print(f"   {clf:20s}: {acc*100:6.2f}%")
        
        # Test 2: Complement Dawid-Skene
        print("\n" + "-" * 100)
        print("2️⃣  COMPLEMENT DAWID-SKENE")
        print("-" * 100)
        
        start_time = time.time()
        model_complement = DawidSkeneComplement(max_iterations=100, tolerance=1e-6)
        model_complement.fit(annotations, all_classes, config['classifiers'])
        time_complement = time.time() - start_time
        
        # Get predictions
        preds_complement = model_complement.predict(return_probabilities=False)
        
        # Calculate metrics
        y_pred_complement = []
        for video_name in preds_complement.keys():
            if video_name in merged['video_name'].values:
                y_pred_complement.append(preds_complement[video_name])
        
        y_pred_complement = np.array(y_pred_complement)
        
        metrics_complement = calculate_metrics(y_true, y_pred_complement, all_classes)
        accuracies_complement = model_complement.get_annotator_accuracy()
        
        print(f"⏱️  Training time: {time_complement:.2f}s")
        print(f"📊 Performance:")
        print(f"   Accuracy:  {metrics_complement['accuracy']*100:.2f}%")
        print(f"   Precision: {metrics_complement['precision']*100:.2f}%")
        print(f"   Recall:    {metrics_complement['recall']*100:.2f}%")
        print(f"   F1 Score:  {metrics_complement['f1']*100:.2f}%")
        
        print(f"\n📈 Classifier Accuracies:")
        for clf, acc in sorted(accuracies_complement.items(), key=lambda x: x[1], reverse=True):
            print(f"   {clf:20s}: {acc*100:6.2f}%")
        
        # Comparison
        print("\n" + "-" * 100)
        print("📊 COMPARISON")
        print("-" * 100)
        
        acc_diff = (metrics_complement['accuracy'] - metrics_standard['accuracy']) * 100
        f1_diff = (metrics_complement['f1'] - metrics_standard['f1']) * 100
        time_diff = time_complement - time_standard
        
        print(f"Accuracy:  Complement vs Standard = {acc_diff:+.2f}%")
        print(f"F1 Score:  Complement vs Standard = {f1_diff:+.2f}%")
        print(f"Time:      Complement vs Standard = {time_diff:+.2f}s")
        
        if acc_diff > 0:
            print(f"✅ Complement wins by {acc_diff:.2f}%!")
            winner = 'Complement'
        elif acc_diff < 0:
            print(f"✅ Standard wins by {-acc_diff:.2f}%!")
            winner = 'Standard'
        else:
            print("🤝 Tie!")
            winner = 'Tie'
        
        # Calculate agreement
        agreement = (y_pred_standard == y_pred_complement).mean()
        print(f"\n🤝 Prediction Agreement: {agreement*100:.2f}%")
        print(f"   ({int(agreement*len(y_pred_standard))} out of {len(y_pred_standard)} predictions match)")
        
        # Store results
        results.append({
            'Configuration': config['name'],
            'Num_Classifiers': len(config['classifiers']),
            'Standard_Accuracy': metrics_standard['accuracy'],
            'Standard_F1': metrics_standard['f1'],
            'Standard_Time': time_standard,
            'Complement_Accuracy': metrics_complement['accuracy'],
            'Complement_F1': metrics_complement['f1'],
            'Complement_Time': time_complement,
            'Accuracy_Diff': acc_diff,
            'F1_Diff': f1_diff,
            'Winner': winner,
            'Agreement': agreement
        })
    
    # Final Summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    
    results_df = pd.DataFrame(results)
    
    # Display summary table
    print("\n📊 Results Across All Configurations:")
    print("-" * 100)
    
    summary_df = results_df[[
        'Configuration', 'Num_Classifiers', 
        'Standard_Accuracy', 'Complement_Accuracy', 'Accuracy_Diff',
        'Standard_F1', 'Complement_F1', 'F1_Diff',
        'Winner', 'Agreement'
    ]].copy()
    
    # Format for display
    summary_df['Standard_Accuracy'] = summary_df['Standard_Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    summary_df['Complement_Accuracy'] = summary_df['Complement_Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    summary_df['Standard_F1'] = summary_df['Standard_F1'].apply(lambda x: f"{x*100:.2f}%")
    summary_df['Complement_F1'] = summary_df['Complement_F1'].apply(lambda x: f"{x*100:.2f}%")
    summary_df['Accuracy_Diff'] = summary_df['Accuracy_Diff'].apply(lambda x: f"{x:+.2f}%")
    summary_df['F1_Diff'] = summary_df['F1_Diff'].apply(lambda x: f"{x:+.2f}%")
    summary_df['Agreement'] = summary_df['Agreement'].apply(lambda x: f"{x*100:.1f}%")
    
    print(summary_df.to_string(index=False))
    print("-" * 100)
    
    # Overall winner
    print("\n🏆 OVERALL RESULTS:")
    standard_wins = (results_df['Winner'] == 'Standard').sum()
    complement_wins = (results_df['Winner'] == 'Complement').sum()
    ties = (results_df['Winner'] == 'Tie').sum()
    
    print(f"   Standard wins: {standard_wins}")
    print(f"   Complement wins: {complement_wins}")
    print(f"   Ties: {ties}")
    
    avg_acc_diff = results_df['Accuracy_Diff'].mean()
    avg_f1_diff = results_df['F1_Diff'].mean()
    
    print(f"\n📊 Average Performance Difference:")
    print(f"   Accuracy: {avg_acc_diff:+.3f}% (Complement vs Standard)")
    print(f"   F1 Score: {avg_f1_diff:+.3f}% (Complement vs Standard)")
    
    if avg_acc_diff > 0.1:
        print(f"\n✅ RECOMMENDATION: Use COMPLEMENT M-step (+{avg_acc_diff:.2f}% better)")
    elif avg_acc_diff < -0.1:
        print(f"\n✅ RECOMMENDATION: Use STANDARD M-step (+{-avg_acc_diff:.2f}% better)")
    else:
        print(f"\n🤝 RECOMMENDATION: Both are equivalent (difference < 0.1%)")
        print("   Use STANDARD for simplicity and established theory")
    
    # Save results
    results_df.to_csv('dawid_skene_comparison_results.csv', index=False, float_format='%.6f')
    print(f"\n💾 Detailed results saved to: dawid_skene_comparison_results.csv")
    
    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)
    
    return results_df


if __name__ == "__main__":
    results = run_comparison()
