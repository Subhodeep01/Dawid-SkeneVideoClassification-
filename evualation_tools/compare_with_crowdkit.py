"""
Comprehensive Comparison: Our Dawid-Skene vs Crowd-Kit Implementation

This script compares our implementation with the production Crowd-Kit library
to validate correctness and analyze performance differences.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from typing import Dict, List

# Import both implementations
from dawid_skene import DawidSkene as OurDawidSkene
from crowdkit.aggregation import DawidSkene as CrowdKitDawidSkene


def load_data():
    """Load the video classification dataset"""
    print("📂 Loading dataset...")
    
    # Load ground truth
    ground_truth = pd.read_csv('classifiers/sampled_labels.csv')
    ground_truth = ground_truth.rename(columns={'filename': 'video_name', 'label': 'true_label'})
    ground_truth = ground_truth[['video_name', 'true_label']]
    
    # Load classifier predictions
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
        df = pd.read_csv(file_path)
        predictions[clf_name] = df[['video_name', 'predicted_class']].copy()
        predictions[clf_name] = predictions[clf_name].rename(
            columns={'predicted_class': clf_name}
        )
    
    all_classes = sorted(ground_truth['true_label'].unique())
    
    print(f"  ✓ Videos: {len(ground_truth)}")
    print(f"  ✓ Classes: {len(all_classes)}")
    print(f"  ✓ Classifiers: {len(predictions)}")
    
    return ground_truth, predictions, all_classes


def prepare_our_format(ground_truth, predictions, classifiers):
    """Prepare data in format for our implementation"""
    # Merge all predictions
    merged = ground_truth.copy()
    for clf_name in classifiers:
        if clf_name in predictions:
            merged = merged.merge(predictions[clf_name], on='video_name', how='inner')
    
    # Convert to annotations dict
    annotations = {}
    for idx, row in merged.iterrows():
        video_name = row['video_name']
        annotations[video_name] = {}
        for clf_name in classifiers:
            if clf_name in row.index and pd.notna(row[clf_name]):
                annotations[video_name][clf_name] = row[clf_name]
    
    return annotations, merged


def prepare_crowdkit_format(ground_truth, predictions, classifiers):
    """Prepare data in format for Crowd-Kit (task, worker, label)"""
    rows = []
    
    # Merge all predictions
    merged = ground_truth.copy()
    for clf_name in classifiers:
        if clf_name in predictions:
            merged = merged.merge(predictions[clf_name], on='video_name', how='inner')
    
    # Convert to long format
    for idx, row in merged.iterrows():
        video_name = row['video_name']
        for clf_name in classifiers:
            if clf_name in row.index and pd.notna(row[clf_name]):
                rows.append({
                    'task': video_name,
                    'worker': clf_name,
                    'label': row[clf_name]
                })
    
    df = pd.DataFrame(rows)
    return df, merged


def run_our_implementation(annotations, all_classes, classifiers):
    """Run our Dawid-Skene implementation"""
    print("\n" + "="*80)
    print("RUNNING OUR IMPLEMENTATION")
    print("="*80)
    
    start_time = time.time()
    
    model = OurDawidSkene(max_iterations=100, tolerance=1e-6)
    model.fit(annotations, all_classes, classifiers)
    
    predictions = model.predict(return_probabilities=False)
    probabilities = model.predict(return_probabilities=True)
    accuracies = model.get_annotator_accuracy()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n✓ Completed in {execution_time:.2f} seconds")
    print(f"\n📊 Classifier Accuracy Estimates:")
    for clf, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {clf:15s}: {acc:.4f} ({acc*100:.2f}%)")
    
    return predictions, probabilities, accuracies, execution_time


def run_crowdkit_implementation(df_crowdkit):
    """Run Crowd-Kit's Dawid-Skene implementation"""
    print("\n" + "="*80)
    print("RUNNING CROWD-KIT IMPLEMENTATION")
    print("="*80)
    
    start_time = time.time()
    
    # Crowd-Kit uses n_iter parameter
    model = CrowdKitDawidSkene(n_iter=100)
    predictions_series = model.fit_predict(df_crowdkit)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Convert Series to dict
    predictions = predictions_series.to_dict()
    
    # Get probabilities if available
    try:
        probabilities_df = model.probas_
        probabilities = probabilities_df.to_dict('index')
    except:
        probabilities = None
    
    # Get worker skills (similar to our accuracies)
    try:
        worker_skills = model.skills_
        accuracies = worker_skills.to_dict()
        print(f"\n📊 Worker Skills (Classifier Accuracies):")
        for clf, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {clf:15s}: {acc:.4f} ({acc*100:.2f}%)")
    except:
        accuracies = None
        print("\n⚠️  Worker skills not available in Crowd-Kit output")
    
    print(f"\n✓ Completed in {execution_time:.2f} seconds")
    
    return predictions, probabilities, accuracies, execution_time


def compare_predictions(our_preds, crowdkit_preds, ground_truth):
    """Compare predictions from both implementations"""
    print("\n" + "="*80)
    print("COMPARING PREDICTIONS")
    print("="*80)
    
    # Align predictions with ground truth
    y_true = []
    y_our = []
    y_crowdkit = []
    
    gt_dict = dict(zip(ground_truth['video_name'], ground_truth['true_label']))
    
    for video_name in our_preds.keys():
        if video_name in crowdkit_preds and video_name in gt_dict:
            y_true.append(gt_dict[video_name])
            y_our.append(our_preds[video_name])
            y_crowdkit.append(crowdkit_preds[video_name])
    
    # Calculate accuracies
    our_accuracy = accuracy_score(y_true, y_our)
    crowdkit_accuracy = accuracy_score(y_true, y_crowdkit)
    
    print(f"\n📊 Overall Accuracy:")
    print(f"  Our Implementation:  {our_accuracy:.4f} ({our_accuracy*100:.2f}%)")
    print(f"  Crowd-Kit:           {crowdkit_accuracy:.4f} ({crowdkit_accuracy*100:.2f}%)")
    print(f"  Difference:          {abs(our_accuracy - crowdkit_accuracy):.4f} ({abs(our_accuracy - crowdkit_accuracy)*100:.2f}%)")
    
    # Agreement between implementations
    agreement = sum([1 for i in range(len(y_our)) if y_our[i] == y_crowdkit[i]]) / len(y_our)
    print(f"\n🤝 Agreement between implementations: {agreement:.4f} ({agreement*100:.2f}%)")
    
    # Find disagreements
    disagreements = []
    for i in range(len(y_true)):
        if y_our[i] != y_crowdkit[i]:
            video_name = list(our_preds.keys())[i]
            disagreements.append({
                'video': video_name,
                'ground_truth': y_true[i],
                'our_prediction': y_our[i],
                'crowdkit_prediction': y_crowdkit[i],
                'our_correct': y_our[i] == y_true[i],
                'crowdkit_correct': y_crowdkit[i] == y_true[i]
            })
    
    print(f"\n📋 Disagreements: {len(disagreements)} out of {len(y_true)} ({len(disagreements)/len(y_true)*100:.2f}%)")
    
    if len(disagreements) > 0:
        print(f"\n   First 10 disagreements:")
        for i, disagreement in enumerate(disagreements[:10], 1):
            our_status = "✓" if disagreement['our_correct'] else "✗"
            ck_status = "✓" if disagreement['crowdkit_correct'] else "✗"
            print(f"   {i}. {disagreement['video']}")
            print(f"      Ground Truth: {disagreement['ground_truth']}")
            print(f"      Our:          {disagreement['our_prediction']} {our_status}")
            print(f"      Crowd-Kit:    {disagreement['crowdkit_prediction']} {ck_status}")
    
    return our_accuracy, crowdkit_accuracy, agreement, disagreements


def compare_probabilities(our_probs, crowdkit_probs, sample_size=5):
    """Compare probability distributions for sample videos"""
    print("\n" + "="*80)
    print("COMPARING PROBABILITY DISTRIBUTIONS")
    print("="*80)
    
    if crowdkit_probs is None:
        print("\n⚠️  Crowd-Kit probabilities not available for comparison")
        return
    
    # Sample videos for comparison
    sample_videos = list(our_probs.keys())[:sample_size]
    
    print(f"\nComparing probabilities for {sample_size} sample videos:\n")
    
    for video in sample_videos:
        if video in crowdkit_probs:
            print(f"📹 {video}:")
            
            our_prob = our_probs[video]
            ck_prob = crowdkit_probs[video]
            
            # Get top 3 classes
            our_top = sorted(our_prob.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"   Our Implementation:")
            for cls, prob in our_top:
                print(f"      {cls:20s}: {prob:.6f}")
            
            print(f"   Crowd-Kit:")
            for cls, _ in our_top:
                ck_p = ck_prob.get(cls, 0.0)
                print(f"      {cls:20s}: {ck_p:.6f}")
            
            # Calculate KL divergence or similar
            print()


def compare_worker_skills(our_accuracies, crowdkit_accuracies):
    """Compare worker/classifier accuracy estimates"""
    print("\n" + "="*80)
    print("COMPARING CLASSIFIER ACCURACY ESTIMATES")
    print("="*80)
    
    if crowdkit_accuracies is None:
        print("\n⚠️  Crowd-Kit worker skills not available for comparison")
        return
    
    print("\n| Classifier | Our Estimate | Crowd-Kit | Difference |")
    print("|------------|--------------|-----------|------------|")
    
    total_diff = 0
    count = 0
    
    for clf in sorted(our_accuracies.keys()):
        our_acc = our_accuracies[clf]
        ck_acc = crowdkit_accuracies.get(clf, None)
        
        if ck_acc is not None:
            diff = abs(our_acc - ck_acc)
            total_diff += diff
            count += 1
            print(f"| {clf:10s} | {our_acc:.4f} | {ck_acc:.4f} | {diff:.4f} |")
        else:
            print(f"| {clf:10s} | {our_acc:.4f} | N/A | N/A |")
    
    if count > 0:
        avg_diff = total_diff / count
        print(f"\nAverage difference in accuracy estimates: {avg_diff:.4f} ({avg_diff*100:.2f}%)")


def main():
    """Main comparison function"""
    print("="*80)
    print("DAWID-SKENE IMPLEMENTATION COMPARISON")
    print("Our Implementation vs Crowd-Kit Library")
    print("="*80)
    
    # Load data
    ground_truth, predictions, all_classes = load_data()
    
    classifiers = list(predictions.keys())
    
    # Prepare data for both implementations
    print("\n📝 Preparing data formats...")
    annotations, merged_our = prepare_our_format(ground_truth, predictions, classifiers)
    df_crowdkit, merged_ck = prepare_crowdkit_format(ground_truth, predictions, classifiers)
    
    print(f"  ✓ Our format: {len(annotations)} videos with annotations")
    print(f"  ✓ Crowd-Kit format: {len(df_crowdkit)} annotation records")
    
    # Run our implementation
    our_preds, our_probs, our_accs, our_time = run_our_implementation(
        annotations, all_classes, classifiers
    )
    
    # Run Crowd-Kit implementation
    ck_preds, ck_probs, ck_accs, ck_time = run_crowdkit_implementation(df_crowdkit)
    
    # Compare results
    our_acc, ck_acc, agreement, disagreements = compare_predictions(
        our_preds, ck_preds, ground_truth
    )
    
    compare_probabilities(our_probs, ck_probs, sample_size=5)
    
    compare_worker_skills(our_accs, ck_accs)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n⏱️  Execution Time:")
    print(f"  Our Implementation:  {our_time:.2f}s")
    print(f"  Crowd-Kit:           {ck_time:.2f}s")
    print(f"  Ratio:               {our_time/ck_time:.2f}x")
    
    print(f"\n🎯 Accuracy:")
    print(f"  Our Implementation:  {our_acc*100:.2f}%")
    print(f"  Crowd-Kit:           {ck_acc*100:.2f}%")
    print(f"  Difference:          {abs(our_acc - ck_acc)*100:.2f}%")
    
    print(f"\n🤝 Agreement:")
    print(f"  Predictions match:   {agreement*100:.2f}%")
    print(f"  Disagreements:       {len(disagreements)} / {len(our_preds)}")
    
    if abs(our_acc - ck_acc) < 0.001 and agreement > 0.99:
        print("\n✅ CONCLUSION: Implementations produce nearly identical results!")
        print("   Our implementation is correctly validated against Crowd-Kit.")
    elif abs(our_acc - ck_acc) < 0.01 and agreement > 0.95:
        print("\n✅ CONCLUSION: Implementations produce very similar results!")
        print("   Minor differences likely due to initialization or numerical precision.")
    else:
        print("\n⚠️  CONCLUSION: Some differences detected.")
        print("   May be due to different initialization, convergence criteria, or implementation details.")
    
    # Save detailed comparison
    if len(disagreements) > 0:
        disagreements_df = pd.DataFrame(disagreements)
        disagreements_df.to_csv('comparison_disagreements.csv', index=False)
        print(f"\n💾 Saved disagreements to: comparison_disagreements.csv")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
