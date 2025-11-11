"""
Comparison: Our Dawid-Skene vs Crowd-Kit 
Configuration: Gemini + Bottom-1 (MoonDream2) only
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import time

# Import both implementations
from dawid_skene import DawidSkene as OurDawidSkene
from crowdkit.aggregation import DawidSkene as CrowdKitDawidSkene


def main():
    print("="*80)
    print("DAWID-SKENE COMPARISON: Gemini + Bottom-1 (MoonDream2)")
    print("="*80)
    
    # Load ground truth
    print("\n📂 Loading data...")
    ground_truth = pd.read_csv('classifiers/sampled_labels.csv')
    ground_truth = ground_truth.rename(columns={'filename': 'video_name', 'label': 'true_label'})
    ground_truth = ground_truth[['video_name', 'true_label']]
    
    # Load only Gemini and MoonDream2
    classifiers = ['Gemini', 'MoonDream2']
    
    gemini = pd.read_csv('predictions/gemini_predictions.csv')
    moondream = pd.read_csv('predictions/moondream_predictions.csv')
    
    predictions = {
        'Gemini': gemini[['video_name', 'predicted_class']].rename(columns={'predicted_class': 'Gemini'}),
        'MoonDream2': moondream[['video_name', 'predicted_class']].rename(columns={'predicted_class': 'MoonDream2'})
    }
    
    all_classes = sorted(ground_truth['true_label'].unique())
    
    print(f"  ✓ Videos: {len(ground_truth)}")
    print(f"  ✓ Classes: {len(all_classes)}")
    print(f"  ✓ Classifiers: {classifiers}")
    
    # Merge predictions
    merged = ground_truth.copy()
    for clf_name in classifiers:
        merged = merged.merge(predictions[clf_name], on='video_name', how='inner')
    
    print(f"  ✓ Videos with both annotations: {len(merged)}")
    
    # Prepare data for our implementation
    print("\n" + "="*80)
    print("RUNNING OUR IMPLEMENTATION")
    print("="*80)
    
    annotations = {}
    for idx, row in merged.iterrows():
        video_name = row['video_name']
        annotations[video_name] = {}
        for clf_name in classifiers:
            if clf_name in row.index and pd.notna(row[clf_name]):
                annotations[video_name][clf_name] = row[clf_name]
    
    start_time = time.time()
    our_model = OurDawidSkene(max_iterations=100, tolerance=1e-6)
    our_model.fit(annotations, all_classes, classifiers)
    our_predictions = our_model.predict(return_probabilities=False)
    our_probs = our_model.predict(return_probabilities=True)
    our_accuracies = our_model.get_annotator_accuracy()
    our_time = time.time() - start_time
    
    print(f"\n✓ Completed in {our_time:.2f} seconds")
    print(f"\n📊 Classifier Accuracy Estimates:")
    for clf, acc in sorted(our_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {clf:15s}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Prepare data for Crowd-Kit
    print("\n" + "="*80)
    print("RUNNING CROWD-KIT IMPLEMENTATION")
    print("="*80)
    
    rows = []
    for idx, row in merged.iterrows():
        video_name = row['video_name']
        for clf_name in classifiers:
            if clf_name in row.index and pd.notna(row[clf_name]):
                rows.append({
                    'task': video_name,
                    'worker': clf_name,
                    'label': row[clf_name]
                })
    
    df_crowdkit = pd.DataFrame(rows)
    
    start_time = time.time()
    ck_model = CrowdKitDawidSkene(n_iter=100)
    ck_predictions_series = ck_model.fit_predict(df_crowdkit)
    ck_time = time.time() - start_time
    
    ck_predictions = ck_predictions_series.to_dict()
    
    # Try to get worker skills
    try:
        ck_accuracies = ck_model.skills_.to_dict()
        print(f"\n📊 Worker Skills:")
        for clf, acc in sorted(ck_accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {clf:15s}: {acc:.4f} ({acc*100:.2f}%)")
    except:
        ck_accuracies = None
        print("\n⚠️  Worker skills not available")
    
    print(f"\n✓ Completed in {ck_time:.2f} seconds")
    
    # Compare predictions
    print("\n" + "="*80)
    print("COMPARING PREDICTIONS")
    print("="*80)
    
    y_true = []
    y_our = []
    y_ck = []
    
    gt_dict = dict(zip(ground_truth['video_name'], ground_truth['true_label']))
    
    for video_name in our_predictions.keys():
        if video_name in ck_predictions and video_name in gt_dict:
            y_true.append(gt_dict[video_name])
            y_our.append(our_predictions[video_name])
            y_ck.append(ck_predictions[video_name])
    
    our_accuracy = accuracy_score(y_true, y_our)
    ck_accuracy = accuracy_score(y_true, y_ck)
    
    print(f"\n📊 Overall Accuracy:")
    print(f"  Our Implementation:  {our_accuracy:.4f} ({our_accuracy*100:.2f}%)")
    print(f"  Crowd-Kit:           {ck_accuracy:.4f} ({ck_accuracy*100:.2f}%)")
    print(f"  Difference:          {(our_accuracy - ck_accuracy):.4f} ({(our_accuracy - ck_accuracy)*100:+.2f}%)")
    
    # Agreement
    agreement = sum([1 for i in range(len(y_our)) if y_our[i] == y_ck[i]]) / len(y_our)
    print(f"\n🤝 Agreement: {agreement:.4f} ({agreement*100:.2f}%)")
    
    # Find disagreements
    disagreements = []
    for i in range(len(y_true)):
        if y_our[i] != y_ck[i]:
            video_name = list(our_predictions.keys())[i]
            disagreements.append({
                'video': video_name,
                'ground_truth': y_true[i],
                'our_prediction': y_our[i],
                'crowdkit_prediction': y_ck[i],
                'our_correct': y_our[i] == y_true[i],
                'crowdkit_correct': y_ck[i] == y_true[i]
            })
    
    print(f"\n📋 Disagreements: {len(disagreements)} out of {len(y_true)} ({len(disagreements)/len(y_true)*100:.2f}%)")
    
    if len(disagreements) > 0:
        our_wins = sum([1 for d in disagreements if d['our_correct'] and not d['crowdkit_correct']])
        ck_wins = sum([1 for d in disagreements if d['crowdkit_correct'] and not d['our_correct']])
        both_wrong = sum([1 for d in disagreements if not d['our_correct'] and not d['crowdkit_correct']])
        
        print(f"\n  When they disagree:")
        print(f"    Our implementation correct:  {our_wins} ({our_wins/len(disagreements)*100:.1f}%)")
        print(f"    Crowd-Kit correct:           {ck_wins} ({ck_wins/len(disagreements)*100:.1f}%)")
        print(f"    Both wrong:                  {both_wrong} ({both_wrong/len(disagreements)*100:.1f}%)")
        
        print(f"\n  First 10 disagreements:")
        for i, d in enumerate(disagreements[:10], 1):
            our_status = "✓" if d['our_correct'] else "✗"
            ck_status = "✓" if d['crowdkit_correct'] else "✗"
            print(f"    {i}. {d['video']}")
            print(f"       Ground Truth: {d['ground_truth']}")
            print(f"       Our:          {d['our_prediction']} {our_status}")
            print(f"       Crowd-Kit:    {d['crowdkit_prediction']} {ck_status}")
    
    # Check for invalid predictions
    print("\n" + "="*80)
    print("VALIDITY CHECK")
    print("="*80)
    
    valid_classes = set(all_classes)
    
    invalid_our = [p for p in our_predictions.values() if p not in valid_classes]
    invalid_ck = [p for p in ck_predictions.values() if p not in valid_classes]
    
    print(f"\n✓ Valid classes in dataset: {len(valid_classes)}")
    print(f"\nOur Implementation:")
    if len(invalid_our) > 0:
        print(f"  ⚠️  Invalid predictions: {len(invalid_our)}")
        print(f"  Invalid labels: {set(invalid_our)}")
    else:
        print(f"  ✅ All predictions valid (100%)")
    
    print(f"\nCrowd-Kit:")
    if len(invalid_ck) > 0:
        print(f"  ⚠️  Invalid predictions: {len(invalid_ck)}")
        print(f"  Invalid labels: {set(invalid_ck)}")
    else:
        print(f"  ✅ All predictions valid (100%)")
    
    # Compare probability distributions for sample
    print("\n" + "="*80)
    print("SAMPLE PROBABILITY DISTRIBUTIONS")
    print("="*80)
    
    try:
        ck_probs_df = ck_model.probas_
        ck_probs = ck_probs_df.to_dict('index')
        
        sample_videos = list(our_predictions.keys())[:5]
        
        for video in sample_videos:
            if video in ck_probs:
                print(f"\n📹 {video}:")
                print(f"   Ground Truth: {gt_dict.get(video, 'N/A')}")
                
                our_prob = our_probs[video]
                ck_prob = ck_probs[video]
                
                # Get top 3 predictions from our model
                our_top = sorted(our_prob.items(), key=lambda x: x[1], reverse=True)[:3]
                
                print(f"\n   Our Implementation (Top 3):")
                for cls, prob in our_top:
                    print(f"      {cls:30s}: {prob:.6f}")
                
                print(f"\n   Crowd-Kit (same classes):")
                for cls, _ in our_top:
                    ck_p = ck_prob.get(cls, 0.0)
                    print(f"      {cls:30s}: {ck_p:.6f}")
    except Exception as e:
        print(f"\n⚠️  Could not compare probabilities: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n⏱️  Execution Time:")
    print(f"  Our Implementation:  {our_time:.2f}s")
    print(f"  Crowd-Kit:           {ck_time:.2f}s")
    if ck_time > 0:
        print(f"  Ratio:               {our_time/ck_time:.2f}x")
    
    print(f"\n🎯 Accuracy:")
    print(f"  Our Implementation:  {our_accuracy*100:.2f}%")
    print(f"  Crowd-Kit:           {ck_accuracy*100:.2f}%")
    print(f"  Difference:          {(our_accuracy - ck_accuracy)*100:+.2f}%")
    
    print(f"\n🤝 Agreement: {agreement*100:.2f}%")
    
    if abs(our_accuracy - ck_accuracy) < 0.01 and agreement > 0.95:
        print("\n✅ CONCLUSION: Implementations produce very similar results!")
    elif our_accuracy > ck_accuracy:
        print(f"\n✅ CONCLUSION: Our implementation is MORE ACCURATE by {(our_accuracy - ck_accuracy)*100:.2f}%")
    else:
        print(f"\n⚠️  CONCLUSION: Crowd-Kit is more accurate by {(ck_accuracy - our_accuracy)*100:.2f}%")
    
    # Save disagreements
    if len(disagreements) > 0:
        df_dis = pd.DataFrame(disagreements)
        df_dis.to_csv('gemini_bottom1_disagreements.csv', index=False)
        print(f"\n💾 Saved disagreements to: gemini_bottom1_disagreements.csv")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
