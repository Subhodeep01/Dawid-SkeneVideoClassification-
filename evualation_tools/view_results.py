"""
Quick Results Viewer - Display evaluation results in a formatted table
"""

import pandas as pd
import os


def display_results():
    """Display formatted evaluation results."""
    
    results_file = 'evaluation_results/model_comparison.csv'
    
    if not os.path.exists(results_file):
        print("❌ Results file not found. Please run evaluation.py first.")
        return
    
    df = pd.read_csv(results_file)
    
    print("\n" + "=" * 120)
    print("VIDEO CLASSIFICATION MODEL EVALUATION - SUMMARY")
    print("=" * 120)
    
    # Create formatted display
    print("\n📊 OVERALL PERFORMANCE RANKINGS")
    print("-" * 120)
    
    # Add rank column
    df['Rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    display_cols = ['Rank', 'Model', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    display_df = df[display_cols].copy()
    
    # Format percentages
    for col in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")
    
    # Rename columns for display
    display_df.columns = ['Rank', 'Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    print(display_df.to_string(index=False))
    print("-" * 120)
    
    # Highlight top performers
    print("\n🏆 TOP PERFORMERS")
    print("-" * 120)
    
    top_3 = df.head(3)
    for idx, row in top_3.iterrows():
        print(f"{int(row['Rank'])}. {row['Model']}")
        print(f"   Accuracy: {row['accuracy']*100:.2f}% | "
              f"F1: {row['f1_macro']*100:.2f}% | "
              f"Precision: {row['precision_macro']*100:.2f}% | "
              f"Recall: {row['recall_macro']*100:.2f}%")
        print()
    
    # Performance tiers
    print("\n📈 PERFORMANCE TIERS")
    print("-" * 120)
    
    excellent = df[df['accuracy'] >= 0.90]
    good = df[(df['accuracy'] >= 0.80) & (df['accuracy'] < 0.90)]
    moderate = df[(df['accuracy'] >= 0.60) & (df['accuracy'] < 0.80)]
    poor = df[df['accuracy'] < 0.60]
    
    if len(excellent) > 0:
        print(f"🥇 EXCELLENT (≥90%): {', '.join(excellent['Model'].tolist())}")
    if len(good) > 0:
        print(f"🥈 GOOD (80-90%): {', '.join(good['Model'].tolist())}")
    if len(moderate) > 0:
        print(f"🥉 MODERATE (60-80%): {', '.join(moderate['Model'].tolist())}")
    if len(poor) > 0:
        print(f"❌ POOR (<60%): {', '.join(poor['Model'].tolist())}")
    
    # Key insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 120)
    
    best_acc = df.iloc[0]
    best_individual = df[~df['Model'].isin(['Ensemble', 'Dawid-Skene'])].iloc[0]
    
    print(f"✓ Best Overall: {best_acc['Model']} ({best_acc['accuracy']*100:.2f}%)")
    print(f"✓ Best Individual Model: {best_individual['Model']} ({best_individual['accuracy']*100:.2f}%)")
    
    # Check if aggregation helps
    ensemble = df[df['Model'] == 'Ensemble']
    dawid_skene = df[df['Model'] == 'Dawid-Skene']
    
    if len(ensemble) > 0:
        improvement_ens = (ensemble['accuracy'].values[0] - best_individual['accuracy']) * 100
        print(f"✓ Ensemble Improvement: +{improvement_ens:.2f}% over best individual")
    
    if len(dawid_skene) > 0:
        improvement_ds = (dawid_skene['accuracy'].values[0] - best_individual['accuracy']) * 100
        print(f"✓ Dawid-Skene Improvement: +{improvement_ds:.2f}% over best individual")
    
    # Worst performer
    worst = df.iloc[-1]
    print(f"✓ Lowest Performance: {worst['Model']} ({worst['accuracy']*100:.2f}%) - Consider excluding")
    
    print("\n" + "=" * 120)
    print("\n📁 Detailed results available in: evaluation_results/")
    print("   - model_comparison.csv")
    print("   - detailed_comparison.csv")
    print("   - *_confusion_matrix.csv (per model)")
    print("   - *_classification_report.csv (per model)")
    print("   - best_models_by_metric.csv")
    print("\n📖 Full analysis: EVALUATION_SUMMARY.md")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    display_results()
