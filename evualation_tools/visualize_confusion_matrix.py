"""
Confusion Matrix Visualization Script

Creates visualizations from the confusion matrix CSV:
1. Full confusion matrix heatmap (60x60)
2. Per-class accuracy (diagonal values) bar chart
3. Most confused pairs analysis
4. Normalized confusion matrix
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def load_confusion_matrix(filepath):
    """Load confusion matrix from CSV."""
    cm_df = pd.read_csv(filepath, index_col=0)
    return cm_df


def plot_confusion_matrix_heatmap(cm_df, output_dir, method_name='Dawid-Skene'):
    """
    Create full confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(24, 22))
    
    # Create heatmap
    sns.heatmap(
        cm_df.values,
        annot=False,  # Too many cells to annotate
        fmt='d',
        cmap='YlOrRd',
        xticklabels=cm_df.columns,
        yticklabels=cm_df.index,
        cbar_kws={'label': 'Count'},
        linewidths=0.1,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title(f'{method_name}: Confusion Matrix\n(True Labels vs Predicted Labels)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_normalized_confusion_matrix(cm_df, output_dir, method_name='Dawid-Skene'):
    """
    Create normalized confusion matrix (row-wise).
    Shows percentage of predictions for each true class.
    """
    # Normalize by row (true label)
    cm_normalized = cm_df.div(cm_df.sum(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(24, 22))
    
    # Create heatmap
    sns.heatmap(
        cm_normalized.values,
        annot=False,
        fmt='.2f',
        cmap='RdYlGn_r',
        xticklabels=cm_df.columns,
        yticklabels=cm_df.index,
        cbar_kws={'label': 'Proportion'},
        vmin=0,
        vmax=1,
        linewidths=0.1,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title(f'{method_name}: Normalized Confusion Matrix\n(Row-wise: True Label Distribution)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Rotate labels
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_confusion_matrix_normalized.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_per_class_accuracy_from_cm(cm_df, output_dir, method_name='Dawid-Skene'):
    """
    Plot per-class accuracy from confusion matrix diagonal.
    """
    # Get diagonal (correct predictions)
    classes = cm_df.index.tolist()
    diagonal = np.diag(cm_df.values)
    totals = cm_df.sum(axis=1).values
    
    # Calculate accuracy for each class
    accuracies = diagonal / totals
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]
    sorted_counts = diagonal[sorted_indices]
    sorted_totals = totals[sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(20, 10))
    
    x_pos = np.arange(len(sorted_classes))
    
    # Color code
    colors = []
    for acc in sorted_accuracies:
        if acc >= 0.9:
            colors.append('#2ecc71')  # Green
        elif acc >= 0.7:
            colors.append('#f39c12')  # Orange
        elif acc >= 0.5:
            colors.append('#e67e22')  # Dark orange
        else:
            colors.append('#e74c3c')  # Red
    
    bars = ax.bar(x_pos, sorted_accuracies, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)
    
    # Add average line
    avg_acc = accuracies.mean()
    ax.axhline(y=avg_acc, color='blue', linestyle='-', linewidth=2.5,
               label=f'Average: {avg_acc:.4f} ({avg_acc*100:.2f}%)', alpha=0.7)
    
    # Add threshold lines
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.4, label='Perfect (100%)')
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.3)
    ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=1.5, alpha=0.3)
    
    # Customize
    ax.set_xlabel('Activity Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (Correct / Total)', fontsize=14, fontweight='bold')
    ax.set_title(f'{method_name}: Per-Class Accuracy from Confusion Matrix\n(Sorted by Accuracy)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_classes, rotation=90, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add labels for poor performers
    for i, (acc, count, total, cls) in enumerate(zip(sorted_accuracies, sorted_counts, sorted_totals, sorted_classes)):
        if acc < 0.8:
            ax.text(i, acc + 0.02, f'{int(count)}/{int(total)}', 
                   ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_per_class_accuracy_from_cm.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    # Return accuracy data
    return pd.DataFrame({
        'Class': classes,
        'Correct': diagonal,
        'Total': totals,
        'Accuracy': accuracies
    }).sort_values('Accuracy', ascending=False)


def analyze_confusion_pairs(cm_df, output_dir, method_name='Dawid-Skene', top_n=20):
    """
    Analyze and visualize the most confused class pairs.
    """
    # Get all off-diagonal confusion counts
    confusion_pairs = []
    
    for i, true_class in enumerate(cm_df.index):
        for j, pred_class in enumerate(cm_df.columns):
            if i != j:  # Off-diagonal
                count = cm_df.iloc[i, j]
                if count > 0:
                    confusion_pairs.append({
                        'True_Class': true_class,
                        'Predicted_Class': pred_class,
                        'Count': count
                    })
    
    # Sort by count
    confusion_df = pd.DataFrame(confusion_pairs).sort_values('Count', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_confusion_pairs.csv')
    confusion_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Plot top N confusions
    top_confusions = confusion_df.head(top_n)
    
    if len(top_confusions) > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        labels = [f"{row['True_Class']}\n→ {row['Predicted_Class']}" 
                 for _, row in top_confusions.iterrows()]
        counts = top_confusions['Count'].values
        
        y_pos = np.arange(len(labels))
        
        bars = ax.barh(y_pos, counts, color='coral', alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_name}: Top {top_n} Most Confused Class Pairs\n(True Class → Predicted Class)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{int(count)}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_top_confusions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    return confusion_df


def create_accuracy_summary(cm_df, accuracy_df, output_dir, method_name='Dawid-Skene'):
    """
    Create summary statistics from confusion matrix.
    """
    total_samples = cm_df.sum().sum()
    correct_predictions = np.diag(cm_df.values).sum()
    overall_accuracy = correct_predictions / total_samples
    
    summary = {
        'Metric': [],
        'Value': []
    }
    
    summary['Metric'].append('Total Samples')
    summary['Value'].append(f"{int(total_samples)}")
    
    summary['Metric'].append('Correct Predictions')
    summary['Value'].append(f"{int(correct_predictions)}")
    
    summary['Metric'].append('Overall Accuracy')
    summary['Value'].append(f"{overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    summary['Metric'].append('Number of Classes')
    summary['Value'].append(f"{len(cm_df)}")
    
    # Perfect classifications
    perfect = (accuracy_df['Accuracy'] == 1.0).sum()
    summary['Metric'].append('Classes with 100% Accuracy')
    summary['Value'].append(f"{perfect} ({perfect/len(cm_df)*100:.1f}%)")
    
    # Good classifications
    good = (accuracy_df['Accuracy'] >= 0.9).sum()
    summary['Metric'].append('Classes with ≥90% Accuracy')
    summary['Value'].append(f"{good} ({good/len(cm_df)*100:.1f}%)")
    
    # Poor classifications
    poor = (accuracy_df['Accuracy'] < 0.7).sum()
    summary['Metric'].append('Classes with <70% Accuracy')
    summary['Value'].append(f"{poor} ({poor/len(cm_df)*100:.1f}%)")
    
    # Best and worst
    best_class = accuracy_df.iloc[0]
    worst_class = accuracy_df.iloc[-1]
    
    summary['Metric'].append('Best Class')
    summary['Value'].append(f"{best_class['Class']} ({best_class['Accuracy']:.4f})")
    
    summary['Metric'].append('Worst Class')
    summary['Value'].append(f"{worst_class['Class']} ({worst_class['Accuracy']:.4f})")
    
    # Average accuracy
    avg_acc = accuracy_df['Accuracy'].mean()
    summary['Metric'].append('Average Class Accuracy')
    summary['Value'].append(f"{avg_acc:.4f} ({avg_acc*100:.2f}%)")
    
    # Misclassifications
    total_misclassifications = total_samples - correct_predictions
    summary['Metric'].append('Total Misclassifications')
    summary['Value'].append(f"{int(total_misclassifications)} ({total_misclassifications/total_samples*100:.2f}%)")
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary)
    
    # Save
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_cm_summary_statistics.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    return summary_df


def plot_class_distribution(cm_df, output_dir, method_name='Dawid-Skene'):
    """
    Plot the distribution of samples per class.
    """
    class_counts = cm_df.sum(axis=1).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    x_pos = np.arange(len(class_counts))
    bars = ax.bar(x_pos, class_counts.values, color='steelblue', alpha=0.7, 
                  edgecolor='black', linewidth=0.5)
    
    # Add average line
    avg_count = class_counts.mean()
    ax.axhline(y=avg_count, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_count:.1f}', alpha=0.7)
    
    ax.set_xlabel('Activity Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title(f'{method_name}: Sample Distribution per Class\n(Sorted by Count)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_counts.index, rotation=90, ha='right', fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_class_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("=" * 100)
    print("CONFUSION MATRIX VISUALIZATION")
    print("=" * 100)
    
    # Input file
    input_file = 'evaluation_results/dawid_skene_confusion_matrix.csv'
    output_dir = 'evaluation_results/visualizations'
    method_name = 'Dawid-Skene'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 Input: {input_file}")
    print(f"📁 Output: {output_dir}/")
    
    # Load data
    print(f"\n📊 Loading confusion matrix...")
    cm_df = load_confusion_matrix(input_file)
    print(f"   ✓ Loaded {len(cm_df)}x{len(cm_df.columns)} confusion matrix")
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    
    print(f"\n1. Full confusion matrix heatmap...")
    plot_confusion_matrix_heatmap(cm_df, output_dir, method_name)
    
    print(f"\n2. Normalized confusion matrix...")
    plot_normalized_confusion_matrix(cm_df, output_dir, method_name)
    
    print(f"\n3. Per-class accuracy from diagonal...")
    accuracy_df = plot_per_class_accuracy_from_cm(cm_df, output_dir, method_name)
    
    print(f"\n4. Most confused class pairs...")
    confusion_pairs_df = analyze_confusion_pairs(cm_df, output_dir, method_name, top_n=20)
    
    print(f"\n5. Class sample distribution...")
    plot_class_distribution(cm_df, output_dir, method_name)
    
    print(f"\n6. Summary statistics...")
    summary_df = create_accuracy_summary(cm_df, accuracy_df, output_dir, method_name)
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print(f"\n📊 Confusion Matrix Statistics:")
    for _, row in summary_df.iterrows():
        print(f"   {row['Metric']:30s}: {row['Value']}")
    
    print(f"\n🔍 Top 5 Most Confused Pairs:")
    for i, row in confusion_pairs_df.head(5).iterrows():
        print(f"   {row['True_Class']:25s} → {row['Predicted_Class']:25s}: {int(row['Count'])} errors")
    
    print(f"\n📁 Generated Files:")
    print(f"   • {method_name.lower().replace(' ', '_')}_confusion_matrix.png")
    print(f"   • {method_name.lower().replace(' ', '_')}_confusion_matrix_normalized.png")
    print(f"   • {method_name.lower().replace(' ', '_')}_per_class_accuracy_from_cm.png")
    print(f"   • {method_name.lower().replace(' ', '_')}_top_confusions.png")
    print(f"   • {method_name.lower().replace(' ', '_')}_class_distribution.png")
    print(f"   • {method_name.lower().replace(' ', '_')}_confusion_pairs.csv")
    print(f"   • {method_name.lower().replace(' ', '_')}_cm_summary_statistics.csv")
    
    print("\n" + "=" * 100)
    print("VISUALIZATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
