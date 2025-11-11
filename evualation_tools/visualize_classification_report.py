"""
Visualization Script for Classification Report

Creates visualizations from the classification report CSV:
1. Per-class accuracy (F1-score) bar chart
2. Precision-Recall-F1 heatmap for all classes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def load_classification_report(filepath):
    """Load and clean the classification report CSV."""
    df = pd.read_csv(filepath, index_col=0)
    
    # Remove summary rows (accuracy, macro avg, weighted avg)
    summary_rows = ['accuracy', 'macro avg', 'weighted avg']
    df_classes = df[~df.index.isin(summary_rows)].copy()
    
    # Sort by F1-score descending
    df_classes = df_classes.sort_values('f1-score', ascending=False)
    
    return df_classes, df


def plot_per_class_metrics(df_classes, output_dir, method_name='Dawid-Skene'):
    """
    Create bar chart showing precision, recall, and F1-score for each class.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Prepare data
    classes = df_classes.index.tolist()
    x_pos = np.arange(len(classes))
    width = 0.25
    
    precision = df_classes['precision'].values
    recall = df_classes['recall'].values
    f1_score = df_classes['f1-score'].values
    
    # Create bars
    bars1 = ax.bar(x_pos - width, precision, width, label='Precision', 
                   alpha=0.8, color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos, recall, width, label='Recall', 
                   alpha=0.8, color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x_pos + width, f1_score, width, label='F1-Score', 
                   alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    # Add reference lines
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='90% threshold')
    ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='70% threshold')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='50% threshold')
    
    # Customize plot
    ax.set_xlabel('Activity Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(f'{method_name}: Per-Class Performance Metrics\n(Sorted by F1-Score)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add average line
    avg_f1 = f1_score.mean()
    ax.axhline(y=avg_f1, color='purple', linestyle='-', linewidth=2, 
               label=f'Average F1: {avg_f1:.3f}')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_per_class_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_f1_score_only(df_classes, output_dir, method_name='Dawid-Skene'):
    """
    Create bar chart showing only F1-score with color coding.
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    classes = df_classes.index.tolist()
    x_pos = np.arange(len(classes))
    f1_scores = df_classes['f1-score'].values
    
    # Color code based on performance
    colors = []
    for score in f1_scores:
        if score >= 0.9:
            colors.append('#2ecc71')  # Green - Excellent
        elif score >= 0.7:
            colors.append('#f39c12')  # Orange - Good
        elif score >= 0.5:
            colors.append('#e67e22')  # Dark orange - Fair
        else:
            colors.append('#e74c3c')  # Red - Poor
    
    bars = ax.bar(x_pos, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add average line
    avg_f1 = f1_scores.mean()
    ax.axhline(y=avg_f1, color='blue', linestyle='-', linewidth=2.5, 
               label=f'Average F1: {avg_f1:.4f} ({avg_f1*100:.2f}%)', alpha=0.7)
    
    # Add threshold lines
    ax.axhline(y=0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.4)
    ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=1.5, alpha=0.4)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.4)
    
    # Customize
    ax.set_xlabel('Activity Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_title(f'{method_name}: F1-Score per Class\n(Green ≥0.9, Orange ≥0.7, Dark Orange ≥0.5, Red <0.5)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars for low performers
    for i, (score, bar) in enumerate(zip(f1_scores, bars)):
        if score < 0.7:  # Label poor performers
            ax.text(bar.get_x() + bar.get_width()/2, score + 0.02, 
                   f'{score:.3f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_f1_scores.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_metrics_heatmap(df_classes, output_dir, method_name='Dawid-Skene'):
    """
    Create heatmap showing precision, recall, and F1-score for all classes.
    """
    # Prepare data matrix
    data_matrix = df_classes[['precision', 'recall', 'f1-score']].values.T
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 6))
    
    # Create heatmap
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(df_classes)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(df_classes.index, rotation=90, ha='right', fontsize=9)
    ax.set_yticklabels(['Precision', 'Recall', 'F1-Score'], fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
    cbar.set_label('Score', fontsize=12, fontweight='bold')
    
    # Add values in cells
    for i in range(3):
        for j in range(len(df_classes)):
            value = data_matrix[i, j]
            color = 'white' if value < 0.5 else 'black'
            text = ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                          color=color, fontsize=7, fontweight='bold')
    
    # Title
    ax.set_title(f'{method_name}: Performance Metrics Heatmap\n(All Activity Classes)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_metrics_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_support_vs_f1(df_classes, output_dir, method_name='Dawid-Skene'):
    """
    Scatter plot showing relationship between support (sample size) and F1-score.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    support = df_classes['support'].values
    f1_scores = df_classes['f1-score'].values
    
    # Color code by F1-score
    colors = []
    for score in f1_scores:
        if score >= 0.9:
            colors.append('#2ecc71')
        elif score >= 0.7:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')
    
    # Scatter plot
    scatter = ax.scatter(support, f1_scores, c=colors, s=100, alpha=0.6, 
                        edgecolors='black', linewidth=1)
    
    # Add class labels for poor performers
    for idx, (sup, f1, label) in enumerate(zip(support, f1_scores, df_classes.index)):
        if f1 < 0.7:
            ax.annotate(label, (sup, f1), fontsize=8, 
                       xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # Add trend line
    z = np.polyfit(support, f1_scores, 1)
    p = np.poly1d(z)
    ax.plot(support, p(support), "r--", alpha=0.5, linewidth=2, 
            label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
    
    ax.set_xlabel('Support (Number of Samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{method_name}: F1-Score vs Sample Size\n(Correlation Analysis)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    
    # Add correlation coefficient
    corr = np.corrcoef(support, f1_scores)[0, 1]
    ax.text(0.02, 0.98, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_support_vs_f1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_statistics(df_classes, df_full, output_dir, method_name='Dawid-Skene'):
    """
    Create a summary statistics table.
    """
    summary = {
        'Metric': [],
        'Value': []
    }
    
    # Overall metrics
    summary['Metric'].append('Overall Accuracy')
    summary['Value'].append(f"{df_full.loc['accuracy', 'precision']:.4f} ({df_full.loc['accuracy', 'precision']*100:.2f}%)")
    
    summary['Metric'].append('Macro Average F1')
    summary['Value'].append(f"{df_full.loc['macro avg', 'f1-score']:.4f} ({df_full.loc['macro avg', 'f1-score']*100:.2f}%)")
    
    summary['Metric'].append('Weighted Average F1')
    summary['Value'].append(f"{df_full.loc['weighted avg', 'f1-score']:.4f} ({df_full.loc['weighted avg', 'f1-score']*100:.2f}%)")
    
    # Class-level statistics
    summary['Metric'].append('Number of Classes')
    summary['Value'].append(str(len(df_classes)))
    
    summary['Metric'].append('Classes with F1 ≥ 0.9')
    summary['Value'].append(f"{(df_classes['f1-score'] >= 0.9).sum()} ({(df_classes['f1-score'] >= 0.9).sum() / len(df_classes) * 100:.1f}%)")
    
    summary['Metric'].append('Classes with F1 ≥ 0.7')
    summary['Value'].append(f"{(df_classes['f1-score'] >= 0.7).sum()} ({(df_classes['f1-score'] >= 0.7).sum() / len(df_classes) * 100:.1f}%)")
    
    summary['Metric'].append('Classes with F1 < 0.7')
    summary['Value'].append(f"{(df_classes['f1-score'] < 0.7).sum()} ({(df_classes['f1-score'] < 0.7).sum() / len(df_classes) * 100:.1f}%)")
    
    # Best and worst
    best_class = df_classes['f1-score'].idxmax()
    worst_class = df_classes['f1-score'].idxmin()
    
    summary['Metric'].append('Best Class (F1)')
    summary['Value'].append(f"{best_class} ({df_classes.loc[best_class, 'f1-score']:.4f})")
    
    summary['Metric'].append('Worst Class (F1)')
    summary['Value'].append(f"{worst_class} ({df_classes.loc[worst_class, 'f1-score']:.4f})")
    
    # Average metrics
    summary['Metric'].append('Average Precision')
    summary['Value'].append(f"{df_classes['precision'].mean():.4f}")
    
    summary['Metric'].append('Average Recall')
    summary['Value'].append(f"{df_classes['recall'].mean():.4f}")
    
    summary['Metric'].append('Average F1-Score')
    summary['Value'].append(f"{df_classes['f1-score'].mean():.4f}")
    
    # Total samples
    summary['Metric'].append('Total Samples')
    summary['Value'].append(f"{int(df_classes['support'].sum())}")
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_summary_statistics.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    return summary_df


def main():
    """Main function to generate all visualizations."""
    print("=" * 100)
    print("CLASSIFICATION REPORT VISUALIZATION")
    print("=" * 100)
    
    # Input file
    input_file = 'evaluation_results/dawid_skene_classification_report.csv'
    output_dir = 'evaluation_results/visualizations'
    method_name = 'Dawid-Skene'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 Input: {input_file}")
    print(f"📁 Output: {output_dir}/")
    
    # Load data
    print(f"\n📊 Loading classification report...")
    df_classes, df_full = load_classification_report(input_file)
    print(f"   ✓ Loaded {len(df_classes)} classes")
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    
    print(f"\n1. Per-class metrics (3 metrics bar chart)...")
    plot_per_class_metrics(df_classes, output_dir, method_name)
    
    print(f"\n2. F1-scores only (color-coded bar chart)...")
    plot_f1_score_only(df_classes, output_dir, method_name)
    
    print(f"\n3. Metrics heatmap...")
    plot_metrics_heatmap(df_classes, output_dir, method_name)
    
    print(f"\n4. Support vs F1-score scatter plot...")
    plot_support_vs_f1(df_classes, output_dir, method_name)
    
    print(f"\n5. Summary statistics...")
    summary_df = create_summary_statistics(df_classes, df_full, output_dir, method_name)
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print(f"\n📊 Performance Summary:")
    for _, row in summary_df.iterrows():
        print(f"   {row['Metric']:30s}: {row['Value']}")
    
    print(f"\n📁 Generated Files:")
    print(f"   • {method_name.lower().replace(' ', '_')}_per_class_metrics.png")
    print(f"   • {method_name.lower().replace(' ', '_')}_f1_scores.png")
    print(f"   • {method_name.lower().replace(' ', '_')}_metrics_heatmap.png")
    print(f"   • {method_name.lower().replace(' ', '_')}_support_vs_f1.png")
    print(f"   • {method_name.lower().replace(' ', '_')}_summary_statistics.csv")
    
    print("\n" + "=" * 100)
    print("VISUALIZATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
