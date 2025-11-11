"""
Dawid-Skene Visualization Script

This script generates:
1. Annotator accuracy heatmaps for each classifier (final iteration)
2. Label convergence tracking across all iterations (CSV)
3. Confusion matrix visualizations for each annotator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple
from dawid_skene import DawidSkene


class DawidSkeneVisualizer(DawidSkene):
    """Extended Dawid-Skene class that tracks convergence history."""
    
    def __init__(self, max_iterations=100, tolerance=1e-6):
        super().__init__(max_iterations, tolerance)
        self.convergence_history = []  # Store label predictions at each iteration
        self.log_likelihood_history = []  # Store log-likelihood at each iteration
        
    def fit(self, annotations: Dict[str, Dict[str, str]], 
            classes: List[str], 
            annotators: List[str]):
        """
        Fit model and track convergence history.
        """
        self.class_labels = classes
        self.num_classes = len(classes)
        self.annotator_names = annotators
        self.num_annotators = len(annotators)
        
        # Create mappings
        class_to_idx = {c: i for i, c in enumerate(classes)}
        annotator_to_idx = {a: i for i, a in enumerate(annotators)}
        
        # Convert annotations to matrix format
        items = sorted(annotations.keys())
        num_items = len(items)
        
        response = np.full((num_items, self.num_annotators), -1, dtype=int)
        
        for item_idx, item_id in enumerate(items):
            for annotator_name, label in annotations[item_id].items():
                if annotator_name in annotator_to_idx and label in class_to_idx:
                    annotator_idx = annotator_to_idx[annotator_name]
                    class_idx = class_to_idx[label]
                    response[item_idx, annotator_idx] = class_idx
        
        # Initialize parameters
        self._initialize_parameters(response, num_items)
        
        # EM algorithm with tracking
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iterations):
            # E-step
            self._e_step(response, num_items)
            
            # M-step
            self._m_step(response, num_items)
            
            # Calculate log-likelihood
            log_likelihood = self._compute_log_likelihood(response, num_items)
            self.log_likelihood_history.append(log_likelihood)
            
            # Store current predictions (based on item_classes)
            current_predictions = {}
            for item_idx, item in enumerate(items):
                predicted_idx = np.argmax(self.item_classes[item_idx])
                current_predictions[item] = classes[predicted_idx]
            self.convergence_history.append(current_predictions)
            
            # Check convergence
            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                print(f"   Converged after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = log_likelihood
        else:
            print(f"   Reached maximum iterations ({self.max_iterations})")
        
        self.items = items  # Store item order for prediction
        return self


def load_data():
    """Load ground truth and all classifier predictions."""
    print("📂 Loading data...")
    
    # Load ground truth
    ground_truth = pd.read_csv('classifiers/sampled_labels.csv')
    ground_truth = ground_truth.rename(columns={'filename': 'video_name', 'label': 'true_label'})
    
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
            print(f"   ✓ Loaded {clf_name}: {len(df)} predictions")
        else:
            print(f"   ✗ {file_path} not found, skipping {clf_name}")
    
    return ground_truth, predictions


def prepare_annotations(ground_truth, predictions, classifier_subset):
    """Prepare annotations in Dawid-Skene format."""
    # Merge all predictions
    merged = ground_truth.copy()
    for clf_name in classifier_subset:
        if clf_name in predictions:
            pred_df = predictions[clf_name].rename(columns={'predicted_class': clf_name})
            merged = merged.merge(pred_df, on='video_name', how='inner')
    
    # Create annotations dictionary
    annotations = {}
    for idx, row in merged.iterrows():
        video_name = row['video_name']
        annotations[video_name] = {}
        for clf_name in classifier_subset:
            if clf_name in row.index and pd.notna(row[clf_name]):
                annotations[video_name][clf_name] = row[clf_name]
    
    return annotations


def plot_annotator_heatmap(error_matrix, annotator_name, classes, output_dir):
    """
    Plot confusion matrix heatmap for a single annotator.
    
    Args:
        error_matrix: 2D array [true_class, predicted_class]
        annotator_name: Name of the annotator
        classes: List of class names
        output_dir: Output directory for saving
    """
    plt.figure(figsize=(20, 18))
    
    # Create heatmap
    sns.heatmap(
        error_matrix,
        annot=False,  # Don't show numbers (too many classes)
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Probability'},
        vmin=0,
        vmax=1
    )
    
    plt.title(f'Annotator Accuracy Matrix: {annotator_name}\n(Final Iteration)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class (Annotator Response)', fontsize=12, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    
    # Rotate labels for readability
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{annotator_name}_accuracy_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved heatmap: {output_path}")
    plt.close()


def plot_diagonal_accuracy(error_matrix, annotator_name, classes, output_dir):
    """
    Plot the diagonal (correct prediction) accuracies for each class.
    """
    diagonal_accuracies = np.diag(error_matrix)
    
    plt.figure(figsize=(16, 6))
    
    x_pos = np.arange(len(classes))
    colors = ['green' if acc > 0.7 else 'orange' if acc > 0.4 else 'red' 
              for acc in diagonal_accuracies]
    
    plt.bar(x_pos, diagonal_accuracies, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=np.mean(diagonal_accuracies), color='blue', linestyle='--', 
                linewidth=2, label=f'Average: {np.mean(diagonal_accuracies):.3f}')
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (Diagonal Probability)', fontsize=12, fontweight='bold')
    plt.title(f'Per-Class Accuracy for {annotator_name}\n(Probability of Correct Classification)', 
              fontsize=14, fontweight='bold')
    plt.xticks(x_pos, classes, rotation=90, ha='right', fontsize=8)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{annotator_name}_per_class_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved per-class accuracy: {output_path}")
    plt.close()


def save_convergence_history(convergence_history, items, output_path):
    """
    Save label convergence history to CSV.
    
    Args:
        convergence_history: List of dictionaries (one per iteration)
        items: List of item names
        output_path: Path to save CSV
    """
    # Create DataFrame with iterations as columns
    data = {'video_name': items}
    
    for iteration, predictions in enumerate(convergence_history):
        col_name = f'iteration_{iteration + 1}'
        data[col_name] = [predictions.get(item, 'N/A') for item in items]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"   ✓ Saved convergence history: {output_path}")
    
    return df


def plot_convergence_metrics(log_likelihood_history, convergence_history, 
                             ground_truth_dict, output_dir):
    """
    Plot convergence metrics: log-likelihood and accuracy over iterations.
    """
    iterations = range(1, len(log_likelihood_history) + 1)
    
    # Calculate accuracy at each iteration
    accuracies = []
    for predictions in convergence_history:
        correct = sum(1 for item, pred in predictions.items() 
                     if item in ground_truth_dict and pred == ground_truth_dict[item])
        accuracy = correct / len(predictions) if predictions else 0
        accuracies.append(accuracy)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot log-likelihood
    ax1.plot(iterations, log_likelihood_history, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Log-Likelihood', fontsize=12, fontweight='bold')
    ax1.set_title('Dawid-Skene Convergence: Log-Likelihood', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(iterations, accuracies, 'g-', linewidth=2, marker='s')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Dawid-Skene Convergence: Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    
    # Add final accuracy text
    final_acc = accuracies[-1] if accuracies else 0
    ax2.text(0.02, 0.98, f'Final Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'convergence_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved convergence metrics: {output_path}")
    plt.close()


def analyze_label_changes(convergence_df, output_dir):
    """
    Analyze how many labels changed between iterations.
    """
    iteration_cols = [col for col in convergence_df.columns if col.startswith('iteration_')]
    
    changes_data = []
    for i in range(len(iteration_cols) - 1):
        curr_col = iteration_cols[i]
        next_col = iteration_cols[i + 1]
        
        changes = (convergence_df[curr_col] != convergence_df[next_col]).sum()
        changes_data.append({
            'From_Iteration': i + 1,
            'To_Iteration': i + 2,
            'Num_Changes': changes,
            'Percent_Changed': (changes / len(convergence_df)) * 100
        })
    
    changes_df = pd.DataFrame(changes_data)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'label_changes_per_iteration.csv')
    changes_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"   ✓ Saved label changes: {output_path}")
    
    # Plot changes
    plt.figure(figsize=(12, 6))
    plt.plot(changes_df['From_Iteration'], changes_df['Num_Changes'], 
             'r-', linewidth=2, marker='o', markersize=8)
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Label Changes', fontsize=12, fontweight='bold')
    plt.title('Label Changes Between Iterations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'label_changes_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved label changes plot: {output_path}")
    plt.close()
    
    return changes_df


def run_visualization_analysis(classifier_subset=None):
    """
    Main function to run visualization analysis.
    
    Args:
        classifier_subset: List of classifiers to include (None = all)
    """
    print("=" * 100)
    print("DAWID-SKENE VISUALIZATION ANALYSIS")
    print("=" * 100)
    
    # Load data
    ground_truth, predictions = load_data()
    all_classes = sorted(ground_truth['true_label'].unique())
    
    # Default to all classifiers
    if classifier_subset is None:
        classifier_subset = list(predictions.keys())
    
    print(f"\n📊 Dataset: {len(ground_truth)} videos, {len(all_classes)} classes")
    print(f"🔧 Using {len(classifier_subset)} classifiers: {', '.join(classifier_subset)}")
    
    # Prepare annotations
    print("\n🔹 Preparing annotations...")
    annotations = prepare_annotations(ground_truth, predictions, classifier_subset)
    print(f"   ✓ Prepared annotations for {len(annotations)} videos")
    
    # Create output directory
    output_dir = 'dawid_skene_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}/")
    
    # Fit Dawid-Skene with tracking
    print("\n🔹 Fitting Dawid-Skene model with convergence tracking...")
    model = DawidSkeneVisualizer(max_iterations=100, tolerance=1e-6)
    model.fit(annotations, all_classes, classifier_subset)
    
    print(f"\n✓ Model converged in {len(model.convergence_history)} iterations")
    
    # Generate heatmaps for each annotator
    print("\n🎨 Generating annotator accuracy heatmaps...")
    for j, annotator in enumerate(classifier_subset):
        print(f"\n   Annotator: {annotator}")
        error_matrix = model.error_matrices[j]
        
        # Full heatmap
        plot_annotator_heatmap(error_matrix, annotator, all_classes, output_dir)
        
        # Per-class accuracy bar chart
        plot_diagonal_accuracy(error_matrix, annotator, all_classes, output_dir)
    
    # Save convergence history
    print("\n💾 Saving convergence history...")
    convergence_path = os.path.join(output_dir, 'label_convergence_history.csv')
    convergence_df = save_convergence_history(
        model.convergence_history, 
        model.items, 
        convergence_path
    )
    
    # Create ground truth dictionary
    ground_truth_dict = dict(zip(ground_truth['video_name'], ground_truth['true_label']))
    
    # Plot convergence metrics
    print("\n📈 Plotting convergence metrics...")
    plot_convergence_metrics(
        model.log_likelihood_history,
        model.convergence_history,
        ground_truth_dict,
        output_dir
    )
    
    # Analyze label changes
    print("\n🔍 Analyzing label changes...")
    changes_df = analyze_label_changes(convergence_df, output_dir)
    
    # Save annotator accuracy summary
    print("\n📊 Saving annotator accuracy summary...")
    annotator_accuracies = model.get_annotator_accuracy()
    accuracy_df = pd.DataFrame([
        {'Annotator': annotator, 'Estimated_Accuracy': acc}
        for annotator, acc in annotator_accuracies.items()
    ]).sort_values('Estimated_Accuracy', ascending=False)
    
    accuracy_path = os.path.join(output_dir, 'annotator_accuracies.csv')
    accuracy_df.to_csv(accuracy_path, index=False, float_format='%.6f')
    print(f"   ✓ Saved annotator accuracies: {accuracy_path}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print(f"\n📊 Annotator Accuracies:")
    for _, row in accuracy_df.iterrows():
        print(f"   {row['Annotator']:15s}: {row['Estimated_Accuracy']:.4f} ({row['Estimated_Accuracy']*100:.2f}%)")
    
    print(f"\n🔄 Convergence:")
    print(f"   Total iterations: {len(model.convergence_history)}")
    print(f"   Final log-likelihood: {model.log_likelihood_history[-1]:.2f}")
    
    final_predictions = model.convergence_history[-1]
    correct = sum(1 for item, pred in final_predictions.items() 
                 if item in ground_truth_dict and pred == ground_truth_dict[item])
    final_accuracy = correct / len(final_predictions)
    print(f"   Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    print(f"\n📁 Generated files:")
    print(f"   • {len(classifier_subset) * 2} annotator visualizations (heatmaps + bar charts)")
    print(f"   • label_convergence_history.csv (predictions at each iteration)")
    print(f"   • label_changes_per_iteration.csv (tracking changes)")
    print(f"   • convergence_metrics.png (log-likelihood and accuracy plots)")
    print(f"   • label_changes_plot.png")
    print(f"   • annotator_accuracies.csv")
    
    print("\n" + "=" * 100)
    print("VISUALIZATION ANALYSIS COMPLETE")
    print("=" * 100)
    
    return model, convergence_df, changes_df


if __name__ == "__main__":
    # Run with all classifiers
    print("\n🚀 Running visualization analysis with ALL classifiers...")
    model, convergence_df, changes_df = run_visualization_analysis()
    
    print("\n✨ Analysis complete! Check the 'dawid_skene_visualizations' directory for outputs.")
