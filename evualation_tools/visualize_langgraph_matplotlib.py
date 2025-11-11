"""
LangGraph Visualization Script (Alternative using matplotlib + networkx)

Creates visual representations of the video classification LangGraph workflow
without requiring Graphviz system binary.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import numpy as np


def create_langgraph_visualization_matplotlib(output_dir='langgraph_visualizations'):
    """
    Create visualizations of the video classification LangGraph using matplotlib.
    """
    print("=" * 100)
    print("LANGGRAPH VISUALIZATION (Matplotlib Version)")
    print("=" * 100)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}/")
    
    # Define the graph structure
    nodes_info = {
        'START': {'label': 'START', 'color': '#95a5a6', 'type': 'control'},
        'load_videos': {'label': 'Load Videos\n& Metadata', 'color': '#3498db', 'type': 'process'},
        'gemini': {'label': 'Gemini', 'color': '#2ecc71', 'type': 'classifier'},
        'twelvelabs': {'label': 'Twelve Labs', 'color': '#2ecc71', 'type': 'classifier'},
        'gpt4o': {'label': 'GPT-4o-mini', 'color': '#2ecc71', 'type': 'classifier'},
        'gpt5mini': {'label': 'GPT-5-mini', 'color': '#2ecc71', 'type': 'classifier'},
        'replicate': {'label': 'Replicate\n(LLaVA)', 'color': '#2ecc71', 'type': 'classifier'},
        'moondream': {'label': 'MoonDream2', 'color': '#2ecc71', 'type': 'classifier'},
        'qwen': {'label': 'Qwen-VL', 'color': '#2ecc71', 'type': 'classifier'},
        'ensemble': {'label': 'Ensemble\n(Majority)', 'color': '#e74c3c', 'type': 'aggregation'},
        'dawid_skene': {'label': 'Dawid-Skene\n(EM)', 'color': '#e74c3c', 'type': 'aggregation'},
        'END': {'label': 'END', 'color': '#95a5a6', 'type': 'control'}
    }
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node_id in nodes_info.keys():
        G.add_node(node_id)
    
    # Add edges
    G.add_edge('START', 'load_videos')
    
    classifiers = ['gemini', 'twelvelabs', 'gpt4o', 'gpt5mini', 'replicate', 'moondream', 'qwen']
    
    # Load videos to classifiers
    for clf in classifiers:
        G.add_edge('load_videos', clf)
    
    # Classifiers to aggregation methods
    for clf in classifiers:
        G.add_edge(clf, 'ensemble')
        G.add_edge(clf, 'dawid_skene')
    
    # Aggregation to END
    G.add_edge('ensemble', 'END')
    G.add_edge('dawid_skene', 'END')
    
    # Create layout - hierarchical
    print("\n🎨 Creating workflow diagram...")
    
    # Manual positioning for better layout
    pos = {
        'START': (0, 5),
        'load_videos': (0, 4),
        'gemini': (-3, 3),
        'twelvelabs': (-2, 3),
        'gpt4o': (-1, 3),
        'gpt5mini': (0, 3),
        'replicate': (1, 3),
        'moondream': (2, 3),
        'qwen': (3, 3),
        'ensemble': (-1, 1.5),
        'dawid_skene': (1, 1.5),
        'END': (0, 0)
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Draw edges first
    for edge in G.edges():
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]
        
        # Determine edge color
        if edge[1] in ['ensemble', 'dawid_skene']:
            color = '#3498db'
            width = 1.5
            alpha = 0.6
        elif edge[0] in classifiers:
            color = '#3498db'
            width = 1.5
            alpha = 0.6
        else:
            color = '#2c3e50'
            width = 2
            alpha = 0.8
        
        ax.annotate('', xy=end_pos, xytext=start_pos,
                   arrowprops=dict(arrowstyle='->', lw=width, color=color, alpha=alpha))
    
    # Draw nodes
    for node_id, node_info in nodes_info.items():
        x, y = pos[node_id]
        
        if node_info['type'] == 'control':
            # Circle for START/END
            circle = plt.Circle((x, y), 0.2, color=node_info['color'], 
                               ec='black', linewidth=2, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y, node_info['label'], ha='center', va='center',
                   fontsize=14, fontweight='bold', color='black', zorder=11)
        
        elif node_info['type'] == 'classifier':
            # Ellipse for classifiers
            ellipse = mpatches.Ellipse((x, y), 0.6, 0.35, color=node_info['color'],
                                      ec='black', linewidth=2, zorder=10)
            ax.add_patch(ellipse)
            ax.text(x, y, node_info['label'], ha='center', va='center',
                   fontsize=12, fontweight='bold', color='black', zorder=11)
        
        else:
            # Rectangle for process/aggregation nodes
            if node_info['type'] == 'process':
                width, height = 0.7, 0.4
            else:
                width, height = 0.8, 0.45
            
            rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                                 boxstyle='round,pad=0.05', 
                                 facecolor=node_info['color'],
                                 edgecolor='black', linewidth=2, zorder=10)
            ax.add_patch(rect)
            ax.text(x, y, node_info['label'], ha='center', va='center',
                   fontsize=13, fontweight='bold', color='black', zorder=11)
    
    # Add stage labels
    ax.text(-4.5, 4, 'Stage 1:\nInitialization', fontsize=14, fontweight='bold', color='black',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    ax.text(-4.5, 3, 'Stage 2:\nParallel\nClassification\n(7 models)', fontsize=14, fontweight='bold', color='black',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.text(-4.5, 1.5, 'Stage 3:\nParallel\nAggregation\n(2 methods)', fontsize=14, fontweight='bold', color='black',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    # Set limits and remove axes
    ax.set_xlim(-5, 4)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')
    
    # Add title
    plt.title('LangGraph Video Classification Workflow', fontsize=22, fontweight='bold', pad=20, color='black')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#95a5a6', label='Control Flow'),
        mpatches.Patch(color='#3498db', label='Data Processing'),
        mpatches.Patch(color='#2ecc71', label='Classifiers (Parallel)'),
        mpatches.Patch(color='#e74c3c', label='Aggregation (Parallel)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'langgraph_workflow.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {output_path}")
    plt.close()
    
    # Create simplified diagram
    print("\n🎨 Creating simplified diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Positions for simplified view
    simple_pos = {
        'START': (1, 5),
        'load_videos': (1, 4.2),
        'classifiers': (1, 3),
        'ensemble': (0.3, 1.5),
        'dawid_skene': (1.7, 1.5),
        'END': (1, 0.3)
    }
    
    # Draw boxes
    boxes = {
        'START': {'pos': simple_pos['START'], 'size': (0.3, 0.3), 'color': '#95a5a6', 'shape': 'circle', 'label': 'START'},
        'load_videos': {'pos': simple_pos['load_videos'], 'size': (1.2, 0.4), 'color': '#3498db', 'shape': 'rect', 'label': 'Load Videos & Metadata'},
        'classifiers': {'pos': simple_pos['classifiers'], 'size': (2.5, 1.2), 'color': '#2ecc71', 'shape': 'rect', 
                       'label': '7 Parallel Classifiers\n\nGemini • Twelve Labs • GPT-4o-mini\nGPT-5-mini • Replicate • MoonDream2 • Qwen-VL'},
        'ensemble': {'pos': simple_pos['ensemble'], 'size': (1.0, 0.6), 'color': '#e74c3c', 'shape': 'rect', 'label': 'Ensemble\n(Majority Voting)'},
        'dawid_skene': {'pos': simple_pos['dawid_skene'], 'size': (1.0, 0.6), 'color': '#e74c3c', 'shape': 'rect', 'label': 'Dawid-Skene\n(EM Algorithm)'},
        'END': {'pos': simple_pos['END'], 'size': (0.3, 0.3), 'color': '#95a5a6', 'shape': 'circle', 'label': 'END'}
    }
    
    for box_id, box_info in boxes.items():
        x, y = box_info['pos']
        
        if box_info['shape'] == 'circle':
            circle = plt.Circle((x, y), 0.15, color=box_info['color'], 
                               ec='black', linewidth=2, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y, box_info['label'], ha='center', va='center',
                   fontsize=13, fontweight='bold', color='black', zorder=11)
        else:
            w, h = box_info['size']
            rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                                 boxstyle='round,pad=0.1',
                                 facecolor=box_info['color'],
                                 edgecolor='black', linewidth=2.5, zorder=10)
            ax.add_patch(rect)
            ax.text(x, y, box_info['label'], ha='center', va='center',
                   fontsize=13, fontweight='bold', color='black', zorder=11)
    
    # Draw arrows
    arrows = [
        ('START', 'load_videos', 'black'),
        ('load_videos', 'classifiers', 'black'),
        ('classifiers', 'ensemble', '#3498db'),
        ('classifiers', 'dawid_skene', '#3498db'),
        ('ensemble', 'END', 'black'),
        ('dawid_skene', 'END', 'black')
    ]
    
    for start, end, color in arrows:
        start_pos = simple_pos[start]
        end_pos = simple_pos[end]
        
        # Adjust for parallel arrows
        if start == 'classifiers' and end in ['ensemble', 'dawid_skene']:
            if end == 'ensemble':
                start_x = start_pos[0] - 0.5
            else:
                start_x = start_pos[0] + 0.5
            start_y = start_pos[1] - 0.6
            
            ax.annotate('', xy=end_pos, xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color=color, alpha=0.7))
        else:
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', lw=3, color=color, alpha=0.8))
    
    # Add annotations
    ax.text(2.5, 3, 'Parallel\nExecution', fontsize=13, fontweight='bold', color='black',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.text(2.5, 1.5, 'Parallel\nAggregation', fontsize=13, fontweight='bold', color='black',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Set limits and remove axes
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    
    plt.title('LangGraph Simplified Workflow', fontsize=20, fontweight='bold', pad=20, color='black')
    
    plt.tight_layout()
    
    output_path_simple = os.path.join(output_dir, 'langgraph_workflow_simplified.png')
    plt.savefig(output_path_simple, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {output_path_simple}")
    plt.close()
    
    # Create node information files (same as before)
    print("\n📋 Creating documentation files...")
    
    node_info_path = os.path.join(output_dir, 'workflow_documentation.txt')
    with open(node_info_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("LANGGRAPH VIDEO CLASSIFICATION WORKFLOW DOCUMENTATION\n")
        f.write("="*100 + "\n\n")
        
        f.write("WORKFLOW OVERVIEW\n")
        f.write("-" * 100 + "\n")
        f.write("A parallel video classification system using LangGraph orchestration.\n")
        f.write("The workflow processes videos through 7 classifiers in parallel, then\n")
        f.write("aggregates results using 2 different methods (also in parallel).\n\n")
        
        f.write("STAGES\n")
        f.write("-" * 100 + "\n\n")
        
        f.write("Stage 1: Initialization\n")
        f.write("  Node: load_videos\n")
        f.write("  • Loads video files from sampled_videos/\n")
        f.write("  • Reads class labels from metadata.txt\n")
        f.write("  • Initializes cache manager\n")
        f.write("  • Sets up video batch (start_index to end_index)\n\n")
        
        f.write("Stage 2: Parallel Classification (7 Models)\n")
        f.write("  All classifiers run simultaneously:\n")
        f.write("  1. Gemini (Google) - 93.6% accuracy\n")
        f.write("  2. GPT-5-mini (OpenAI) - 93.3% accuracy\n")
        f.write("  3. Twelve Labs - 86.1% accuracy\n")
        f.write("  4. GPT-4o-mini (OpenAI) - 84.7% accuracy\n")
        f.write("  5. Qwen-VL - 71.8% accuracy\n")
        f.write("  6. Replicate (LLaVA-13b) - 64.2% accuracy\n")
        f.write("  7. MoonDream2 - 5.8% accuracy\n\n")
        
        f.write("Stage 3: Parallel Aggregation (2 Methods)\n")
        f.write("  Both methods run simultaneously:\n")
        f.write("  1. Ensemble (Majority Voting)\n")
        f.write("     • Simple majority vote across all classifiers\n")
        f.write("     • Returns ERROR on ties (no clear majority)\n")
        f.write("     • Output: ensemble_predictions.csv\n")
        f.write("     • Fast execution\n\n")
        f.write("  2. Dawid-Skene (Probabilistic EM Algorithm)\n")
        f.write("     • Estimates annotator accuracy matrices\n")
        f.write("     • Uses confusion matrices for each classifier\n")
        f.write("     • Iterative refinement via Expectation-Maximization\n")
        f.write("     • Output: dawid_skene_predictions.csv\n")
        f.write("     • Higher accuracy but slower execution\n\n")
        
        f.write("KEY FEATURES\n")
        f.write("-" * 100 + "\n")
        f.write("• Parallel Execution: Classifiers run simultaneously for speed\n")
        f.write("• Caching: Reuses previous predictions when available\n")
        f.write("• Flexible Batching: Process any range of videos (start_index to end_index)\n")
        f.write("• State Management: GraphState TypedDict tracks all information\n")
        f.write("• Error Handling: Graceful failure handling per classifier\n")
        f.write("• Dual Aggregation: Compare simple vs sophisticated methods\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 100 + "\n")
        f.write("Dataset: 1000 videos, 60 activity classes\n\n")
        f.write("Individual Classifier Accuracy:\n")
        f.write("  Best:  Gemini (93.6%)\n")
        f.write("  Worst: MoonDream2 (5.8%)\n\n")
        f.write("Aggregation Results:\n")
        f.write("  Ensemble:     93.60% accuracy\n")
        f.write("  Dawid-Skene:  93.80% accuracy (+0.20% improvement)\n\n")
        f.write("Execution Time (for 1000 videos):\n")
        f.write("  Ensemble:     ~0.15 seconds\n")
        f.write("  Dawid-Skene:  ~80 seconds (500x slower, but more accurate)\n\n")
        
        f.write("OUTPUT FILES\n")
        f.write("-" * 100 + "\n")
        f.write("predictions/\n")
        f.write("  • gemini_predictions.csv\n")
        f.write("  • gpt-5-mini_predictions.csv\n")
        f.write("  • twelvelabs_predictions.csv\n")
        f.write("  • gpt4o_predictions.csv\n")
        f.write("  • qwen_predictions.csv\n")
        f.write("  • replicate_predictions.csv\n")
        f.write("  • moondream_predictions.csv\n")
        f.write("  • ensemble_predictions.csv\n")
        f.write("  • dawid_skene_predictions.csv\n\n")
    
    print(f"   ✓ Saved: {node_info_path}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print(f"\n📊 Graph Statistics:")
    print(f"   Total Nodes: 11 (START, 1 loader, 7 classifiers, 2 aggregators, END)")
    print(f"   Total Edges: 22")
    print(f"   Parallel Execution Points: 2 (classifiers + aggregation)")
    print(f"   Entry Point: load_videos")
    print(f"   Exit Points: 2 (ensemble + dawid_skene)")
    
    print(f"\n📁 Generated Files:")
    print(f"   • langgraph_workflow.png (detailed workflow)")
    print(f"   • langgraph_workflow_simplified.png (simplified view)")
    print(f"   • workflow_documentation.txt (complete documentation)")
    
    print("\n" + "=" * 100)
    print("VISUALIZATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    create_langgraph_visualization_matplotlib()
    print("\n✨ All visualizations created successfully!")
    print("\n💡 Tip: Open the PNG files to view the workflow diagrams.")
