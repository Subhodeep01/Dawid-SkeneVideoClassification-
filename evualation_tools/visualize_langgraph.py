"""
LangGraph Visualization Script

Creates visual representations of the video classification LangGraph workflow:
1. Graph structure diagram using graphviz
2. Node and edge information tables
3. Workflow flowchart
"""

import os
import graphviz
from pathlib import Path


def create_langgraph_visualization(output_dir='langgraph_visualizations'):
    """
    Create visualizations of the video classification LangGraph.
    """
    print("=" * 100)
    print("LANGGRAPH VISUALIZATION")
    print("=" * 100)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}/")
    
    # Define the graph structure based on video_classification_graph.py
    nodes = {
        'load_videos': {
            'label': 'Load Videos',
            'description': 'Load video files and metadata',
            'color': '#3498db',  # Blue
            'shape': 'box'
        },
        'gemini': {
            'label': 'Gemini',
            'description': 'Google Gemini classifier',
            'color': '#2ecc71',  # Green
            'shape': 'ellipse'
        },
        'twelvelabs': {
            'label': 'Twelve Labs',
            'description': 'Twelve Labs classifier',
            'color': '#2ecc71',
            'shape': 'ellipse'
        },
        'gpt4o': {
            'label': 'GPT-4o-mini',
            'description': 'OpenAI GPT-4o-mini classifier',
            'color': '#2ecc71',
            'shape': 'ellipse'
        },
        'gpt5mini': {
            'label': 'GPT-5-mini',
            'description': 'OpenAI GPT-5-mini classifier',
            'color': '#2ecc71',
            'shape': 'ellipse'
        },
        'replicate': {
            'label': 'Replicate\n(LLaVA-13b)',
            'description': 'Replicate LLaVA-13b classifier',
            'color': '#2ecc71',
            'shape': 'ellipse'
        },
        'moondream': {
            'label': 'MoonDream2',
            'description': 'MoonDream2 classifier',
            'color': '#2ecc71',
            'shape': 'ellipse'
        },
        'qwen': {
            'label': 'Qwen-VL',
            'description': 'Qwen-VL classifier',
            'color': '#2ecc71',
            'shape': 'ellipse'
        },
        'ensemble': {
            'label': 'Ensemble\n(Majority Voting)',
            'description': 'Majority voting aggregation',
            'color': '#e74c3c',  # Red
            'shape': 'box'
        },
        'dawid_skene': {
            'label': 'Dawid-Skene\n(EM Algorithm)',
            'description': 'Dawid-Skene probabilistic aggregation',
            'color': '#e74c3c',
            'shape': 'box'
        }
    }
    
    # Define edges (based on the workflow structure)
    edges = {
        'load_videos': ['gemini', 'twelvelabs', 'gpt4o', 'gpt5mini', 'replicate', 'moondream', 'qwen'],
        'gemini': ['ensemble', 'dawid_skene'],
        'twelvelabs': ['ensemble', 'dawid_skene'],
        'gpt4o': ['ensemble', 'dawid_skene'],
        'gpt5mini': ['ensemble', 'dawid_skene'],
        'replicate': ['ensemble', 'dawid_skene'],
        'moondream': ['ensemble', 'dawid_skene'],
        'qwen': ['ensemble', 'dawid_skene'],
    }
    
    # Create main workflow graph
    print("\n🎨 Creating main workflow graph...")
    dot = graphviz.Digraph(
        'LangGraph_Video_Classification',
        comment='Video Classification Workflow',
        format='png'
    )
    
    # Graph attributes
    dot.attr(rankdir='TB')  # Top to Bottom layout
    dot.attr(size='16,12')
    dot.attr(dpi='300')
    dot.attr(bgcolor='white')
    dot.attr(fontname='Arial')
    dot.attr(fontsize='12')
    
    # Add START node
    dot.node('START', 'START', shape='circle', style='filled', fillcolor='#95a5a6', fontcolor='white')
    
    # Add all nodes
    for node_id, node_info in nodes.items():
        dot.node(
            node_id,
            node_info['label'],
            shape=node_info['shape'],
            style='filled',
            fillcolor=node_info['color'],
            fontcolor='white',
            fontsize='11',
            fontname='Arial Bold'
        )
    
    # Add END nodes
    dot.node('END1', 'END', shape='circle', style='filled', fillcolor='#95a5a6', fontcolor='white')
    dot.node('END2', 'END', shape='circle', style='filled', fillcolor='#95a5a6', fontcolor='white')
    
    # Add edges from START
    dot.edge('START', 'load_videos', label='entry_point', fontsize='9')
    
    # Add all edges
    for source, targets in edges.items():
        for target in targets:
            if target in ['ensemble', 'dawid_skene']:
                # Different style for aggregation edges
                dot.edge(source, target, color='#3498db', penwidth='2')
            else:
                # Regular edges
                dot.edge(source, target, color='#2c3e50', penwidth='1.5')
    
    # Add edges to END
    dot.edge('ensemble', 'END1', label='output', fontsize='9', color='#e74c3c', penwidth='2')
    dot.edge('dawid_skene', 'END2', label='output', fontsize='9', color='#e74c3c', penwidth='2')
    
    # Save main graph
    output_path = os.path.join(output_dir, 'langgraph_workflow')
    dot.render(output_path, cleanup=True)
    print(f"   ✓ Saved: {output_path}.png")
    
    # Create simplified graph (grouped by stages)
    print("\n🎨 Creating simplified grouped graph...")
    dot_simple = graphviz.Digraph(
        'LangGraph_Simplified',
        comment='Simplified Video Classification Workflow',
        format='png'
    )
    
    dot_simple.attr(rankdir='TB')
    dot_simple.attr(size='12,10')
    dot_simple.attr(dpi='300')
    dot_simple.attr(bgcolor='white')
    
    # Create subgraphs for parallel execution
    with dot_simple.subgraph(name='cluster_classifiers') as c:
        c.attr(label='Parallel Classifier Execution', fontsize='14', style='dashed', color='#2ecc71')
        c.attr(fontname='Arial Bold')
        
        for node_id in ['gemini', 'twelvelabs', 'gpt4o', 'gpt5mini', 'replicate', 'moondream', 'qwen']:
            node_info = nodes[node_id]
            c.node(node_id, node_info['label'], shape='ellipse', style='filled', 
                   fillcolor=node_info['color'], fontcolor='white', fontsize='10')
    
    with dot_simple.subgraph(name='cluster_aggregation') as c:
        c.attr(label='Parallel Aggregation Methods', fontsize='14', style='dashed', color='#e74c3c')
        c.attr(fontname='Arial Bold')
        
        for node_id in ['ensemble', 'dawid_skene']:
            node_info = nodes[node_id]
            c.node(node_id, node_info['label'], shape='box', style='filled', 
                   fillcolor=node_info['color'], fontcolor='white', fontsize='10')
    
    # Add load_videos and START/END nodes outside clusters
    dot_simple.node('START', 'START', shape='circle', style='filled', fillcolor='#95a5a6', fontcolor='white')
    dot_simple.node('load_videos', 'Load Videos\n& Metadata', shape='box', style='filled', 
                    fillcolor='#3498db', fontcolor='white', fontsize='12', fontname='Arial Bold')
    dot_simple.node('END', 'END', shape='doublecircle', style='filled', fillcolor='#95a5a6', fontcolor='white')
    
    # Add edges
    dot_simple.edge('START', 'load_videos')
    
    # From load_videos to all classifiers
    for clf in ['gemini', 'twelvelabs', 'gpt4o', 'gpt5mini', 'replicate', 'moondream', 'qwen']:
        dot_simple.edge('load_videos', clf, color='#2c3e50')
    
    # From all classifiers to both aggregation methods
    for clf in ['gemini', 'twelvelabs', 'gpt4o', 'gpt5mini', 'replicate', 'moondream', 'qwen']:
        dot_simple.edge(clf, 'ensemble', color='#3498db', style='dashed')
        dot_simple.edge(clf, 'dawid_skene', color='#3498db', style='dashed')
    
    # From aggregation to END
    dot_simple.edge('ensemble', 'END', color='#e74c3c', penwidth='2')
    dot_simple.edge('dawid_skene', 'END', color='#e74c3c', penwidth='2')
    
    # Save simplified graph
    output_path_simple = os.path.join(output_dir, 'langgraph_workflow_simplified')
    dot_simple.render(output_path_simple, cleanup=True)
    print(f"   ✓ Saved: {output_path_simple}.png")
    
    # Create legend
    print("\n🎨 Creating legend...")
    dot_legend = graphviz.Digraph(
        'Legend',
        comment='Graph Legend',
        format='png'
    )
    
    dot_legend.attr(rankdir='TB')
    dot_legend.attr(size='6,8')
    dot_legend.attr(dpi='300')
    dot_legend.attr(bgcolor='white')
    
    # Legend nodes
    dot_legend.node('leg_start', 'Start/End Node', shape='circle', style='filled', fillcolor='#95a5a6', fontcolor='white')
    dot_legend.node('leg_load', 'Data Loading', shape='box', style='filled', fillcolor='#3498db', fontcolor='white')
    dot_legend.node('leg_clf', 'Classifier Node', shape='ellipse', style='filled', fillcolor='#2ecc71', fontcolor='white')
    dot_legend.node('leg_agg', 'Aggregation Node', shape='box', style='filled', fillcolor='#e74c3c', fontcolor='white')
    
    # Edge examples
    dot_legend.node('leg_seq', 'Sequential', shape='plaintext')
    dot_legend.node('leg_par', 'Parallel', shape='plaintext')
    dot_legend.edge('leg_seq', 'leg_par', label='Sequential Edge', color='#2c3e50', penwidth='1.5')
    dot_legend.edge('leg_clf', 'leg_agg', label='Parallel Edge', color='#3498db', penwidth='2', style='dashed')
    
    output_path_legend = os.path.join(output_dir, 'langgraph_legend')
    dot_legend.render(output_path_legend, cleanup=True)
    print(f"   ✓ Saved: {output_path_legend}.png")
    
    # Create node information table (as text file)
    print("\n📋 Creating node information...")
    node_info_path = os.path.join(output_dir, 'node_information.txt')
    with open(node_info_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("LANGGRAPH NODE INFORMATION\n")
        f.write("="*100 + "\n\n")
        
        f.write("1. ENTRY NODE\n")
        f.write("-" * 100 + "\n")
        f.write(f"   Node: load_videos\n")
        f.write(f"   Description: {nodes['load_videos']['description']}\n")
        f.write(f"   Output: List of videos, class labels, cache manager\n\n")
        
        f.write("2. CLASSIFIER NODES (Parallel Execution)\n")
        f.write("-" * 100 + "\n")
        classifiers = ['gemini', 'twelvelabs', 'gpt4o', 'gpt5mini', 'replicate', 'moondream', 'qwen']
        for clf in classifiers:
            f.write(f"   • {nodes[clf]['label']}: {nodes[clf]['description']}\n")
        f.write("\n")
        
        f.write("3. AGGREGATION NODES (Parallel Execution)\n")
        f.write("-" * 100 + "\n")
        f.write(f"   • {nodes['ensemble']['label']}: {nodes['ensemble']['description']}\n")
        f.write(f"     - Method: Simple majority voting\n")
        f.write(f"     - Output: ensemble_predictions.csv\n\n")
        f.write(f"   • {nodes['dawid_skene']['label']}: {nodes['dawid_skene']['description']}\n")
        f.write(f"     - Method: Expectation-Maximization (EM) algorithm\n")
        f.write(f"     - Estimates annotator accuracy and true labels\n")
        f.write(f"     - Output: dawid_skene_predictions.csv\n\n")
        
        f.write("4. EXECUTION FLOW\n")
        f.write("-" * 100 + "\n")
        f.write("   START → load_videos → [All 7 Classifiers in Parallel]\n")
        f.write("   All Classifiers → ensemble (parallel)\n")
        f.write("   All Classifiers → dawid_skene (parallel)\n")
        f.write("   ensemble → END\n")
        f.write("   dawid_skene → END\n\n")
        
        f.write("5. KEY FEATURES\n")
        f.write("-" * 100 + "\n")
        f.write("   • Parallel classifier execution (7 models run simultaneously)\n")
        f.write("   • Caching support (reuse previous predictions)\n")
        f.write("   • Two aggregation methods run in parallel\n")
        f.write("   • State management via GraphState TypedDict\n")
        f.write("   • Batch processing support (start_index to end_index)\n\n")
    
    print(f"   ✓ Saved: {node_info_path}")
    
    # Create edge information table
    print("\n📋 Creating edge information...")
    edge_info_path = os.path.join(output_dir, 'edge_information.txt')
    with open(edge_info_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("LANGGRAPH EDGE INFORMATION\n")
        f.write("="*100 + "\n\n")
        
        total_edges = 0
        for source, targets in edges.items():
            f.write(f"From: {source} ({nodes[source]['label']})\n")
            f.write(f"To:\n")
            for target in targets:
                f.write(f"   → {target} ({nodes[target]['label']})\n")
                total_edges += 1
            f.write("\n")
        
        f.write(f"Total Edges: {total_edges + 1}\n")  # +1 for START edge
        f.write(f"Total Nodes: {len(nodes) + 1}\n")  # +1 for START
        
        f.write("\n" + "="*100 + "\n")
        f.write("EXECUTION PATTERNS\n")
        f.write("="*100 + "\n\n")
        
        f.write("1. Fan-out Pattern (load_videos → classifiers):\n")
        f.write("   1 → 7 parallel branches\n\n")
        
        f.write("2. Fan-in Pattern (classifiers → aggregation):\n")
        f.write("   7 → 2 parallel aggregations\n\n")
        
        f.write("3. Parallel Aggregation:\n")
        f.write("   Both ensemble and dawid_skene run simultaneously\n")
        f.write("   Results saved to separate CSV files\n\n")
    
    print(f"   ✓ Saved: {edge_info_path}")
    
    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print(f"\n📊 Graph Statistics:")
    print(f"   Nodes: {len(nodes)}")
    print(f"   Entry Point: load_videos")
    print(f"   Parallel Classifiers: 7")
    print(f"   Aggregation Methods: 2 (running in parallel)")
    print(f"   Total Edges: {sum(len(targets) for targets in edges.values()) + 1}")
    
    print(f"\n🎯 Execution Pattern:")
    print(f"   1. Load videos and metadata")
    print(f"   2. Run 7 classifiers in parallel")
    print(f"   3. Run 2 aggregation methods in parallel")
    print(f"   4. Output results to CSV files")
    
    print(f"\n📁 Generated Files:")
    print(f"   • langgraph_workflow.png (detailed graph)")
    print(f"   • langgraph_workflow_simplified.png (grouped view)")
    print(f"   • langgraph_legend.png (legend)")
    print(f"   • node_information.txt")
    print(f"   • edge_information.txt")
    
    print("\n" + "=" * 100)
    print("VISUALIZATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    try:
        create_langgraph_visualization()
        print("\n✨ All visualizations created successfully!")
        print("\n💡 Tip: Open the PNG files to view the workflow diagrams.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Note: This script requires graphviz to be installed.")
        print("   Install with: pip install graphviz")
        print("   Also install Graphviz system binary: https://graphviz.org/download/")
