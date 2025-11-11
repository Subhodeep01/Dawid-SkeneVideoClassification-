"""
LangGraph Video Classification Orchestrator
Coordinates multiple video classification models with caching support.
"""

import os
import csv
import json
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Optional
from dataclasses import dataclass, asdict
import operator
import numpy as np

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from dawid_skene import DawidSkene


@dataclass
class VideoInfo:
    """Information about a video to classify"""
    filename: str
    path: str
    index: int


@dataclass
class ClassificationResult:
    """Result from a classifier"""
    classifier_name: str
    video_filename: str
    predicted_class: str
    used_cache: bool
    error: Optional[str] = None


class GraphState(TypedDict):
    """State of the classification graph"""
    videos: List[VideoInfo]
    classes: List[str]  # List of all possible class labels from metadata.txt
    start_index: int  # Starting video index (1-based)
    end_index: int  # Ending video index (1-based)
    current_video_index: int
    results: Annotated[List[ClassificationResult], operator.add]  # Re-add operator.add for parallel execution
    use_cache: bool
    enabled_classifiers: List[str]
    messages: Annotated[List[BaseMessage], operator.add]
    cache_manager: object  # CacheManager instance (using object to avoid forward reference)


# Available classifiers with their cache file paths
CLASSIFIERS = {
    "gemini": {
        "name": "Gemini",
        "module": "classifiers.gemini_classifier",
        "function": "classify_video_gemini",
        "cache_file": "predictions/gemini_predictions.csv"
    },
    "twelvelabs": {
        "name": "Twelve Labs",
        "module": "classifiers.twelvelabs_classifier",
        "function": "classify_video_twelvelabs",
        "cache_file": "predictions/twelvelabs_predictions.csv"
    },
    "gpt4o": {
        "name": "GPT-4o-mini",
        "module": "classifiers.gpt4o_classifier",
        "function": "classify_video_gpt4o",
        "cache_file": "predictions/gpt4o_predictions.csv"
    },
    "gpt5mini": {
        "name": "GPT-5-mini",
        "module": "classifiers.gpt5_mini_classifier",
        "function": "classify_video_gpt5",
        "cache_file": "predictions/gpt-5-mini_predictions.csv"
    },
    "replicate": {
        "name": "Replicate (LLaVA-13b)",
        "module": "classifiers.replicate_classifier",
        "function": "classify_video_replicate",
        "cache_file": "predictions/replicate_predictions.csv"
    },
    "moondream": {
        "name": "MoonDream2",
        "module": "classifiers.moondream_classifier",
        "function": "classify_video_moondream",
        "cache_file": "predictions/moondream_predictions.csv"
    },
    "qwen": {
        "name": "Qwen-VL",
        "module": "classifiers.qwen_classifier",
        "function": "classify_video_qwen",
        "cache_file": "predictions/qwen_predictions.csv"
    }
}


class CacheManager:
    """Manages cached predictions from CSV files"""
    
    def __init__(self, verbose=False):
        self.caches = {}
        self.verbose = verbose
        self._load_all_caches()
    
    def _load_all_caches(self):
        """Load all available prediction caches"""
        for classifier_id, config in CLASSIFIERS.items():
            cache_file = config["cache_file"]
            if os.path.exists(cache_file):
                self.caches[classifier_id] = self._load_cache(cache_file)
                if self.verbose:
                    print(f"  [OK] Loaded cache for {config['name']}: {len(self.caches[classifier_id])} predictions")
            else:
                self.caches[classifier_id] = {}
                if self.verbose:
                    print(f"  [WARN] No cache found for {config['name']}")
    
    def _load_cache(self, cache_file: str) -> Dict[str, str]:
        """Load predictions from a CSV file"""
        cache = {}
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Handle different CSV column names
                    video_name = row.get('video_name') or row.get('filename') or row.get('video')
                    prediction = row.get('predicted_class') or row.get('prediction') or row.get('class')
                    if video_name and prediction:
                        cache[video_name] = prediction
        except Exception as e:
            print(f"  ✗ Error loading cache from {cache_file}: {e}")
        return cache
    
    def get_prediction(self, classifier_id: str, video_filename: str) -> Optional[str]:
        """Get cached prediction for a video"""
        return self.caches.get(classifier_id, {}).get(video_filename)


def load_videos_node(state: GraphState) -> GraphState:
    """Load videos from the sampled_videos directory and classes from metadata.txt"""
    video_dir = "sampled_videos"
    
    # Load classes from metadata.txt
    classes = []
    metadata_file = "metadata.txt"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Parse lines like "abseiling: 24 video(s)"
                if ':' in line and 'video(s)' in line:
                    class_name = line.split(':')[0].strip()
                    classes.append(class_name)
    
    if not classes:
        print("[WARN] Warning: Could not load classes from metadata.txt")
    
    # Load video files and apply range filtering
    all_video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    
    # Get range from state (set during initialization)
    start_index = state.get("start_index", 1)
    end_index = state.get("end_index", len(all_video_files))
    
    # Apply range filter
    start_index = max(1, start_index)
    end_index = min(end_index, len(all_video_files))
    
    selected_files = all_video_files[start_index - 1:end_index]
    
    videos = [
        VideoInfo(
            filename=f,
            path=os.path.join(video_dir, f),
            index=start_index + i
        )
        for i, f in enumerate(selected_files)
    ]
    
    print(f"\n{'='*60}")
    print(f"Video Classification Graph - LangGraph")
    print(f"{'='*60}")
    print(f"Total videos loaded: {len(videos)}")
    print(f"Total classes: {len(classes)}")
    print(f"Enabled classifiers: {', '.join(state['enabled_classifiers'])}")
    print(f"Using cache: {state['use_cache']}")
    print(f"{'='*60}\n")
    
    # Return only the updated fields - LangGraph will merge them into state
    return {
        "videos": videos,
        "classes": classes,
        "messages": [HumanMessage(content=f"Loaded {len(videos)} videos and {len(classes)} classes")]
    }


def classify_with_gemini_node(state: GraphState) -> GraphState:
    """Classify with Gemini model"""
    if "gemini" not in state["enabled_classifiers"]:
        return {}  # Return empty dict for disabled classifiers
    
    return _classify_with_classifier(state, "gemini")


def classify_with_twelvelabs_node(state: GraphState) -> GraphState:
    """Classify with Twelve Labs model"""
    if "twelvelabs" not in state["enabled_classifiers"]:
        return {}  # Return empty dict for disabled classifiers
    
    return _classify_with_classifier(state, "twelvelabs")


def classify_with_gpt4o_node(state: GraphState) -> GraphState:
    """Classify with GPT-4o-mini model"""
    if "gpt4o" not in state["enabled_classifiers"]:
        return {}  # Return empty dict for disabled classifiers
    
    return _classify_with_classifier(state, "gpt4o")


def classify_with_gpt5mini_node(state: GraphState) -> GraphState:
    """Classify with GPT-5-mini model"""
    if "gpt5mini" not in state["enabled_classifiers"]:
        return {}  # Return empty dict for disabled classifiers
    
    return _classify_with_classifier(state, "gpt5mini")


def classify_with_replicate_node(state: GraphState) -> GraphState:
    """Classify with Replicate model"""
    if "replicate" not in state["enabled_classifiers"]:
        return {}  # Return empty dict for disabled classifiers
    
    return _classify_with_classifier(state, "replicate")


def classify_with_moondream_node(state: GraphState) -> GraphState:
    """Classify with MoonDream2 model"""
    if "moondream" not in state["enabled_classifiers"]:
        return {}  # Return empty dict for disabled classifiers
    
    return _classify_with_classifier(state, "moondream")


def classify_with_qwen_node(state: GraphState) -> GraphState:
    """Classify with Qwen-VL model"""
    if "qwen" not in state["enabled_classifiers"]:
        return {}  # Return empty dict for disabled classifiers
    
    return _classify_with_classifier(state, "qwen")


def _classify_with_classifier(state: GraphState, classifier_id: str) -> GraphState:
    """Generic classifier node implementation"""
    cache_manager = state["cache_manager"]  # Use cache manager from state
    config = CLASSIFIERS[classifier_id]
    videos = state["videos"]
    results = []
    
    print(f"\n--- {config['name']} Classifier ---")
    
    for video in videos:
        # Check cache first if enabled
        if state["use_cache"]:
            cached_prediction = cache_manager.get_prediction(classifier_id, video.filename)
            if cached_prediction:
                result = ClassificationResult(
                    classifier_name=config['name'],
                    video_filename=video.filename,
                    predicted_class=cached_prediction,
                    used_cache=True
                )
                results.append(result)
                # print(f"  [{video.index}] {video.filename}: {cached_prediction} (cached)")
                continue
        
        # Call actual classifier API
        try:
            # Dynamically import the classifier module
            import importlib
            module = importlib.import_module(config['module'])
            classify_func = getattr(module, config['function'])
            
            # Get the classes list from the module
            CLASSES = getattr(module, 'CLASSES')
            
            # Call the classifier
            print(f"  [{video.index}] {video.filename}: Calling API...")
            prediction = classify_func(video.path, CLASSES)
            
            result = ClassificationResult(
                classifier_name=config['name'],
                video_filename=video.filename,
                predicted_class=prediction if prediction else "ERROR",
                used_cache=False
            )
            results.append(result)
            print(f"  [{video.index}] {video.filename}: {prediction} (API)")
            
        except Exception as e:
            result = ClassificationResult(
                classifier_name=config['name'],
                video_filename=video.filename,
                predicted_class="ERROR",
                used_cache=False,
                error=str(e)
            )
            results.append(result)
            print(f"  [{video.index}] {video.filename}: ERROR - {e}")
    
    # Return ONLY the fields that are being updated (for parallel execution)
    # Fields with operator.add will be merged by LangGraph
    return {
        "results": results,  # Will be concatenated with existing results
        "messages": [HumanMessage(
            content=f"{config['name']}: Classified {len(results)} videos"
        )]
    }


def ensemble_predictions_node(state: GraphState) -> GraphState:
    """
    Majority voting ensemble node - combines predictions using simple voting.
    Saves results to: ensemble_predictions.csv
    """
    print(f"\n{'='*60}")
    print(f"Ensemble Predictions (Majority Voting)")
    print(f"{'='*60}")
    
    # Organize results by video
    results_by_video = {}
    for result in state["results"]:
        if result.video_filename not in results_by_video:
            results_by_video[result.video_filename] = {}
        results_by_video[result.video_filename][result.classifier_name] = result.predicted_class
    
    ensemble_results = []
    
    for video_filename in sorted(results_by_video.keys()):
        predictions = results_by_video[video_filename]
        
        # Simple majority voting implementation
        vote_counts = {}
        for classifier, predicted_class in predictions.items():
            if predicted_class != "ERROR":
                vote_counts[predicted_class] = vote_counts.get(predicted_class, 0) + 1
        
        # Get the class with most votes
        if vote_counts:
            # Check if there's a clear majority
            sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_votes) == 1:
                # Only one unique vote - clear majority
                ensemble_prediction = sorted_votes[0][0]
                vote_count = sorted_votes[0][1]
            elif sorted_votes[0][1] > sorted_votes[1][1]:
                # First place has more votes than second place - clear majority
                ensemble_prediction = sorted_votes[0][0]
                vote_count = sorted_votes[0][1]
            else:
                # Tie - no clear majority
                ensemble_prediction = "ERROR"
                vote_count = sorted_votes[0][1]
            
            total_votes = len([p for p in predictions.values() if p != "ERROR"])
            confidence = vote_count / total_votes if total_votes > 0 else 0
        else:
            ensemble_prediction = "ERROR"
            confidence = 0.0
            vote_count = 0
            total_votes = 0
        
        ensemble_results.append({
            'video_name': video_filename,
            'ensemble_prediction': ensemble_prediction,
            'confidence': confidence,
            'vote_count': f"{vote_count}/{total_votes}" if vote_counts else "0/0"
        })
        
        # print(f"  {video_filename}: {ensemsble_prediction} (confidence: {confidence:.2%}, votes: {vote_count}/{total_votes})")
    
    # Save ensemble results
    ensemble_file = "ensemble_predictions.csv"
    with open(ensemble_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['video_name', 'ensemble_prediction', 'confidence', 'vote_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ensemble_results)
    
    print(f"\n[OK] Saved ensemble results to: {ensemble_file}")
    print(f"{'='*60}\n")
    
    return {
        "messages": [HumanMessage(content=f"Ensemble complete: {len(ensemble_results)} videos")]
    }


def dawid_skene_node(state: GraphState) -> GraphState:
    """
    Dawid-Skene ensemble aggregation node.
    
    Uses the Dawid-Skene algorithm to aggregate predictions from multiple classifiers,
    accounting for varying classifier accuracies and confusion patterns.
    
    Saves results to: dawid_skene_predictions.csv
    """
    print(f"\n{'='*60}")
    print(f"Dawid-Skene Aggregation")
    print(f"{'='*60}")

    # Organize results by video: {video_filename: {classifier_name: predicted_class}}
    annotations = {}
    for result in state["results"]:
        video_name = result.video_filename
        classifier_name = result.classifier_name
        predicted_class = result.predicted_class
        
        # Skip ERROR predictions
        if predicted_class == "ERROR":
            continue
        
        if video_name not in annotations:
            annotations[video_name] = {}
        annotations[video_name][classifier_name] = predicted_class
    
    # Get list of annotators (classifiers) that actually made predictions
    all_annotators = set()
    for video_annotations in annotations.values():
        all_annotators.update(video_annotations.keys())
    annotator_names = sorted(list(all_annotators))
    
    if len(annotations) == 0:
        print("  [WARN] No valid annotations to aggregate!")
        return {
            "messages": [HumanMessage(content="Dawid-Skene: No valid annotations")]
        }
    
    if len(annotator_names) < 2:
        print(f"  [WARN] Only {len(annotator_names)} annotator(s), Dawid-Skene requires multiple annotators")
        print("  Falling back to simple prediction selection")
        # Fall back to simple selection
        predictions = {}
        for video_name, video_annotations in annotations.items():
            predictions[video_name] = list(video_annotations.values())[0]
    else:
        # Use Dawid-Skene algorithm
        print(f"  Number of videos: {len(annotations)}")
        print(f"  Number of classifiers: {len(annotator_names)}")
        print(f"  Number of classes: {len(state['classes'])}")
        print(f"  Annotators: {', '.join(annotator_names)}")
        
        # Create and fit Dawid-Skene model
        model = DawidSkene(max_iterations=100, tolerance=1e-6)
        model.fit(annotations, state["classes"], annotator_names)
        
        # Get predictions with probabilities
        predictions_with_probs = model.predict(return_probabilities=True)
        predictions = model.predict(return_probabilities=False)
        
        # Get and display annotator accuracies
        accuracies = model.get_annotator_accuracy()

        print(f"\n  Estimated Classifier Accuracies:")
        for annotator, accuracy in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            print(f"    {annotator}: {accuracy:.2%}")
        
        # Save classifier accuracies to separate CSV
        accuracies_file = "dawid_skene_classifier_accuracies.csv"
        with open(accuracies_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['classifier_name', 'estimated_accuracy'])
            writer.writeheader()
            
            for annotator, accuracy in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
                writer.writerow({
                    'classifier_name': annotator,
                    'estimated_accuracy': f"{accuracy:.6f}"
                })
        
        print(f"\n  [OK] Saved classifier accuracies to: {accuracies_file}")

        # print("Classifier reliability matrix:")
        # for names in annotator_names:
        #     model.get_confusion_matrix(names)
    
    # Save predictions to CSV with confidence scores
    output_file = "dawid_skene_predictions.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_name', 'predicted_class', 'confidence'])
        writer.writeheader()
        
        for video_name in sorted(predictions.keys()):
            predicted_class = predictions[video_name]
            
            # Get confidence (probability of predicted class)
            if len(annotator_names) >= 2:
                # For Dawid-Skene, get the probability of the predicted class
                prob_dist = predictions_with_probs[video_name]
                confidence = prob_dist[predicted_class]
            else:
                # For fallback case, confidence is 1.0
                confidence = 1.0
            
            writer.writerow({
                'video_name': video_name,
                'predicted_class': predicted_class,
                'confidence': f"{confidence:.6f}"
            })
    
    print(f"\n  [OK] Saved Dawid-Skene results to: {output_file}")
    print(f"{'='*60}")
    
    return {
        "messages": [HumanMessage(
            content=f"Dawid-Skene: Aggregated {len(predictions)} predictions using {len(annotator_names)} classifiers"
        )]
    }
def create_classification_graph():
    """Create the LangGraph workflow with parallel classifier execution"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("load_videos", load_videos_node)
    workflow.add_node("gemini", classify_with_gemini_node)
    workflow.add_node("twelvelabs", classify_with_twelvelabs_node)
    workflow.add_node("gpt4o", classify_with_gpt4o_node)
    workflow.add_node("gpt5mini", classify_with_gpt5mini_node)
    workflow.add_node("replicate", classify_with_replicate_node)
    workflow.add_node("moondream", classify_with_moondream_node)
    workflow.add_node("qwen", classify_with_qwen_node)
    workflow.add_node("ensemble", ensemble_predictions_node)
    workflow.add_node("dawid_skene", dawid_skene_node)
    
    # Define edges for PARALLEL execution
    workflow.set_entry_point("load_videos")
    
    # All classifiers run in parallel after loading videos
    workflow.add_edge("load_videos", "gemini")
    workflow.add_edge("load_videos", "twelvelabs")
    workflow.add_edge("load_videos", "gpt4o")
    workflow.add_edge("load_videos", "gpt5mini")
    workflow.add_edge("load_videos", "replicate")
    workflow.add_edge("load_videos", "moondream")
    workflow.add_edge("load_videos", "qwen")
    
    # All classifiers converge to BOTH ensemble nodes (running in parallel)
    workflow.add_edge("gemini", "ensemble")
    workflow.add_edge("twelvelabs", "ensemble")
    workflow.add_edge("gpt4o", "ensemble")
    workflow.add_edge("gpt5mini", "ensemble")
    workflow.add_edge("replicate", "ensemble")
    workflow.add_edge("moondream", "ensemble")
    workflow.add_edge("qwen", "ensemble")
    
    workflow.add_edge("gemini", "dawid_skene")
    workflow.add_edge("twelvelabs", "dawid_skene")
    workflow.add_edge("gpt4o", "dawid_skene")
    workflow.add_edge("gpt5mini", "dawid_skene")
    workflow.add_edge("replicate", "dawid_skene")
    workflow.add_edge("moondream", "dawid_skene")
    workflow.add_edge("qwen", "dawid_skene")
    
    # Both ensemble nodes end independently
    workflow.add_edge("ensemble", END)
    workflow.add_edge("dawid_skene", END)
    
    return workflow.compile()


def run_classification(
    start_index: int = 1,
    end_index: int = None,
    use_cache: bool = True,
    enabled_classifiers: List[str] = None
):
    """
    Run video classification with LangGraph
    
    Args:
        start_index: Starting video index (1-based)
        end_index: Ending video index (1-based, None for all)
        use_cache: Whether to use cached predictions
        enabled_classifiers: List of classifier IDs to enable (None for all)
    """
    # Set enabled classifiers
    if enabled_classifiers is None:
        enabled_classifiers = list(CLASSIFIERS.keys())
    
    # Determine end_index if not specified
    if end_index is None:
        video_dir = "sampled_videos"
        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
        end_index = len(video_files)
    
    # Create cache manager once
    print("\nInitializing cache manager...")
    cache_manager = CacheManager(verbose=True)
    
    # Create initial state - videos and classes will be loaded by load_videos_node
    initial_state = {
        "videos": [],  # Will be populated by load_videos_node
        "classes": [],  # Will be populated by load_videos_node
        "start_index": start_index,
        "end_index": end_index,
        "current_video_index": 0,
        "results": [],
        "use_cache": use_cache,
        "enabled_classifiers": enabled_classifiers,
        "messages": [],
        "cache_manager": cache_manager
    }
    
    # Create and run graph
    graph = create_classification_graph()
    final_state = graph.invoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph Video Classification Orchestrator")
    parser.add_argument("--start", type=int, default=1, help="Start video index (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End video index (1-based)")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache, force API calls")
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=list(CLASSIFIERS.keys()),
        default=None,
        help="Specific classifiers to enable"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LangGraph Video Classification Orchestrator")
    print("="*60)
    
    final_state = run_classification(
        start_index=args.start,
        end_index=args.end,
        use_cache=not args.no_cache,
        enabled_classifiers=args.classifiers
    )
    
    print("\n[OK] Classification complete!")
    print(f"Ensemble results saved to:")
    print(f"  - ensemble_predictions.csv (majority voting)")
    print(f"  - dawid_skene_predictions.csv (Dawid-Skene algorithm - add your implementation)")

