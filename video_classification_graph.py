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

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage


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
    current_video_index: int
    results: List[ClassificationResult]  # Remove operator.add annotation
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
                    print(f"  ✓ Loaded cache for {config['name']}: {len(self.caches[classifier_id])} predictions")
            else:
                self.caches[classifier_id] = {}
                if self.verbose:
                    print(f"  ⚠ No cache found for {config['name']}")
    
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
    """Load videos from the sampled_videos directory"""
    video_dir = "sampled_videos"
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    
    # Filter by range if specified (videos are already in state from init)
    videos = state["videos"]
    
    print(f"\n{'='*60}")
    print(f"Video Classification Graph - LangGraph")
    print(f"{'='*60}")
    print(f"Total videos loaded: {len(videos)}")
    print(f"Enabled classifiers: {', '.join(state['enabled_classifiers'])}")
    print(f"Using cache: {state['use_cache']}")
    print(f"{'='*60}\n")
    
    state["messages"].append(HumanMessage(content=f"Loaded {len(videos)} videos for classification"))
    return state


def classify_with_gemini_node(state: GraphState) -> GraphState:
    """Classify with Gemini model"""
    if "gemini" not in state["enabled_classifiers"]:
        return state
    
    return _classify_with_classifier(state, "gemini")


def classify_with_twelvelabs_node(state: GraphState) -> GraphState:
    """Classify with Twelve Labs model"""
    if "twelvelabs" not in state["enabled_classifiers"]:
        return state
    
    return _classify_with_classifier(state, "twelvelabs")


def classify_with_gpt4o_node(state: GraphState) -> GraphState:
    """Classify with GPT-4o-mini model"""
    if "gpt4o" not in state["enabled_classifiers"]:
        return state
    
    return _classify_with_classifier(state, "gpt4o")


def classify_with_gpt5mini_node(state: GraphState) -> GraphState:
    """Classify with GPT-5-mini model"""
    if "gpt5mini" not in state["enabled_classifiers"]:
        return state
    
    return _classify_with_classifier(state, "gpt5mini")


def classify_with_replicate_node(state: GraphState) -> GraphState:
    """Classify with Replicate model"""
    if "replicate" not in state["enabled_classifiers"]:
        return state
    
    return _classify_with_classifier(state, "replicate")


def classify_with_moondream_node(state: GraphState) -> GraphState:
    """Classify with MoonDream2 model"""
    if "moondream" not in state["enabled_classifiers"]:
        return state
    
    return _classify_with_classifier(state, "moondream")


def classify_with_qwen_node(state: GraphState) -> GraphState:
    """Classify with Qwen-VL model"""
    if "qwen" not in state["enabled_classifiers"]:
        return state
    
    return _classify_with_classifier(state, "qwen")


def _classify_with_classifier(state: GraphState, classifier_id: str) -> GraphState:
    """Generic classifier node implementation"""
    cache_manager = state["cache_manager"]  # Use cache manager from state
    config = CLASSIFIERS[classifier_id]
    videos = state["videos"]
    results = []
    
    print(f"\n--- {config['name']} Classifier ---")
    print(f"  [DEBUG] State has {len(state['results'])} existing results at start")
    print(f"  [DEBUG] Processing {len(videos)} videos")
    
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
                print(f"  [{video.index}] {video.filename}: {cached_prediction} (cached)")
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
    
    # Add results to state
    print(f"  [DEBUG] Adding {len(results)} results to state (current state has {len(state['results'])} results)")
    
    # Create new results list by extending the existing one
    updated_results = state["results"] + results
    print(f"  [DEBUG] After extend, will have {len(updated_results)} results")
    
    return {
        **state,
        "results": updated_results,
        "messages": state["messages"] + [HumanMessage(
            content=f"{config['name']}: Classified {len(results)} videos"
        )]
    }


def aggregate_results_node(state: GraphState) -> GraphState:
    """Aggregate and save all classification results"""
    print(f"\n{'='*60}")
    print(f"Aggregating Results")
    print(f"{'='*60}")
    
    # Organize results by video
    results_by_video = {}
    for result in state["results"]:
        if result.video_filename not in results_by_video:
            results_by_video[result.video_filename] = []
        results_by_video[result.video_filename].append(result)
    
    # Save aggregated results
    output_file = "aggregated_predictions.csv"
    
    # Get all classifier names
    classifier_names = list(set([r.classifier_name for r in state["results"]]))
    classifier_names.sort()
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['video_name'] + classifier_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for video_filename in sorted(results_by_video.keys()):
            row = {'video_name': video_filename}
            for result in results_by_video[video_filename]:
                suffix = " (cache)" if result.used_cache else ""
                row[result.classifier_name] = result.predicted_class + suffix
            writer.writerow(row)
    
    print(f"✓ Saved aggregated results to: {output_file}")
    
    # Print summary
    total_videos = len(results_by_video)
    
    # Debug: Check for duplicate results
    unique_results = set()
    for r in state["results"]:
        unique_results.add((r.classifier_name, r.video_filename))
    
    total_predictions = len(unique_results)  # Count unique (classifier, video) pairs
    all_predictions = len(state["results"])  # Total including duplicates
    cached_predictions = sum(1 for r in state["results"] if r.used_cache)
    api_predictions = all_predictions - cached_predictions
    
    if all_predictions != total_predictions:
        print(f"\n⚠ WARNING: Found {all_predictions - total_predictions} duplicate results!")
        print(f"  Unique results: {total_predictions}")
        print(f"  Total results (including duplicates): {all_predictions}")
    
    print(f"\nSummary:")
    print(f"  Total videos: {total_videos}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  From cache: {cached_predictions}")
    print(f"  From API: {total_predictions - cached_predictions}")  # Use unique count
    print(f"  Classifiers used: {len(classifier_names)}")
    
    # Debug info
    if all_predictions != total_predictions:
        print(f"\n  [DEBUG] Total entries in state['results']: {all_predictions}")
        print(f"  [DEBUG] Duplicates: {all_predictions - total_predictions}")
    
    print(f"{'='*60}\n")
    
    state["messages"].append(HumanMessage(
        content=f"Aggregated {total_predictions} predictions from {len(classifier_names)} classifiers"
    ))
    
    return state


def create_classification_graph():
    """Create the LangGraph workflow"""
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
    workflow.add_node("aggregate", aggregate_results_node)
    
    # Define edges (execution flow)
    workflow.set_entry_point("load_videos")
    workflow.add_edge("load_videos", "gemini")
    workflow.add_edge("gemini", "twelvelabs")
    workflow.add_edge("twelvelabs", "gpt4o")
    workflow.add_edge("gpt4o", "gpt5mini")
    workflow.add_edge("gpt5mini", "replicate")
    workflow.add_edge("replicate", "moondream")
    workflow.add_edge("moondream", "qwen")
    workflow.add_edge("qwen", "aggregate")
    workflow.add_edge("aggregate", END)
    
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
    # Load videos
    video_dir = "sampled_videos"
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    
    # Apply range filter
    if end_index is None:
        end_index = len(video_files)
    
    start_index = max(1, start_index)
    end_index = min(end_index, len(video_files))
    
    selected_files = video_files[start_index - 1:end_index]
    
    videos = [
        VideoInfo(
            filename=f,
            path=os.path.join(video_dir, f),
            index=start_index + i
        )
        for i, f in enumerate(selected_files)
    ]
    
    # Set enabled classifiers
    if enabled_classifiers is None:
        enabled_classifiers = list(CLASSIFIERS.keys())
    
    # Create cache manager once
    print("\nInitializing cache manager...")
    cache_manager = CacheManager(verbose=True)
    
    # Create initial state
    initial_state = {
        "videos": videos,
        "current_video_index": 0,
        "results": [],
        "use_cache": use_cache,
        "enabled_classifiers": enabled_classifiers,
        "messages": [],
        "cache_manager": cache_manager  # Add cache manager to state
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
    
    print("\n✓ Classification complete!")
    print(f"Results saved to: aggregated_predictions.csv")
