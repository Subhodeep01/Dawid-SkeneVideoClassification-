"""
Example usage of the LangGraph Video Classification Orchestrator
"""

from video_classification_graph import run_classification

# Example 1: Process first 10 videos with all classifiers using cache
print("\n" + "="*60)
print("Example 1: First 10 videos, all classifiers, with cache")
print("="*60)

result = run_classification(
    start_index=1,
    end_index=10,
    use_cache=True,
    enabled_classifiers=None  # None means all classifiers
)

print(f"\nProcessed {len(result['videos'])} videos")
print(f"Total predictions: {len(result['results'])}")
print(f"Cached: {sum(1 for r in result['results'] if r.used_cache)}")
print(f"API calls: {sum(1 for r in result['results'] if not r.used_cache)}")

# Example 2: Process specific range with only certain classifiers
print("\n" + "="*60)
print("Example 2: Videos 50-60, only Gemini and Qwen, with cache")
print("="*60)

result = run_classification(
    start_index=50,
    end_index=60,
    use_cache=True,
    enabled_classifiers=["gemini", "qwen"]
)

# Example 3: Force fresh predictions (no cache)
print("\n" + "="*60)
print("Example 3: Videos 1-5, all classifiers, NO CACHE (fresh API calls)")
print("="*60)

result = run_classification(
    start_index=1,
    end_index=5,
    use_cache=False,  # Force API calls
    enabled_classifiers=None
)

print("\n✓ All examples complete!")
print("Check aggregated_predictions.csv for results")
