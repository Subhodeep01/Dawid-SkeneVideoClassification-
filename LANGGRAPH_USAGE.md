# LangGraph Video Classification Orchestrator

A LangGraph-based framework that orchestrates multiple video classification models with intelligent caching.

## Architecture

The framework uses **LangGraph** to create a state machine that:
1. Loads videos from the specified range
2. Runs each enabled classifier in sequence
3. Uses cached predictions when available (if cache is enabled)
4. Aggregates all results into a single CSV file

### Flow Diagram

```
Start → Load Videos → Gemini → Twelve Labs → GPT-4o → GPT-5-mini → Replicate → MoonDream → Qwen → Aggregate → End
```

Each classifier node:
- Checks if the classifier is enabled
- Looks for cached predictions (if cache enabled)
- Falls back to API call if no cache or cache disabled
- Stores results in the shared state

## Features

✅ **Intelligent Caching**: Automatically uses existing predictions from CSV files  
✅ **Selective Classifiers**: Enable only specific classifiers  
✅ **Range Processing**: Process specific video ranges (start-end)  
✅ **Parallel-Ready**: Easy to modify for parallel execution  
✅ **Error Handling**: Gracefully handles API failures  
✅ **Aggregated Output**: Single CSV with all classifier predictions  

## Usage

### Basic Usage (All Videos, All Classifiers, With Cache)

```bash
python video_classification_graph.py
```

### Process Specific Range

```bash
# Process videos 1-10
python video_classification_graph.py --start 1 --end 10

# Process videos 100-200
python video_classification_graph.py --start 100 --end 200
```

### Force API Calls (Disable Cache)

```bash
python video_classification_graph.py --no-cache
```

### Use Specific Classifiers Only

```bash
# Only use Gemini and GPT-4o
python video_classification_graph.py --classifiers gemini gpt4o

# Only use GPT-5-mini
python video_classification_graph.py --classifiers gpt5mini

# Multiple classifiers
python video_classification_graph.py --classifiers gemini twelvelabs gpt5mini qwen
```

### Combined Examples

```bash
# Videos 1-50, only Gemini, GPT-5-mini and Qwen, no cache
python video_classification_graph.py --start 1 --end 50 --classifiers gemini gpt5mini qwen --no-cache

# Videos 501-1000, all classifiers, use cache
python video_classification_graph.py --start 501 --end 1000
```

## Available Classifiers

| ID | Name | Cache File |
|----|------|------------|
| `gemini` | Gemini | `predictions/gemini_predictions.csv` |
| `twelvelabs` | Twelve Labs | `predictions/twelvelabs_predictions.csv` |
| `gpt4o` | GPT-4o-mini | `predictions/gpt4o_predictions.csv` |
| `gpt5mini` | GPT-5-mini | `predictions/gpt-5-mini_predictions.csv` |
| `replicate` | Replicate (LLaVA-13b) | `predictions/replicate_predictions.csv` |
| `moondream` | MoonDream2 | `predictions/moondream_predictions.csv` |
| `qwen` | Qwen-VL | `predictions/qwen_predictions.csv` |

## Cache Management

### How Caching Works

1. When `use_cache=True` (default), the framework first looks in `predictions/` directory
2. If a prediction exists for a video, it uses that instead of calling the API
3. Cache entries are marked with `(cache)` suffix in the output
4. If no cache exists or cache is disabled, API is called

### Setting Up Cache Files

Move your existing prediction CSVs to the `predictions/` directory:

```bash
# Move existing predictions
mv gemini_predictions3.csv predictions/gemini_predictions.csv
mv twelvelabs_predictions2.csv predictions/twelvelabs_predictions.csv
mv gpt-5-mini_predictions2.csv predictions/gpt-5-mini_predictions.csv
mv qwen_predictions2.csv predictions/qwen_predictions.csv
# ... etc
```

### Cache File Format

CSV files should have these columns:
- `video_name` or `filename` or `video`
- `predicted_class` or `prediction` or `class`

Example:
```csv
video_name,predicted_class
001.mp4,skateboarding
002.mp4,jogging
```

## Output

### aggregated_predictions.csv

Single CSV file with all classifier predictions:

```csv
video_name,Gemini,Twelve Labs,GPT-4o-mini,GPT-5-mini,Replicate (LLaVA-13b),MoonDream2,Qwen-VL
001.mp4,skateboarding (cache),skateboarding (cache),skateboarding,skateboarding (cache),skateboarding,abseiling,skateboarding (cache)
002.mp4,jogging (cache),jogging (cache),jogging,jogging (cache),jogging,jogging,jogging (cache)
```

The `(cache)` suffix indicates the prediction was loaded from cache.

## Programmatic Usage

```python
from video_classification_graph import run_classification

# Run with custom settings
final_state = run_classification(
    start_index=1,
    end_index=100,
    use_cache=True,
    enabled_classifiers=["gemini", "gpt5mini", "qwen", "gpt4o"]
)

# Access results
for result in final_state["results"]:
    print(f"{result.video_filename}: {result.predicted_class} (cache={result.used_cache})")
```

## Advanced: Modifying the Graph

### Add a New Classifier

1. Add to `CLASSIFIERS` dict:
```python
"mynewmodel": {
    "name": "My New Model",
    "module": "classifiers.mynewmodel_classifier",
    "function": "classify_video_mynewmodel",
    "cache_file": "predictions/mynewmodel_predictions.csv"
}
```

2. Add a node:
```python
def classify_with_mynewmodel_node(state: GraphState) -> GraphState:
    if "mynewmodel" not in state["enabled_classifiers"]:
        return state
    return _classify_with_classifier(state, "mynewmodel")
```

3. Wire it in the graph:
```python
workflow.add_node("mynewmodel", classify_with_mynewmodel_node)
workflow.add_edge("qwen", "mynewmodel")
workflow.add_edge("mynewmodel", "aggregate")
```

### Enable Parallel Execution

Modify the edges to run classifiers in parallel:

```python
# Instead of sequential edges
workflow.add_edge("load_videos", "gemini")
workflow.add_edge("load_videos", "twelvelabs")
workflow.add_edge("load_videos", "gpt4o")
# ... etc

# Then use conditional edges to wait for all
workflow.add_conditional_edges(
    "gemini",
    lambda state: "aggregate" if all_classifiers_done(state) else "wait",
    {"aggregate": "aggregate", "wait": "wait"}
)
```

## Troubleshooting

### "No cache found for X"
- Move the prediction CSV to `predictions/` directory
- Ensure filename matches the `cache_file` in `CLASSIFIERS`

### Import errors
- Make sure all classifier scripts are in `classifiers/` directory
- Check that function names match in the classifier modules

### API calls when cache expected
- Verify video filename matches exactly in cache CSV
- Check CSV column names (should be `video_name` and `predicted_class`)

## Dependencies

```bash
pip install langgraph langchain langchain-core
```

All classifier dependencies (openai, google-generativeai, replicate, dashscope, etc.) are required for their respective classifiers.
