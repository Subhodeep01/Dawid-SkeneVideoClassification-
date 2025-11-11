# Video Classification with LangGraph & Dawid-Skene

This project implements video classification using multiple Large Language Model APIs orchestrated through LangGraph, with advanced probabilistic aggregation using the Dawid-Skene algorithm.

## Overview

Classifies 1000 videos across 60 action classes using:
- **Gemini API** (Google's multimodal LLM) - 93.6% accuracy
- **GPT-5-mini** (OpenAI) - 93.3% accuracy
- **Twelve Labs API** (Specialized video understanding) - 86.1% accuracy
- **GPT-4o-mini** (OpenAI) - 84.7% accuracy
- **Qwen-VL** (Alibaba Cloud) - 71.8% accuracy
- **Replicate (LLaVA-13b)** - 64.2% accuracy
- **MoonDream2** - 5.8% accuracy

**Aggregation Methods:**
- **Ensemble (Majority Voting)**: 93.60% accuracy
- **Dawid-Skene (EM Algorithm)**: 93.80% accuracy (best performance)

## Dataset

- **Total Videos**: 1000 videos
- **Number of Classes**: 60 action classes
- **Source**: Kinetics-400 dataset
- **Resolution**: Upscaled to minimum 360x360 for API compatibility

### Action Classes

The 60 classes include: abseiling, applauding, applying cream, baby waking up, balloon blowing, bandaging, bench pressing, blasting sand, canoeing or kayaking, capoeira, changing oil, changing wheel, cooking on campfire, dancing ballet, dancing charleston, dancing macarena, doing nails, driving car, dunking basketball, feeding goats, fixing hair, frying vegetables, hurdling, javelin throw, jogging, juggling soccer ball, laughing, laying bricks, lunge, making snowman, moving furniture, plastering, playing badminton, playing chess, playing didgeridoo, playing keyboard, playing trombone, playing xylophone, pole vault, pumping fist, pushing wheelchair, riding elephant, riding mountain bike, riding unicycle, ripping paper, sharpening knives, shuffling cards, sign language interpreting, skateboarding, snatch weight lifting, snorkeling, spray painting, squat, swinging legs, tango dancing, trimming or shaving beard, tying bow tie, unloading truck, vault, waiting in line.

## Project Structure

```
DSforVidClassify/
├── video_classification_graph.py     # Main LangGraph orchestrator
├── dawid_skene.py                     # Dawid-Skene EM algorithm
├── metadata.txt                       # Dataset metadata
├── .env                               # API keys (NOT in git)
├── classifiers/                       # Classifier modules
│   ├── gemini_classifier.py
│   ├── gpt4o_classifier.py
│   ├── gpt5_mini_classifier.py
│   ├── twelvelabs_classifier.py
│   ├── qwen_classifier.py
│   ├── replicate_classifier.py
│   └── moondream_classifier.py
├── predictions/                       # Classifier outputs (CSV)
├── evaluation_results/                # Performance metrics
├── dawid_skene_visualizations/       # Heatmaps, convergence plots
├── evualation_tools/                 # Visualization scripts
│   ├── evaluation.py
│   ├── dawid_skene_visualization.py
│   ├── visualize_classification_report.py
│   └── visualize_confusion_matrix.py
└── dataset_tools/                    # Dataset preparation scripts
```

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DSforVidClassify
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
```

### 3. Install Dependencies

```powershell
# Core dependencies
pip install numpy pandas scikit-learn

# LangGraph and LangChain
pip install langgraph langchain langchain-core

# Visualization
pip install matplotlib seaborn

# Classifier APIs (install only what you need)
pip install google-generativeai       # Gemini
pip install twelvelabs                 # Twelve Labs
pip install openai                     # GPT-4o-mini, GPT-5-mini
pip install replicate                  # Replicate (LLaVA, MoonDream)
pip install dashscope                  # Qwen-VL

# Optional: Video processing
pip install opencv-python              # Frame extraction
pip install fiftyone                   # Dataset management
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
TWELVELABS_API_KEY=your_twelvelabs_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
REPLICATE_API_TOKEN=your_replicate_token_here
DASHSCOPE_API_KEY=your_dashscope_key_here
```

**Get API Keys:**
- **Gemini**: https://aistudio.google.com/apikey
- **Twelve Labs**: https://playground.twelvelabs.io/
- **OpenAI**: https://platform.openai.com/api-keys
- **Replicate**: https://replicate.com/account/api-tokens
- **DashScope (Qwen)**: https://dashscope.console.aliyun.com/apiKey

## Usage

### Running the LangGraph Classification System

The main classification system is orchestrated through `video_classification_graph.py`, which:
- Loads videos from `sampled_videos/` directory
- Runs classifiers in parallel
- Uses cached predictions when available
- Generates ensemble and Dawid-Skene aggregations

#### Basic Usage

```powershell
# Process all videos with all classifiers (uses cache by default)
python video_classification_graph.py
```

#### Command Line Options

```powershell
# Process specific video range (1-based indexing)
python video_classification_graph.py --start 1 --end 100

# Disable cache and force API calls
python video_classification_graph.py --no-cache

# Use only specific classifiers
python video_classification_graph.py --classifiers gemini gpt5mini twelvelabs

# Combine options: Process videos 501-600 with only Gemini and GPT-5-mini, no cache
python video_classification_graph.py --start 501 --end 600 --classifiers gemini gpt5mini --no-cache
```

#### Available Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--start` | int | 1 | Starting video index (1-based) |
| `--end` | int | None | Ending video index (1-based, None = all videos) |
| `--no-cache` | flag | False | Disable cache, force fresh API calls |
| `--classifiers` | list | All | Specific classifiers to enable |

#### Classifier IDs

Use these IDs with the `--classifiers` argument:
- `gemini` - Google Gemini 2.5 Flash
- `gpt5mini` - OpenAI GPT-5-mini
- `twelvelabs` - Twelve Labs Pegasus 1.2
- `gpt4o` - OpenAI GPT-4o-mini
- `qwen` - Alibaba Qwen-VL
- `replicate` - Replicate LLaVA-13b
- `moondream` - MoonDream2

### Output Files

After running `video_classification_graph.py`, you'll get:

```
predictions/
├── ensemble_predictions.csv          # Majority voting results
├── dawid_skene_predictions.csv       # Dawid-Skene EM results
├── gemini_predictions.csv             # Individual classifier outputs
├── gpt-5-mini_predictions.csv
├── twelvelabs_predictions.csv
├── gpt4o_predictions.csv
├── qwen_predictions.csv
├── replicate_predictions.csv
└── moondream_predictions.csv
```

Each CSV contains:
- `video_name`: Video filename
- `predicted_class`: Predicted action class

### Evaluating Performance

To compute accuracy, precision, recall, F1-score, and confusion matrices:

```powershell
cd evualation_tools
python evaluation.py
```

**Output:**
```
evaluation_results/
├── model_comparison.csv                      # Summary of all models
├── ensemble_classification_report.csv        # Per-class metrics
├── ensemble_confusion_matrix.csv
├── dawid_skene_classification_report.csv
├── dawid_skene_confusion_matrix.csv
└── ... (reports for all classifiers)
```

### Visualizing Dawid-Skene Results

The Dawid-Skene algorithm tracks convergence and estimates annotator (classifier) quality. Visualize these results:

#### 1. Generate Annotator Accuracy Heatmaps & Convergence Plots

```powershell
cd evualation_tools
python dawid_skene_visualization.py
```

**Outputs** (in `dawid_skene_visualizations/`):
- **Annotator Accuracy Heatmaps**: 7 confusion matrices (60×60), one per classifier
  - `gemini_accuracy_heatmap.png`
  - `gpt-5-mini_accuracy_heatmap.png`
  - `twelvelabs_accuracy_heatmap.png`
  - `gpt-4o-mini_accuracy_heatmap.png`
  - `qwen-vl_accuracy_heatmap.png`
  - `replicate_(llava-13b)_accuracy_heatmap.png`
  - `moondream2_accuracy_heatmap.png`

- **Per-Class Accuracy Bar Charts**: Diagonal accuracy for each classifier
  - `{classifier}_per_class_accuracy.png` (7 files)

- **Convergence Data**:
  - `label_convergence_history.csv` - Predictions at each EM iteration
  - `label_changes_per_iteration.csv` - Number of label changes per iteration
  - `convergence_metrics.png` - Log-likelihood and accuracy over iterations
  - `label_changes_plot.png` - Visualization of convergence speed

- **Annotator Quality**:
  - `annotator_accuracies.csv` - Estimated accuracy for each classifier

#### 2. Visualize Classification Report (Precision, Recall, F1)

```powershell
cd evualation_tools
python visualize_classification_report.py
```

This generates **per-class performance visualizations** from the Dawid-Skene classification report:

**Outputs** (in `evaluation_results/visualizations/`):
- `dawid_skene_per_class_metrics.png` - Bar chart of precision, recall, F1 for all 60 classes
- `dawid_skene_f1_scores.png` - Color-coded F1-scores only
- `dawid_skene_metrics_heatmap.png` - Heatmap of all metrics
- `dawid_skene_support_vs_f1.png` - Scatter plot of support vs F1-score
- `dawid_skene_summary_statistics.csv` - Overall summary

**Note**: You can modify the script to visualize any classifier by changing:
```python
input_file = 'evaluation_results/gemini_classification_report.csv'  # Change this
method_name = 'Gemini'  # Change this
```

#### 3. Visualize Confusion Matrix

```powershell
cd evualation_tools
python visualize_confusion_matrix.py
```

This generates **confusion matrix visualizations** from the Dawid-Skene confusion matrix:

**Outputs** (in `evaluation_results/visualizations/`):
- `dawid_skene_confusion_matrix_heatmap.png` - Full 60×60 heatmap
- `dawid_skene_per_class_accuracy.png` - Diagonal accuracy bar chart
- `dawid_skene_top_confusions.png` - Most frequent misclassifications
- `dawid_skene_normalized_confusion_matrix.png` - Row-normalized heatmap

**Note**: To visualize other classifiers, modify:
```python
input_file = 'evaluation_results/ensemble_confusion_matrix.csv'  # Change this
method_name = 'Ensemble'  # Change this
```

### Complete Workflow Example

```powershell
# Step 1: Run classification (uses cached predictions if available)
python video_classification_graph.py --start 1 --end 1000

# Step 2: Evaluate all models
cd evualation_tools
python evaluation.py

# Step 3: Visualize Dawid-Skene convergence and annotator quality
python dawid_skene_visualization.py

# Step 4: Visualize per-class metrics (precision, recall, F1)
python visualize_classification_report.py

# Step 5: Visualize confusion matrix
python visualize_confusion_matrix.py
```

## Features

- ✅ **LangGraph Orchestration**: Parallel classifier execution with state management
- ✅ **Intelligent Caching**: Automatically reuses existing predictions
- ✅ **Dawid-Skene Algorithm**: Probabilistic aggregation with annotator quality estimation
- ✅ **Ensemble Aggregation**: Majority voting with tie detection
- ✅ **Flexible Processing**: Select video ranges and specific classifiers
- ✅ **Comprehensive Evaluation**: Accuracy, precision, recall, F1, confusion matrices
- ✅ **Rich Visualizations**: Heatmaps, convergence plots, per-class metrics
- ✅ **Error Handling**: Graceful handling of API errors and rate limits
- ✅ **Resume Support**: Continue from where you left off

## Performance Results

| Method | Accuracy | Type |
|--------|----------|------|
| **Dawid-Skene** | **93.80%** | Probabilistic Aggregation (EM) |
| **Ensemble** | **93.60%** | Majority Voting |
| Gemini | 93.6% | Single Classifier |
| GPT-5-mini | 93.3% | Single Classifier |
| Twelve Labs | 86.1% | Single Classifier |
| GPT-4o-mini | 84.7% | Single Classifier |
| Qwen-VL | 71.8% | Single Classifier |
| Replicate | 64.2% | Single Classifier |
| MoonDream2 | 5.8% | Single Classifier |

**Key Finding**: Dawid-Skene outperforms simple majority voting by 0.20% by correctly weighting classifier reliability.

## API Comparison

| API | Cost/Video | Speed | Accuracy | Native Video | Audio Support |
|-----|-----------|-------|----------|--------------|---------------|
| **Gemini Flash** | $0.002-0.01 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ |
| **GPT-5-mini** | $0.01-0.02 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ (frames) | ❌ |
| **Twelve Labs** | $$$ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ |
| **GPT-4o-mini** | $0.01-0.02 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ (frames) | ❌ |
| **Qwen-VL** | $0.005-0.01 | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | ✅ |
| **Replicate** | $0.01-0.03 | ⭐⭐ | ⭐⭐⭐ | ❌ (frames) | ❌ |
| **MoonDream2** | $0.001-0.01 | ⭐⭐ | ⭐ | ❌ (frames) | ❌ |

## Requirements

- Python 3.9+
- FFmpeg (for video upscaling, optional)
- Sufficient disk space for videos (~5-10GB)
- API keys for the services you want to use

## Troubleshooting

### "No module named 'langgraph'"
```powershell
pip install langgraph langchain langchain-core
```

### "No cache found for X"
- Ensure prediction CSVs are in `predictions/` directory
- Check that filenames match the classifier cache files
- Verify CSV has `video_name` and `predicted_class` columns

### Import errors
- Ensure all classifier scripts are in `classifiers/` directory
- Check that function names match in the classifier modules

### API quota errors
- Classification will save progress and you can resume
- Use `--start` argument to continue from where you left off

## License

This project is for research and educational purposes.

## Acknowledgments

- Dataset: Kinetics-400
- APIs: Google Gemini, OpenAI, Twelve Labs, Alibaba Cloud, Replicate
- Framework: LangGraph for orchestration
- Algorithm: Dawid-Skene (Dawid & Skene, 1979)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dawidskene_video_classification,
  title={Video Classification with LangGraph and Dawid-Skene Aggregation},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/repo}}
}
```