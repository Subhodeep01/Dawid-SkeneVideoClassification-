# Comprehensive Ablation Study - Summary

## Overview
This ablation study analyzes the robustness of Ensemble (majority voting) vs Dawid-Skene aggregation methods by systematically testing different classifier configurations.

## Experimental Setup

### Dataset
- **Videos**: 1000 videos
- **Classes**: 60 action categories
- **Classifiers**: 7 models with varying performance levels

### Classifier Rankings (by individual accuracy)
1. Gemini: 93.6%
2. GPT-5-mini: 93.3%
3. TwelveLabs: 86.1%
4. GPT-4o-mini: 84.7%
5. Qwen-VL: 71.8%
6. Replicate: 64.2%
7. MoonDream2: 5.8%

## Test Configurations

### 1. Remove Top-1 (Gemini)
- **Classifiers**: GPT-5-mini, TwelveLabs, GPT-4o-mini, Qwen-VL, Replicate, MoonDream2
- **Ensemble**: 92.50% accuracy (0.08s)
- **Dawid-Skene**: 92.40% accuracy (233.84s)
- **Winner**: Ensemble by 0.10%
- **DS Classifier Estimates**: GPT-5-mini (95.75%), GPT-4o-mini (89.48%), TwelveLabs (88.64%)

### 2. Remove Top-2 (Gemini, GPT-5-mini)
- **Classifiers**: TwelveLabs, GPT-4o-mini, Qwen-VL, Replicate, MoonDream2
- **Ensemble**: 87.50% accuracy (0.08s)
- **Dawid-Skene**: 88.60% accuracy (193.52s)
- **Winner**: Dawid-Skene by 1.10%
- **DS Classifier Estimates**: GPT-4o-mini (89.46%), TwelveLabs (88.43%), Qwen-VL (75.71%)

### 3. Remove Top-3
- **Classifiers**: GPT-4o-mini, Qwen-VL, Replicate, MoonDream2
- **Ensemble**: 82.80% accuracy (0.05s)
- **Dawid-Skene**: 84.20% accuracy (158.02s)
- **Winner**: Dawid-Skene by 1.40%
- **DS Classifier Estimates**: GPT-4o-mini (87.68%), Qwen-VL (79.02%), Replicate (67.79%)

### 4. Remove Top-4
- **Classifiers**: Qwen-VL, Replicate, MoonDream2
- **Ensemble**: 71.40% accuracy (0.05s)
- **Dawid-Skene**: 68.80% accuracy (124.99s)
- **Winner**: Ensemble by 2.60%
- **DS Classifier Estimates**: Qwen-VL (77.49%), Replicate (74.01%), MoonDream2 (11.15%)

### 5. Gemini + Bottom-1 (MoonDream2)
- **Classifiers**: Gemini, MoonDream2
- **Ensemble**: 93.60% accuracy (0.05s)
- **Dawid-Skene**: 72.50% accuracy (84.02s)
- **Winner**: Ensemble by 21.10%
- **DS Classifier Estimates**: Gemini (65.04%), MoonDream2 (21.07%)
- **Note**: Ensemble simply picks Gemini's predictions (the better classifier)

### 6. Gemini + Bottom-2 (Replicate, MoonDream2)
- **Classifiers**: Gemini, Replicate, MoonDream2
- **Ensemble**: 92.80% accuracy (0.05s)
- **Dawid-Skene**: 84.10% accuracy (122.08s)
- **Winner**: Ensemble by 8.70%
- **DS Classifier Estimates**: Gemini (86.65%), Replicate (73.66%), MoonDream2 (6.95%)

### 7. Gemini + Bottom-3 (Qwen-VL, Replicate, MoonDream2)
- **Classifiers**: Gemini, Qwen-VL, Replicate, MoonDream2
- **Ensemble**: 89.40% accuracy (0.06s)
- **Dawid-Skene**: 87.70% accuracy (162.15s)
- **Winner**: Ensemble by 1.70%
- **DS Classifier Estimates**: Gemini (89.17%), Qwen-VL (79.84%), Replicate (68.24%)

## Key Findings

### 1. Overall Performance
- **Head-to-Head Results**:
  - Dawid-Skene wins: 2/7 configurations
  - Ensemble wins: 5/7 configurations
  - Ties: 0/7 configurations

### 2. Best Configurations
- **Best Ensemble**: Gemini + Bottom-1 (93.60%)
  - Simply chooses Gemini's predictions when only 2 classifiers
- **Best Dawid-Skene**: Remove Top-1 (92.40%)
  - Benefits from having 6 diverse classifiers

### 3. Execution Time
- **Average Ensemble Time**: 0.06s (extremely fast)
- **Average Dawid-Skene Time**: 154.09s (2.5 minutes)
- **Time Difference**: Dawid-Skene is ~2,500x slower

### 4. When Dawid-Skene Wins
Dawid-Skene performs better when:
- **Removing top-2 and top-3 classifiers** (1.10% and 1.40% better)
- You have **4-6 classifiers of moderate quality**
- No single classifier dominates

### 5. When Ensemble Wins
Ensemble performs better when:
- **Very few classifiers** (2-3): Simply picks the best one
- **Very weak classifiers** (Remove Top-4): 71.40% vs 68.80%
- **One strong + many weak classifiers**: Gemini+Bottom configurations

### 6. Classifier Accuracy Estimates
Dawid-Skene's accuracy estimates are generally accurate:
- **Remove Top-1**: Estimates GPT-5-mini at 95.75% (actual: 93.30%)
- **Remove Top-4**: Correctly identifies Qwen-VL > Replicate > MoonDream2
- **Gemini+Bottom1**: Shows confusion when only 2 classifiers (underestimates Gemini)

## Critical Insights

### 1. Quality vs Quantity Trade-off
- **Few high-quality classifiers**: Ensemble wins (simpler is better)
- **Many medium-quality classifiers**: Dawid-Skene wins (can model errors)
- **Many with very weak classifiers**: Ensemble wins (DS gets confused)

### 2. Dawid-Skene Limitations
- **Requires diverse classifiers**: Fails with only 2 classifiers
- **Computationally expensive**: 2,500x slower than Ensemble
- **Can be misled by weak classifiers**: Bottom-1,2,3 experiments show degradation

### 3. Ensemble Strength
- **Extremely fast**: 0.05-0.08 seconds
- **Robust to weak classifiers**: Majority voting filters noise
- **Optimal with few classifiers**: Naturally selects best classifier

### 4. Sweet Spot for Dawid-Skene
- **4-6 classifiers of similar quality** (70-90% range)
- When you've removed very top performers but still have diversity
- Remove Top-2 and Top-3 show this: +1.10% and +1.40% improvement

## Recommendations

### Use Ensemble when:
1. You have fewer than 4 classifiers
2. You have one very strong classifier (>90%) and others are weak
3. Speed is critical (real-time applications)
4. Computational resources are limited

### Use Dawid-Skene when:
1. You have 4-6 classifiers of moderate and similar quality (70-90%)
2. Accuracy improvement of 1-2% justifies 2,500x longer runtime
3. You need classifier accuracy estimates for analysis
4. You have sufficient computational resources

### Avoid Dawid-Skene when:
1. Only 2-3 classifiers available
2. Large quality gap between classifiers (one at 93%, others at 5-64%)
3. Many very weak classifiers in the mix

## Files Generated

### 1. comprehensive_ablation_results.csv
Complete metrics for all 7 configurations:
- Accuracy, Precision, Recall, F1 (macro & weighted)
- For both Ensemble and Dawid-Skene
- Execution times

### 2. performance.csv
Timing comparison:
- Ensemble execution time
- Dawid-Skene execution time
- Time difference

### 3. classifier_accuracies/*.csv (7 files)
Dawid-Skene's estimated classifier accuracies for each configuration:
- One file per experiment
- Shows which classifiers DS considers most reliable
- Useful for understanding DS's internal model

## Conclusion

This ablation study demonstrates that:

1. **Ensemble is generally superior** for this dataset (5/7 wins)
2. **Dawid-Skene has a narrow advantage** in specific scenarios (4-6 moderate classifiers)
3. **Speed matters**: Ensemble is 2,500x faster with comparable or better accuracy
4. **Simple can be better**: When quality varies widely, majority voting is more robust

For the video classification task with these 7 classifiers, **Ensemble (majority voting) is the recommended approach** due to:
- Better overall accuracy (93.60% best vs 92.40%)
- Much faster execution (0.06s vs 154s average)
- More robust to weak classifiers
- Simpler to implement and maintain

Dawid-Skene should only be considered if you can curate a subset of 4-6 similarly-performing classifiers (70-90% range) and the 1-2% accuracy gain justifies the computational cost.
