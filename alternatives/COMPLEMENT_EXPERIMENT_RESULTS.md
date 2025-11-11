# Dawid-Skene Complement M-step: Experimental Results

## Summary

We tested a **Complement M-step** variant of the Dawid-Skene algorithm that incorporates negative evidence by adding `(1 - item_classes[i,k])` when the annotator does NOT give label `l`, then normalizing rows to sum to 1.0.

## 🔬 Experimental Setup

**Test Data:** 1000 videos, 60 classes  
**Classifiers:** 7 models (GPT-4o-mini, GPT-5-mini, Gemini, TwelveLabs, Qwen-VL, Replicate, MoonDream2)  
**Configurations Tested:** All 7 classifiers

## 📊 Results

### Configuration: All 7 Classifiers

| Method | Accuracy | F1 Score | Time | Classifier Accuracy Estimates |
|--------|----------|----------|------|-------------------------------|
| **Standard** | **93.80%** | **92.82%** | 203s | GPT-5: 96.4%, Gemini: 94.4% |
| **Complement** | **3.20%** | **0.10%** | 1033s | All ~2-3% (collapsed!) |

### Key Findings

❌ **Complement M-step FAILED catastrophically:**
- Accuracy dropped from **93.80% → 3.20%** (-90.6%)
- F1 score dropped from **92.82% → 0.10%** (-92.7%)
- 5× slower (1033s vs 203s)
- Only **3.4% prediction agreement** with standard method
- Classifier accuracy estimates all collapsed to ~2-3% (meaningless)

## 🔍 Why It Failed

### Theoretical Problem
The complement M-step calculates:
```
score[l] = Σ_i [I(gave l) × T[i,k] + I(gave ¬l) × (1 - T[i,k])]
```

Then normalizes: `θ[j,k,l] = score[l] / Σ_l score[l]`

### The Issue

**Over-smoothing:** Adding `(1 - T[i,k])` for negative evidence creates massive scores for ALL labels:

```
Example (3 classes, annotator gave label 0):
  Label 0: score = 0.9 (gave it, T[i,0]=0.9)
  Label 1: score = 0.1 + 2×0.95 + 2×0.05 = 2.1 (didn't give it, but added complements)
  Label 2: score = 0.05 + 2×0.95 + 2×0.9 = 3.75
```

After normalization, this creates nearly **uniform distributions** → no discriminative power!

**Loss of Signal:** The complement term dominates because:
- Positive evidence: Added once per match
- Negative evidence: Added for ALL non-matches (multiplied by num_classes - 1)

For 60 classes, negative evidence is added 59× more often!

### Empirical Evidence

**Standard M-step produces confident predictions:**
```
θ[GPT-5, class_k, correct_label] = 0.85  (high confidence)
θ[GPT-5, class_k, wrong_label]   = 0.025 (low confidence)
```

**Complement M-step produces uniform noise:**
```
θ[GPT-5, class_k, any_label] ≈ 0.017 ± 0.005  (all nearly equal!)
```

## ✅ Conclusion

**DO NOT use the Complement M-step!**

### Standard M-step is correct because:
1. ✓ **Strong theoretical foundation** - Maximum likelihood estimation
2. ✓ **Empirically validated** - 93.80% accuracy on real data  
3. ✓ **Computationally efficient** - 5× faster
4. ✓ **Negative evidence is implicit** - Other labels get low probability automatically
5. ✓ **Proper normalization** - Σ_l θ[j,k,l] = 1.0 without artificial scaling

### Complement M-step fails because:
1. ✗ **Over-smoothing** - Creates nearly uniform distributions
2. ✗ **Wrong weighting** - Negative evidence dominates (59× for 60 classes)
3. ✗ **Destroys discrimination** - Can't distinguish good from bad classifiers
4. ✗ **Terrible accuracy** - 3.20% (worse than random guessing)
5. ✗ **Slow convergence** - 5× slower, 100 iterations without improvement

## 📈 Recommendation

**Keep the current Standard M-step implementation in `dawid_skene.py`**

The standard approach:
```python
for l in range(num_classes):
    numerator = 0
    for i in range(num_items):
        if response[i, j] == l:  # Only count positive matches
            numerator += item_classes[i, k]
    
    error_matrices[j, k, l] = numerator / denominator
```

Is mathematically correct, empirically superior, and computationally efficient.

## 🎓 Lesson Learned

While the complement approach seemed intuitive ("use information about what labels were NOT given"), it violates the fundamental principle of the EM algorithm:

**The M-step should maximize the expected log-likelihood, not introduce ad-hoc weightings.**

The standard formulation already captures all information optimally through proper probabilistic inference.

---

**Files:**
- `dawid_skene.py` - Standard implementation (✅ Use this)
- `dawid_skene_complement.py` - Complement variant (❌ Experimental failure)
- `test_dawid_skene_comparison.py` - Comparison test script
- `m_step_alternatives.py` - Theoretical analysis

**Test Date:** November 11, 2025  
**Result:** Standard wins decisively. Complement approach is not viable.
