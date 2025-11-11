"""
Alternative M-step formulations for Dawid-Skene

This file demonstrates different ways to think about the M-step update,
including incorporating negative evidence.
"""

import numpy as np

def standard_m_step(item_classes, response, num_items, num_annotators, num_classes):
    """
    Standard Dawid-Skene M-step.
    
    θ[j, k, l] = P(annotator j gives label l | true class is k)
               = Σ_i [I(response[i,j] == l) × T[i,k]] / Σ_i T[i,k]
    """
    error_matrices = np.zeros((num_annotators, num_classes, num_classes))
    
    for j in range(num_annotators):
        for k in range(num_classes):
            denominator = item_classes[:, k].sum()
            
            if denominator > 0:
                for l in range(num_classes):
                    numerator = 0
                    for i in range(num_items):
                        if response[i, j] == l:
                            numerator += item_classes[i, k]
                    
                    error_matrices[j, k, l] = numerator / denominator
            else:
                error_matrices[j, k, :] = 1.0 / num_classes
    
    return error_matrices


def complement_m_step_WRONG(item_classes, response, num_items, num_annotators, num_classes):
    """
    WRONG: Adding complements breaks probability constraint.
    
    This approach would NOT work because:
    - If response == l: add T[i,k]
    - If response != l: add (1 - T[i,k])
    
    Problem: Σ_l θ[j,k,l] ≠ 1.0
    """
    error_matrices = np.zeros((num_annotators, num_classes, num_classes))
    
    for j in range(num_annotators):
        for k in range(num_classes):
            for l in range(num_classes):
                numerator = 0
                for i in range(num_items):
                    if response[i, j] == l:
                        numerator += item_classes[i, k]
                    else:
                        numerator += (1 - item_classes[i, k])
                
                # This denominator doesn't normalize properly!
                error_matrices[j, k, l] = numerator / num_items
    
    # Rows will NOT sum to 1.0!
    return error_matrices


def complement_m_step_NORMALIZED(item_classes, response, num_items, num_annotators, num_classes):
    """
    Normalized version: Adding complements then normalizing rows.
    
    This approach:
    - If response == l: add T[i,k]
    - If response != l: add (1 - T[i,k])
    - Then normalize each row to sum to 1.0
    
    This satisfies probability constraint, but has different interpretation!
    """
    error_matrices = np.zeros((num_annotators, num_classes, num_classes))
    
    for j in range(num_annotators):
        for k in range(num_classes):
            row_unnormalized = np.zeros(num_classes)
            
            for l in range(num_classes):
                numerator = 0
                for i in range(num_items):
                    if response[i, j] >= 0:  # Valid response
                        if response[i, j] == l:
                            numerator += item_classes[i, k]
                        else:
                            numerator += (1 - item_classes[i, k])
                
                row_unnormalized[l] = numerator
            
            # Normalize the row to sum to 1.0
            row_sum = row_unnormalized.sum()
            if row_sum > 0:
                error_matrices[j, k, :] = row_unnormalized / row_sum
            else:
                error_matrices[j, k, :] = 1.0 / num_classes
    
    return error_matrices


def negative_evidence_m_step(item_classes, response, num_items, num_annotators, num_classes):
    """
    Alternative: Use negative evidence to adjust estimates.
    
    For each (j, k, l), we compute:
    - Positive evidence: Times annotator j gave l when true class was k
    - Negative evidence: Times annotator j gave NOT-l when true class was k
    
    Then use a weighted combination.
    """
    error_matrices = np.zeros((num_annotators, num_classes, num_classes))
    
    for j in range(num_annotators):
        for k in range(num_classes):
            denominator = item_classes[:, k].sum()
            
            if denominator > 0:
                for l in range(num_classes):
                    # Positive evidence: annotator gave label l
                    pos_evidence = 0
                    # Negative evidence: annotator gave other labels
                    neg_evidence = 0
                    
                    for i in range(num_items):
                        if response[i, j] >= 0:  # Valid response
                            if response[i, j] == l:
                                pos_evidence += item_classes[i, k]
                            else:
                                # They gave a different label
                                # This is evidence AGAINST label l
                                neg_evidence += item_classes[i, k]
                    
                    # Standard approach: use only positive evidence
                    error_matrices[j, k, l] = pos_evidence / denominator
                    
                    # Alternative (commented): penalize based on negative evidence
                    # total_evidence = pos_evidence + neg_evidence
                    # if total_evidence > 0:
                    #     error_matrices[j, k, l] = pos_evidence / total_evidence
            else:
                error_matrices[j, k, :] = 1.0 / num_classes
    
    return error_matrices


def binary_vote_m_step(item_classes, response, num_items, num_annotators, num_classes):
    """
    Alternative interpretation: Treat each label as binary vote.
    
    For each (j, k, l):
    - "Yes" votes: Times annotator gave label l when true class was k
    - "No" votes: Times annotator gave NOT-l when true class was k
    
    Then: θ[j,k,l] = P(vote "yes" for l | true class k)
    """
    error_matrices = np.zeros((num_annotators, num_classes, num_classes))
    
    for j in range(num_annotators):
        for k in range(num_classes):
            for l in range(num_classes):
                yes_votes = 0  # Gave label l
                total_opportunities = 0  # All items with true class k
                
                for i in range(num_items):
                    if response[i, j] >= 0:  # Valid response
                        weight = item_classes[i, k]  # P(true class is k)
                        total_opportunities += weight
                        
                        if response[i, j] == l:
                            yes_votes += weight
                
                if total_opportunities > 0:
                    error_matrices[j, k, l] = yes_votes / total_opportunities
                else:
                    error_matrices[j, k, l] = 1.0 / num_classes
    
    return error_matrices


# DEMONSTRATION
if __name__ == "__main__":
    # Simple example
    num_items = 3
    num_annotators = 2
    num_classes = 3
    
    # Item class posteriors (after E-step)
    item_classes = np.array([
        [0.9, 0.05, 0.05],  # Item 0: probably class 0
        [0.1, 0.8, 0.1],    # Item 1: probably class 1
        [0.05, 0.05, 0.9],  # Item 2: probably class 2
    ])
    
    # Annotator responses
    response = np.array([
        [0, 0],  # Item 0: both gave class 0
        [1, 2],  # Item 1: disagree (1 vs 2)
        [2, 2],  # Item 2: both gave class 2
    ])
    
    print("=" * 70)
    print("M-STEP COMPARISON")
    print("=" * 70)
    
    # Standard approach
    print("\n1. STANDARD M-STEP (Correct)")
    print("-" * 70)
    error_mat = standard_m_step(item_classes, response, num_items, num_annotators, num_classes)
    print("Annotator 0, True Class 0:")
    print(f"  P(predict 0|true 0) = {error_mat[0, 0, 0]:.4f}")
    print(f"  P(predict 1|true 0) = {error_mat[0, 0, 1]:.4f}")
    print(f"  P(predict 2|true 0) = {error_mat[0, 0, 2]:.4f}")
    print(f"  Sum = {error_mat[0, 0, :].sum():.4f} ✓")
    
    # Wrong approach
    print("\n2. COMPLEMENT M-STEP (WRONG - Violates probability)")
    print("-" * 70)
    error_mat_wrong = complement_m_step_WRONG(item_classes, response, num_items, num_annotators, num_classes)
    print("Annotator 0, True Class 0:")
    print(f"  P(predict 0|true 0) = {error_mat_wrong[0, 0, 0]:.4f}")
    print(f"  P(predict 1|true 0) = {error_mat_wrong[0, 0, 1]:.4f}")
    print(f"  P(predict 2|true 0) = {error_mat_wrong[0, 0, 2]:.4f}")
    print(f"  Sum = {error_mat_wrong[0, 0, :].sum():.4f} ✗ (should be 1.0!)")
    
    # Normalized complement
    print("\n3. COMPLEMENT M-STEP (NORMALIZED)")
    print("-" * 70)
    error_mat_norm = complement_m_step_NORMALIZED(item_classes, response, num_items, num_annotators, num_classes)
    print("Annotator 0, True Class 0:")
    print(f"  P(predict 0|true 0) = {error_mat_norm[0, 0, 0]:.4f}")
    print(f"  P(predict 1|true 0) = {error_mat_norm[0, 0, 1]:.4f}")
    print(f"  P(predict 2|true 0) = {error_mat_norm[0, 0, 2]:.4f}")
    print(f"  Sum = {error_mat_norm[0, 0, :].sum():.4f} ✓")
    print("\nComparison with Standard:")
    print(f"  Standard: 0={error_mat[0, 0, 0]:.4f}, 1={error_mat[0, 0, 1]:.4f}, 2={error_mat[0, 0, 2]:.4f}")
    print(f"  Normalized Complement: 0={error_mat_norm[0, 0, 0]:.4f}, 1={error_mat_norm[0, 0, 1]:.4f}, 2={error_mat_norm[0, 0, 2]:.4f}")
    print(f"  → Differences: 0={error_mat_norm[0, 0, 0] - error_mat[0, 0, 0]:+.4f}, "
          f"1={error_mat_norm[0, 0, 1] - error_mat[0, 0, 1]:+.4f}, "
          f"2={error_mat_norm[0, 0, 2] - error_mat[0, 0, 2]:+.4f}")
    
    # Binary vote
    print("\n4. BINARY VOTE M-STEP (Same as standard for this case)")
    print("-" * 70)
    error_mat_binary = binary_vote_m_step(item_classes, response, num_items, num_annotators, num_classes)
    print("Annotator 0, True Class 0:")
    print(f"  P(predict 0|true 0) = {error_mat_binary[0, 0, 0]:.4f}")
    print(f"  P(predict 1|true 0) = {error_mat_binary[0, 0, 1]:.4f}")
    print(f"  P(predict 2|true 0) = {error_mat_binary[0, 0, 2]:.4f}")
    print(f"  Sum = {error_mat_binary[0, 0, :].sum():.4f} ✓")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
1. STANDARD M-STEP is mathematically correct:
   ✓ Proper normalization: Σ_l θ[j,k,l] = 1.0
   ✓ Represents P(annotator gives l | true class is k)
   ✓ Negative evidence is implicit (other labels get lower probability)

2. COMPLEMENT (unnormalized) BREAKS the algorithm:
   ✗ Violates probability constraint (sum ≠ 1.0)
   ✗ Double-counts evidence
   ✗ Not a valid probability distribution

3. COMPLEMENT (NORMALIZED) is mathematically valid BUT:
   ✓ Rows sum to 1.0 (valid probability distribution)
   ⚠️  Changes the interpretation and weighting
   ⚠️  Emphasizes disagreements MORE than standard approach
   ⚠️  May or may not improve results - needs empirical testing!
   
   The normalized complement approach gives DIFFERENT probabilities:
   - It weighs "not choosing other labels" heavily
   - This is like a soft version of "one-vs-all" classification
   - Could be useful if annotators are very sparse/selective
   
   Whether this helps depends on your data:
   - If annotators rarely give wrong labels: Standard is better
   - If you want to emphasize "definite no" signals: Complement might help
   - Empirical testing needed to decide!

RECOMMENDATION:
- Start with STANDARD M-step (current implementation)
- If you want to try complement approach, test it empirically
- Compare convergence and final accuracy on validation set
    """)
    
    print("\n" + "=" * 70)
    print("DEEPER ANALYSIS: What does normalization change?")
    print("=" * 70)
    
    # Show detailed comparison
    print("\nFor Annotator 0, True Class 0:")
    print("\nStandard M-step calculates:")
    print("  θ[0,0,l] = (times gave l when true=0) / (all times true=0)")
    print(f"  Result: [{error_mat[0, 0, 0]:.4f}, {error_mat[0, 0, 1]:.4f}, {error_mat[0, 0, 2]:.4f}]")
    
    print("\nComplement M-step calculates:")
    print("  Score for l = (times gave l) + (times gave NOT-l) × weight")
    print("  Then normalizes to sum to 1.0")
    print(f"  Result: [{error_mat_norm[0, 0, 0]:.4f}, {error_mat_norm[0, 0, 1]:.4f}, {error_mat_norm[0, 0, 2]:.4f}]")
    
    print("\nEffect of normalization:")
    print(f"  Label 0 (often given): {error_mat[0, 0, 0]:.4f} → {error_mat_norm[0, 0, 0]:.4f} ({error_mat_norm[0, 0, 0]/error_mat[0, 0, 0]:.2f}x)")
    print(f"  Label 1 (rarely given): {error_mat[0, 0, 1]:.4f} → {error_mat_norm[0, 0, 1]:.4f} ({error_mat_norm[0, 0, 1]/error_mat[0, 0, 1]:.2f}x)")
    print(f"  Label 2 (rarely given): {error_mat[0, 0, 2]:.4f} → {error_mat_norm[0, 0, 2]:.4f} ({error_mat_norm[0, 0, 2]/error_mat[0, 0, 2]:.2f}x)")
    
    print("\n→ Normalized complement INFLATES probabilities for rarely-given labels!")
    print("  This makes the distribution more uniform / less confident.")
    print("  Whether this is good or bad depends on your use case.")

