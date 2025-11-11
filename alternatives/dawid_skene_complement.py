"""
Dawid-Skene Algorithm with COMPLEMENT M-step

This variant uses a normalized complement approach in the M-step where:
- Positive evidence: annotator gave label l
- Negative evidence: annotator gave other labels (weighted by 1 - item_classes)
- Then normalize to get valid probability distribution

This is an experimental variant to test if emphasizing negative evidence helps.
"""

import numpy as np
from typing import Dict, List, Tuple
import csv


class DawidSkeneComplement:
    """
    Dawid-Skene model with Complement M-step variant.
    
    Attributes:
        num_annotators: Number of annotators (classifiers)
        num_classes: Number of possible classes
        max_iterations: Maximum number of EM iterations
        tolerance: Convergence threshold for stopping criterion
        class_priors: Prior probability for each class (π)
        error_matrices: Confusion matrix for each annotator (θ)
        item_classes: Estimated probability distribution over classes for each item
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize the Dawid-Skene Complement model.
        
        Args:
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence threshold (change in log-likelihood)
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.num_annotators = 0
        self.num_classes = 0
        self.class_priors = None
        self.error_matrices = None
        self.item_classes = None
        self.class_labels = []
        self.annotator_names = []
        
    def fit(self, 
            annotations: Dict[str, Dict[str, str]], 
            classes: List[str],
            annotator_names: List[str]) -> 'DawidSkeneComplement':
        """
        Fit the Dawid-Skene Complement model using EM algorithm.
        
        Args:
            annotations: Dict mapping item_id -> {annotator_name: label}
            classes: List of all possible class labels
            annotator_names: List of annotator names (in consistent order)
            
        Returns:
            self (fitted model)
        """
        self.class_labels = classes
        self.num_classes = len(classes)
        self.annotator_names = annotator_names
        self.num_annotators = len(annotator_names)
        
        # Create mappings
        class_to_idx = {c: i for i, c in enumerate(classes)}
        annotator_to_idx = {a: i for i, a in enumerate(annotator_names)}
        
        # Convert annotations to matrix format
        items = sorted(annotations.keys())
        num_items = len(items)
        
        response = np.full((num_items, self.num_annotators), -1, dtype=int)
        
        for item_idx, item_id in enumerate(items):
            for annotator_name, label in annotations[item_id].items():
                if annotator_name in annotator_to_idx and label in class_to_idx:
                    annotator_idx = annotator_to_idx[annotator_name]
                    class_idx = class_to_idx[label]
                    response[item_idx, annotator_idx] = class_idx
        
        # Initialize parameters
        self._initialize_parameters(response, num_items)
        
        # Run EM algorithm
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iterations):
            # E-step: Estimate item classes given current parameters
            self._e_step(response, num_items)
            
            # M-step: Update parameters given current item class estimates
            self._m_step_complement(response, num_items)  # COMPLEMENT VERSION
            
            # Check convergence
            log_likelihood = self._compute_log_likelihood(response, num_items)
            
            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                print(f"  Converged after {iteration + 1} iterations")
                break
                
            prev_log_likelihood = log_likelihood
        
        self.items = items
        return self
    
    def _initialize_parameters(self, response: np.ndarray, num_items: int):
        """Initialize model parameters."""
        # Initialize class priors uniformly
        self.class_priors = np.ones(self.num_classes) / self.num_classes
        
        # Initialize item classes using majority voting
        self.item_classes = np.zeros((num_items, self.num_classes))
        
        for i in range(num_items):
            valid_annotations = response[i, response[i, :] >= 0]
            
            if len(valid_annotations) > 0:
                votes = np.bincount(valid_annotations, minlength=self.num_classes)
                self.item_classes[i, :] = votes / votes.sum()
            else:
                self.item_classes[i, :] = self.class_priors
        
        # Initialize error matrices
        self.error_matrices = np.zeros((self.num_annotators, self.num_classes, self.num_classes))
        
        for j in range(self.num_annotators):
            for k in range(self.num_classes):
                self.error_matrices[j, k, k] = 0.7
                off_diagonal_prob = 0.3 / (self.num_classes - 1) if self.num_classes > 1 else 0
                for l in range(self.num_classes):
                    if l != k:
                        self.error_matrices[j, k, l] = off_diagonal_prob
    
    def _e_step(self, response: np.ndarray, num_items: int):
        """E-step: Estimate posterior probability of true class for each item."""
        for i in range(num_items):
            posterior = np.copy(self.class_priors)
            
            for j in range(self.num_annotators):
                observed_label = response[i, j]
                
                if observed_label >= 0:
                    posterior *= self.error_matrices[j, :, observed_label]
            
            posterior_sum = posterior.sum()
            if posterior_sum > 0:
                self.item_classes[i, :] = posterior / posterior_sum
            else:
                self.item_classes[i, :] = self.class_priors
    
    def _m_step_complement(self, response: np.ndarray, num_items: int):
        """
        M-step with COMPLEMENT approach.
        
        For each (j, k, l):
        - Positive evidence: when annotator j gave label l and item is class k
        - Negative evidence: when annotator j gave NOT-l and item is NOT class k
        - Then normalize each row to sum to 1.0
        """
        # Update class priors
        self.class_priors = self.item_classes.mean(axis=0)
        
        # Update error matrices with complement approach
        for j in range(self.num_annotators):
            for k in range(self.num_classes):
                row_unnormalized = np.zeros(self.num_classes)
                
                for l in range(self.num_classes):
                    score = 0
                    for i in range(num_items):
                        if response[i, j] >= 0:  # Valid response exists
                            if response[i, j] == l:
                                # Positive evidence: gave label l, weight by P(class k)
                                score += self.item_classes[i, k]
                            else:
                                # Negative evidence: gave other label, weight by P(not class k)
                                score += (1 - self.item_classes[i, k])
                    
                    row_unnormalized[l] = score
                
                # Normalize row to sum to 1.0
                row_sum = row_unnormalized.sum()
                if row_sum > 0:
                    self.error_matrices[j, k, :] = row_unnormalized / row_sum
                else:
                    self.error_matrices[j, k, :] = 1.0 / self.num_classes
    
    def _compute_log_likelihood(self, response: np.ndarray, num_items: int) -> float:
        """Compute log-likelihood of the data given current parameters."""
        log_likelihood = 0.0
        
        for i in range(num_items):
            item_likelihood = 0.0
            
            for k in range(self.num_classes):
                class_prob = self.class_priors[k]
                
                obs_prob = 1.0
                for j in range(self.num_annotators):
                    observed_label = response[i, j]
                    if observed_label >= 0:
                        obs_prob *= self.error_matrices[j, k, observed_label]
                
                item_likelihood += class_prob * obs_prob
            
            if item_likelihood > 0:
                log_likelihood += np.log(item_likelihood)
        
        return log_likelihood
    
    def predict(self, return_probabilities: bool = False) -> Dict[str, any]:
        """Get the predicted class for each item."""
        if self.item_classes is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        predictions = {}
        
        for i, item_id in enumerate(self.items):
            if return_probabilities:
                prob_dict = {
                    self.class_labels[k]: float(self.item_classes[i, k])
                    for k in range(self.num_classes)
                }
                predictions[item_id] = prob_dict
            else:
                predicted_class_idx = np.argmax(self.item_classes[i, :])
                predictions[item_id] = self.class_labels[predicted_class_idx]
        
        return predictions
    
    def get_annotator_accuracy(self) -> Dict[str, float]:
        """Compute the accuracy of each annotator."""
        accuracies = {}
        
        for j, annotator_name in enumerate(self.annotator_names):
            accuracy = 0.0
            for k in range(self.num_classes):
                accuracy += self.class_priors[k] * self.error_matrices[j, k, k]
            
            accuracies[annotator_name] = float(accuracy)
        
        return accuracies
    
    def get_confusion_matrix(self, annotator_name: str) -> np.ndarray:
        """Get the confusion matrix for a specific annotator."""
        if annotator_name not in self.annotator_names:
            raise ValueError(f"Unknown annotator: {annotator_name}")
        
        annotator_idx = self.annotator_names.index(annotator_name)
        return self.error_matrices[annotator_idx, :, :]
    
    def save_predictions(self, output_file: str, include_probabilities: bool = True):
        """Save predictions to a CSV file."""
        predictions = self.predict(return_probabilities=include_probabilities)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if include_probabilities:
                fieldnames = ['video_name', 'predicted_class', 'confidence'] + \
                            [f'prob_{c}' for c in self.class_labels]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item_id in self.items:
                    prob_dist = predictions[item_id]
                    predicted_class = max(prob_dist, key=prob_dist.get)
                    confidence = prob_dist[predicted_class]
                    
                    row = {
                        'video_name': item_id,
                        'predicted_class': predicted_class,
                        'confidence': f"{confidence:.4f}"
                    }
                    
                    for c in self.class_labels:
                        row[f'prob_{c}'] = f"{prob_dist[c]:.4f}"
                    
                    writer.writerow(row)
            else:
                fieldnames = ['video_name', 'predicted_class']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item_id, predicted_class in predictions.items():
                    writer.writerow({
                        'video_name': item_id,
                        'predicted_class': predicted_class
                    })
