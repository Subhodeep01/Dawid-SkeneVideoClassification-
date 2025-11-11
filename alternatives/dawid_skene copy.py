"""
Dawid-Skene Algorithm for Multi-Annotator Label Aggregation

This implementation estimates the true labels for items that have been
labeled by multiple annotators (classifiers) with varying levels of expertise.

The algorithm uses Expectation-Maximization (EM) to jointly estimate:
1. The true label for each item
2. The error rate (confusion matrix) for each annotator

References:
- Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer
  error-rates using the EM algorithm. Journal of the Royal Statistical Society, 28(1), 20-28.
"""

import numpy as np
from typing import Dict, List, Tuple
import csv


class DawidSkene:
    """
    Dawid-Skene model for aggregating labels from multiple annotators.
    
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
        Initialize the Dawid-Skene model.
        
        Args:
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence threshold (change in log-likelihood)
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.num_annotators = 0
        self.num_classes = 0
        self.class_priors = None  # π: Prior probability for each class
        self.error_matrices = None  # θ: Confusion matrix for each annotator
        self.item_classes = None  # T: Probability distribution over classes for each item
        self.class_labels = []  # Mapping from class index to class name
        self.annotator_names = []  # Names of annotators
        
    def fit(self, 
            annotations: Dict[str, Dict[str, str]], 
            classes: List[str],
            annotator_names: List[str]) -> 'DawidSkene':
        """
        Fit the Dawid-Skene model using EM algorithm.
        
        Args:
            annotations: Dict mapping item_id -> {annotator_name: label}
                        Example: {"video1.mp4": {"Gemini": "walking", "GPT": "running"}}
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
        # response[i, j] = class index that annotator j assigned to item i
        # -1 indicates missing annotation
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
            self._m_step(response, num_items)
            
            # Check convergence
            log_likelihood = self._compute_log_likelihood(response, num_items)
            
            if iteration > 0 and abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                print(f"  Converged after {iteration + 1} iterations")
                break
                
            prev_log_likelihood = log_likelihood
        
        self.items = items  # Store item order for prediction
        return self
    
    def _initialize_parameters(self, response: np.ndarray, num_items: int):
        """
        Initialize model parameters.
        
        Strategy:
        - Class priors: Uniform distribution
        - Error matrices: Start with majority voting results + small random noise
        - Item classes: Based on majority voting
        """
        # Initialize class priors uniformly
        self.class_priors = np.ones(self.num_classes) / self.num_classes
        
        # Initialize item classes using majority voting
        self.item_classes = np.zeros((num_items, self.num_classes))
        
        for i in range(num_items):
            # Get valid annotations for this item
            valid_annotations = response[i, response[i, :] >= 0]
            
            if len(valid_annotations) > 0:
                # Count votes for each class
                votes = np.bincount(valid_annotations, minlength=self.num_classes)
                # Normalize to get probability distribution
                self.item_classes[i, :] = votes / votes.sum()
            else:
                # If no annotations, use uniform distribution
                self.item_classes[i, :] = self.class_priors
        
        # Initialize error matrices
        # error_matrices[j, k, l] = P(annotator j gives label l | true label is k)
        self.error_matrices = np.zeros((self.num_annotators, self.num_classes, self.num_classes))
        
        for j in range(self.num_annotators):
            for k in range(self.num_classes):
                # Start with diagonal (correct classifications) having high probability
                self.error_matrices[j, k, k] = 0.7
                # Distribute remaining probability uniformly among other classes
                off_diagonal_prob = 0.3 / (self.num_classes - 1) if self.num_classes > 1 else 0
                for l in range(self.num_classes):
                    if l != k:
                        self.error_matrices[j, k, l] = off_diagonal_prob
    
    def _e_step(self, response: np.ndarray, num_items: int):
        """
        E-step: Estimate posterior probability of true class for each item.
        
        For each item i, compute:
        T[i, k] = P(true class is k | observed annotations)
        """
        for i in range(num_items):
            # Compute unnormalized posterior for each class
            posterior = np.copy(self.class_priors)  # Start with prior
            
            for j in range(self.num_annotators):
                observed_label = response[i, j]
                
                if observed_label >= 0:  # Valid annotation exists
                    # Multiply by P(observation | true class)
                    # posterior[k] *= P(annotator j gives label observed_label | true class is k)
                    posterior *= self.error_matrices[j, :, observed_label]
            
            # Normalize to get probability distribution
            posterior_sum = posterior.sum()
            if posterior_sum > 0:
                self.item_classes[i, :] = posterior / posterior_sum
            else:
                # If all posteriors are 0 (shouldn't happen), use prior
                self.item_classes[i, :] = self.class_priors
    
    def _m_step(self, response: np.ndarray, num_items: int):
        """
        M-step: Update parameters given current item class estimates.
        
        Update:
        1. Class priors π[k] = average probability of class k across all items
        2. Error matrices θ[j, k, l] = P(annotator j gives label l | true class is k)
        """
        # Update class priors
        self.class_priors = self.item_classes.mean(axis=0)
        
        # Update error matrices
        for j in range(self.num_annotators):
            for k in range(self.num_classes):
                # Denominator: expected count of items with true class k
                denominator = self.item_classes[:, k].sum()
                
                if denominator > 0:
                    for l in range(self.num_classes):
                        # Numerator: expected count of items where true class is k
                        # and annotator j gave label l
                        numerator = 0
                        for i in range(num_items):
                            if response[i, j] == l:  # Annotator j gave label l for item i
                                numerator += self.item_classes[i, k]  # Weight by P(true class is k)
                        
                        self.error_matrices[j, k, l] = numerator / denominator
                else:
                    # If no items expected to have class k, use uniform distribution
                    self.error_matrices[j, k, :] = 1.0 / self.num_classes

                    
    
    def _compute_log_likelihood(self, response: np.ndarray, num_items: int) -> float:
        """
        Compute log-likelihood of the data given current parameters.
        
        Used for monitoring convergence.
        """
        log_likelihood = 0.0
        
        for i in range(num_items):
            item_likelihood = 0.0
            
            for k in range(self.num_classes):
                # P(true class is k)
                class_prob = self.class_priors[k]
                
                # P(observations | true class is k)
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
        """
        Get the predicted class for each item.
        
        Args:
            return_probabilities: If True, return probability distributions
            
        Returns:
            Dictionary mapping item_id -> predicted_class (or probability distribution)
        """
        if self.item_classes is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        predictions = {}
        
        for i, item_id in enumerate(self.items):
            if return_probabilities:
                # Return full probability distribution
                prob_dict = {
                    self.class_labels[k]: float(self.item_classes[i, k])
                    for k in range(self.num_classes)
                }
                predictions[item_id] = prob_dict
            else:
                # Return most likely class (as string, not dict)
                predicted_class_idx = np.argmax(self.item_classes[i, :])
                predictions[item_id] = self.class_labels[predicted_class_idx]
        
        return predictions
    
    def get_annotator_accuracy(self) -> Dict[str, float]:
        """
        Compute the accuracy of each annotator.
        
        Accuracy = average probability of correct classification
        = average of diagonal elements of error matrix
        
        Returns:
            Dictionary mapping annotator_name -> accuracy
        """
        accuracies = {}
        
        for j, annotator_name in enumerate(self.annotator_names):
            # Compute weighted average of diagonal elements
            # Weight by class priors
            accuracy = 0.0
            for k in range(self.num_classes):
                accuracy += self.class_priors[k] * self.error_matrices[j, k, k]
            
            accuracies[annotator_name] = float(accuracy)
        
        return accuracies
    
    def get_confusion_matrix(self, annotator_name: str) -> np.ndarray:
        """
        Get the confusion matrix for a specific annotator.
        
        Args:
            annotator_name: Name of the annotator
            
        Returns:
            Confusion matrix where element [k, l] is P(annotator gives label l | true label is k)
        """
        if annotator_name not in self.annotator_names:
            raise ValueError(f"Unknown annotator: {annotator_name}")
        
        annotator_idx = self.annotator_names.index(annotator_name)
        print(self.error_matrices[annotator_idx, :, :])
        return self.error_matrices[annotator_idx, :, :]
    
    def save_predictions(self, output_file: str, include_probabilities: bool = True):
        """
        Save predictions to a CSV file.
        
        Args:
            output_file: Path to output CSV file
            include_probabilities: Whether to include class probabilities
        """
        predictions = self.predict(return_probabilities=include_probabilities)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if include_probabilities:
                # Write header
                fieldnames = ['video_name', 'predicted_class', 'confidence'] + \
                            [f'prob_{c}' for c in self.class_labels]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write predictions
                for item_id in self.items:
                    prob_dist = predictions[item_id]
                    predicted_class = max(prob_dist, key=prob_dist.get)
                    confidence = prob_dist[predicted_class]
                    
                    row = {
                        'video_name': item_id,
                        'predicted_class': predicted_class,
                        'confidence': f"{confidence:.4f}"
                    }
                    
                    # Add probability for each class
                    for c in self.class_labels:
                        row[f'prob_{c}'] = f"{prob_dist[c]:.4f}"
                    
                    writer.writerow(row)
            else:
                # Simple format
                fieldnames = ['video_name', 'predicted_class']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for item_id, predicted_class in predictions.items():
                    writer.writerow({
                        'video_name': item_id,
                        'predicted_class': predicted_class
                    })


def aggregate_annotations(
    annotations: Dict[str, Dict[str, str]],
    classes: List[str],
    annotator_names: List[str],
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[Dict[str, str], Dict[str, float], DawidSkene]:
    """
    Convenience function to aggregate annotations using Dawid-Skene.
    
    Args:
        annotations: Dict mapping item_id -> {annotator_name: label}
        classes: List of all possible class labels
        annotator_names: List of annotator names
        max_iterations: Maximum EM iterations
        tolerance: Convergence threshold
        
    Returns:
        Tuple of:
        - predictions: Dict mapping item_id -> predicted_class
        - annotator_accuracies: Dict mapping annotator_name -> accuracy
        - model: Fitted DawidSkene model (for further analysis)
    """
    model = DawidSkene(max_iterations=max_iterations, tolerance=tolerance)
    model.fit(annotations, classes, annotator_names)
    
    predictions = model.predict(return_probabilities=False)
    accuracies = model.get_annotator_accuracy()
    
    return predictions, accuracies, model
