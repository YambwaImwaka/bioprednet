"""
Metrics Module

Provides comprehensive metrics tracking for BioPredNet training,
including biological metrics (sparsity, prediction errors) and
performance metrics (accuracy, loss).
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prediction_errors: Optional[List[torch.Tensor]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for a batch.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        prediction_errors: List of prediction errors from each layer
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Classification metrics
    if targets.dim() == 1:
        pred_classes = predictions.argmax(dim=1)
        target_classes = targets
    else:
        pred_classes = predictions.argmax(dim=1)
        target_classes = targets.argmax(dim=1)
    
    # Accuracy
    accuracy = (pred_classes == target_classes).float().mean().item()
    metrics['accuracy'] = accuracy
    
    # Per-class accuracy (if applicable)
    num_classes = predictions.shape[1]
    for c in range(num_classes):
        mask = target_classes == c
        if mask.sum() > 0:
            class_acc = (pred_classes[mask] == target_classes[mask]).float().mean().item()
            metrics[f'class_{c}_accuracy'] = class_acc
    
    # Prediction error metrics
    if prediction_errors is not None:
        total_error = sum(e.abs().mean().item() for e in prediction_errors)
        metrics['total_prediction_error'] = total_error
        metrics['avg_prediction_error'] = total_error / len(prediction_errors)
        
        for i, error in enumerate(prediction_errors):
            metrics[f'layer_{i}_prediction_error'] = error.abs().mean().item()
    
    # Confidence metrics
    probs = torch.softmax(predictions, dim=1)
    max_probs, _ = probs.max(dim=1)
    metrics['avg_confidence'] = max_probs.mean().item()
    metrics['min_confidence'] = max_probs.min().item()
    metrics['max_confidence'] = max_probs.max().item()
    
    return metrics


class MetricsTracker:
    """
    Tracks and aggregates metrics over epochs and batches.
    
    Provides utilities for computing running averages, storing history,
    and generating training reports.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.epoch_metrics = defaultdict(list)
        self.batch_metrics = defaultdict(list)
        self.current_epoch = 0
        
    def update_batch(self, metrics: Dict[str, float]):
        """
        Update with batch metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            self.batch_metrics[key].append(value)
    
    def end_epoch(self):
        """
        Compute epoch statistics from batch metrics and reset.
        """
        # Aggregate batch metrics into epoch metrics
        for key, values in self.batch_metrics.items():
            if len(values) > 0:
                epoch_value = np.mean(values)
                self.epoch_metrics[key].append(epoch_value)
        
        # Reset batch metrics
        self.batch_metrics = defaultdict(list)
        self.current_epoch += 1
    
    def get_epoch_metrics(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Get metrics for a specific epoch.
        
        Args:
            epoch: Epoch index (None for latest)
            
        Returns:
            Dictionary of metrics
        """
        if epoch is None:
            epoch = self.current_epoch - 1
        
        metrics = {}
        for key, values in self.epoch_metrics.items():
            if epoch < len(values):
                metrics[key] = values[epoch]
        
        return metrics
    
    def get_latest_batch_metrics(self, window: int = 10) -> Dict[str, float]:
        """
        Get average metrics over recent batches.
        
        Args:
            window: Number of recent batches to average
            
        Returns:
            Dictionary of averaged metrics
        """
        metrics = {}
        for key, values in self.batch_metrics.items():
            if len(values) > 0:
                recent = values[-window:]
                metrics[key] = np.mean(recent)
        
        return metrics
    
    def get_history(self, metric_name: str) -> List[float]:
        """
        Get full history of a metric across epochs.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            List of values
        """
        return self.epoch_metrics.get(metric_name, [])
    
    def get_best_epoch(self, metric_name: str, mode: str = 'max') -> int:
        """
        Find epoch with best value for a metric.
        
        Args:
            metric_name: Name of metric
            mode: 'max' or 'min'
            
        Returns:
            Epoch index with best value
        """
        history = self.get_history(metric_name)
        if len(history) == 0:
            return -1
        
        if mode == 'max':
            return int(np.argmax(history))
        else:
            return int(np.argmin(history))
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of training.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_epochs': self.current_epoch,
            'metrics': {}
        }
        
        for key, values in self.epoch_metrics.items():
            if len(values) > 0:
                summary['metrics'][key] = {
                    'latest': values[-1],
                    'best': max(values) if 'accuracy' in key or 'confidence' in key else min(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return summary
    
    def print_summary(self):
        """Print training summary."""
        summary = self.get_summary()
        
        print(f"\n{'='*60}")
        print(f"Training Summary ({summary['total_epochs']} epochs)")
        print(f"{'='*60}")
        
        for metric_name, stats in summary['metrics'].items():
            print(f"\n{metric_name}:")
            print(f"  Latest: {stats['latest']:.4f}")
            print(f"  Best:   {stats['best']:.4f}")
            print(f"  Mean:   {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print(f"{'='*60}\n")
    
    def reset(self):
        """Reset all metrics."""
        self.epoch_metrics = defaultdict(list)
        self.batch_metrics = defaultdict(list)
        self.current_epoch = 0


class BiologicalMetrics:
    """
    Specialized metrics for biological plausibility analysis.
    
    Tracks sparsity, activity patterns, weight statistics, and other
    biological properties of the network.
    """
    
    @staticmethod
    def compute_sparsity(activations: torch.Tensor, threshold: float = 1e-6) -> float:
        """Compute fraction of inactive neurons."""
        active = (activations.abs() > threshold).float()
        sparsity = 1.0 - active.mean().item()
        return sparsity
    
    @staticmethod
    def compute_lifetime_sparsity(activations: torch.Tensor) -> float:
        """
        Compute lifetime sparsity (averaged over time/batches).
        
        Args:
            activations: Tensor of shape (num_samples, num_neurons)
        """
        avg_activity = (activations > 0).float().mean(dim=0)
        lifetime_sparsity = 1.0 - avg_activity.mean().item()
        return lifetime_sparsity
    
    @staticmethod
    def compute_population_sparsity(activations: torch.Tensor) -> float:
        """
        Compute population sparsity (averaged over neurons).
        
        Args:
            activations: Tensor of shape (num_samples, num_neurons)
        """
        avg_activity = (activations > 0).float().mean(dim=1)
        population_sparsity = 1.0 - avg_activity.mean().item()
        return population_sparsity
    
    @staticmethod
    def compute_neuron_utilization(activations: torch.Tensor) -> Dict[str, float]:
        """
        Compute what fraction of neurons are used at least once.
        
        Args:
            activations: Tensor of shape (num_samples, num_neurons)
        """
        ever_active = (activations > 0).any(dim=0).float()
        utilization = ever_active.mean().item()
        
        return {
            'utilization': utilization,
            'dead_neurons': 1.0 - utilization,
            'num_active_neurons': int(ever_active.sum().item()),
            'total_neurons': activations.shape[1]
        }
    
    @staticmethod
    def compute_weight_statistics(weights: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of weight matrix."""
        return {
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'max': weights.max().item(),
            'min': weights.min().item(),
            'norm': weights.norm().item(),
            'sparsity': (weights.abs() < 1e-6).float().mean().item()
        }
    
    @staticmethod
    def compute_energy_efficiency(activations: torch.Tensor, flops_per_neuron: float = 1.0) -> Dict[str, float]:
        """
        Estimate energy efficiency based on sparse activation.
        
        Args:
            activations: Activation tensor
            flops_per_neuron: Computational cost per active neuron
        """
        active_neurons = (activations > 0).float().sum(dim=1).mean().item()
        total_neurons = activations.shape[1]
        
        sparse_flops = active_neurons * flops_per_neuron
        dense_flops = total_neurons * flops_per_neuron
        
        efficiency = 1.0 - (sparse_flops / dense_flops)
        
        return {
            'avg_active_neurons': active_neurons,
            'total_neurons': total_neurons,
            'sparse_flops': sparse_flops,
            'dense_flops': dense_flops,
            'efficiency_gain': efficiency,
            'speedup': dense_flops / sparse_flops if sparse_flops > 0 else 0
        }
