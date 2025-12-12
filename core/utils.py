"""
Utility functions for BioPredNet

Provides helper functions for sparse operations, weight initialization,
and activity tracking.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def sparse_topk(x: torch.Tensor, k: float, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k% of values in tensor and return sparse mask.
    
    Implements lateral inhibition by keeping only the most activated neurons.
    
    Args:
        x: Input tensor of shape (..., features)
        k: Sparsity percentage (0-1), e.g., 0.15 for 15% sparsity
        dim: Dimension along which to apply top-k
        
    Returns:
        Tuple of (sparse_tensor, active_indices)
    """
    # Determine number of active neurons
    k_count = max(1, int(k * x.shape[dim]))
    
    # Get top-k values and indices
    topk_values, topk_indices = torch.topk(x, k_count, dim=dim)
    
    # Create sparse tensor
    sparse_tensor = torch.zeros_like(x)
    sparse_tensor.scatter_(dim, topk_indices, topk_values)
    
    return sparse_tensor, topk_indices


def initialize_weights(input_dim: int, output_dim: int, method: str = 'xavier') -> torch.Tensor:
    """
    Initialize weight matrix using specified method.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        method: Initialization method ('xavier', 'he', 'sparse')
        
    Returns:
        Initialized weight tensor of shape (output_dim, input_dim)
    """
    if method == 'xavier':
        # Xavier/Glorot initialization
        std = np.sqrt(2.0 / (input_dim + output_dim))
        weights = torch.randn(output_dim, input_dim) * std
    elif method == 'he':
        # He initialization (good for ReLU)
        std = np.sqrt(2.0 / input_dim)
        weights = torch.randn(output_dim, input_dim) * std
    elif method == 'sparse':
        # Sparse random initialization
        weights = torch.randn(output_dim, input_dim) * 0.01
        mask = torch.rand(output_dim, input_dim) > 0.9  # 10% connectivity
        weights = weights * mask.float()
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return weights


def compute_sparsity(activations: torch.Tensor, threshold: float = 1e-6) -> float:
    """
    Compute sparsity level of activations.
    
    Args:
        activations: Activation tensor
        threshold: Threshold below which values are considered zero
        
    Returns:
        Sparsity as fraction of inactive neurons (0-1)
    """
    active = (activations.abs() > threshold).float()
    sparsity = 1.0 - active.mean().item()
    return sparsity


class ActivityTracker:
    """
    Tracks neuron activity using exponential moving average.
    
    Used for homeostatic plasticity to maintain stable firing rates.
    """
    
    def __init__(self, num_neurons: int, momentum: float = 0.9, device: str = 'cpu'):
        """
        Initialize activity tracker.
        
        Args:
            num_neurons: Number of neurons to track
            momentum: Momentum for exponential moving average (0-1)
            device: Device to store tensors on
        """
        self.num_neurons = num_neurons
        self.momentum = momentum
        self.device = device
        
        # Initialize moving average of activity
        self.activity_avg = torch.zeros(num_neurons, device=device)
        self.num_updates = 0
        
    def update(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Update activity statistics with new batch.
        
        Args:
            activations: Activation tensor of shape (batch_size, num_neurons)
            
        Returns:
            Current average activity per neuron
        """
        # Compute binary activity (neuron fired or not)
        binary_activity = (activations > 0).float()
        
        # Batch average
        batch_avg = binary_activity.mean(dim=0)
        
        # Update moving average
        if self.num_updates == 0:
            self.activity_avg = batch_avg
        else:
            self.activity_avg = (self.momentum * self.activity_avg + 
                                (1 - self.momentum) * batch_avg)
        
        self.num_updates += 1
        
        return self.activity_avg
    
    def get_activity(self) -> torch.Tensor:
        """Return current average activity."""
        return self.activity_avg
    
    def reset(self):
        """Reset activity statistics."""
        self.activity_avg.zero_()
        self.num_updates = 0


def track_activity(activations: torch.Tensor) -> dict:
    """
    Compute comprehensive activity statistics.
    
    Args:
        activations: Activation tensor
        
    Returns:
        Dictionary of activity statistics
    """
    with torch.no_grad():
        stats = {
            'mean': activations.mean().item(),
            'std': activations.std().item(),
            'sparsity': compute_sparsity(activations),
            'max': activations.max().item(),
            'min': activations.min().item(),
            'active_fraction': (activations > 0).float().mean().item()
        }
    
    return stats


def cosine_similarity_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between feature vectors.
    
    Args:
        features: Feature tensor of shape (num_features, feature_dim)
        
    Returns:
        Similarity matrix of shape (num_features, num_features)
    """
    # Normalize features
    features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute similarity
    similarity = features_norm @ features_norm.T
    
    return similarity


def orthogonalize_weights(weights: torch.Tensor, method: str = 'qr') -> torch.Tensor:
    """
    Orthogonalize weight matrix to reduce redundancy.
    
    Args:
        weights: Weight matrix
        method: Orthogonalization method ('qr', 'svd')
        
    Returns:
        Orthogonalized weights
    """
    if method == 'qr':
        Q, R = torch.linalg.qr(weights)
        return Q
    elif method == 'svd':
        U, S, V = torch.svd(weights)
        return U @ V.T
    else:
        raise ValueError(f"Unknown orthogonalization method: {method}")
