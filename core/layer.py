"""
BioPredNet Layer Implementation

Implements a single layer of the BioPredNet architecture with:
- Dual weight matrices (forward encoding + backward prediction)
- Sparse activation via top-k selection
- Local learning rules (Hebbian + prediction error)
- Homeostatic plasticity for self-regulation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np

from .utils import sparse_topk, initialize_weights, ActivityTracker


class BioPredNetLayer(nn.Module):
    """
    A single BioPredNet layer implementing biologically-inspired learning.
    
    This layer learns through local rules only, without requiring backpropagation.
    It maintains two weight matrices:
    - W_forward: Encodes inputs into sparse representations (Hebbian learning)
    - W_backward: Predicts inputs from representations (minimizes prediction error)
    
    Key biological principles:
    1. Sparse activation (10-15% of neurons active)
    2. Local learning (no global error signals)
    3. Predictive coding (minimize prediction error)
    4. Homeostatic plasticity (stable firing rates)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sparsity: float = 0.15,
        lr_forward: float = 0.001,
        lr_backward: float = 0.001,
        lr_homeostatic: float = 0.0001,
        target_activity: Optional[float] = None,
        device: str = 'cpu',
        init_method: str = 'xavier'
    ):
        """
        Initialize BioPredNet layer.
        
        Args:
            input_dim: Dimension of input
            output_dim: Dimension of output (number of neurons in this layer)
            sparsity: Fraction of neurons to keep active (0-1)
            lr_forward: Learning rate for forward (encoding) weights
            lr_backward: Learning rate for backward (prediction) weights
            lr_homeostatic: Learning rate for homeostatic bias adjustment
            target_activity: Target average activity level (defaults to sparsity)
            device: Device to run on ('cpu' or 'cuda')
            init_method: Weight initialization method
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.lr_forward = lr_forward
        self.lr_backward = lr_backward
        self.lr_homeostatic = lr_homeostatic
        self.target_activity = target_activity if target_activity is not None else sparsity
        self.device = device
        
        # Forward weights: encode input -> sparse representation
        # Shape: (output_dim, input_dim)
        self.W_forward = nn.Parameter(
            initialize_weights(input_dim, output_dim, init_method).to(device)
        )
        
        # Backward weights: predict input from representation
        # Shape: (input_dim, output_dim)
        self.W_backward = nn.Parameter(
            initialize_weights(output_dim, input_dim, init_method).to(device)
        )
        
        # Bias for encoding (homeostatic regulation)
        self.bias = nn.Parameter(torch.zeros(output_dim, device=device))
        
        # Activity tracker for homeostatic plasticity
        self.activity_tracker = ActivityTracker(output_dim, momentum=0.9, device=device)
        
        # Storage for learning (activations, inputs, errors)
        self.last_input = None
        self.last_output = None
        self.last_sparse_indices = None
        self.prediction_error = None
        
    def forward(self, x: torch.Tensor, return_sparse_indices: bool = False) -> torch.Tensor:
        """
        Forward pass with sparse activation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_sparse_indices: Whether to return sparse indices
            
        Returns:
            Sparse activations of shape (batch_size, output_dim)
        """
        # Store input for learning
        self.last_input = x.detach()
        
        # Compute pre-activations: z = W_forward @ x + b
        z = torch.matmul(x, self.W_forward.T) + self.bias
        
        # Apply sparse top-k selection (lateral inhibition)
        h_sparse, sparse_indices = sparse_topk(z, self.sparsity, dim=-1)
        
        # Apply ReLU activation on sparse activations
        h = torch.relu(h_sparse)
        
        # Store for learning
        self.last_output = h.detach()
        self.last_sparse_indices = sparse_indices
        
        # Update activity tracker
        self.activity_tracker.update(h)
        
        if return_sparse_indices:
            return h, sparse_indices
        return h
    
    def predict_input(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict what the input should be based on current activations.
        
        This implements the backward (generative) model.
        
        Args:
            h: Current layer activations of shape (batch_size, output_dim)
            
        Returns:
            Predicted input of shape (batch_size, input_dim)
        """
        # x̂ = W_backward @ h
        x_pred = torch.matmul(h, self.W_backward.T)
        return x_pred
    
    def compute_prediction_error(self, x_actual: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction error between actual and predicted input.
        
        Args:
            x_actual: Actual input
            h: Current activations
            
        Returns:
            Prediction error ε = x_actual - x_predicted
        """
        x_pred = self.predict_input(h)
        error = x_actual - x_pred
        
        # Store for analysis
        self.prediction_error = error.detach()
        
        return error
    
    def update_weights(
        self,
        x: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        error: Optional[torch.Tensor] = None
    ):
        """
        Update weights using local learning rules.
        
        Implements three simultaneous updates:
        1. Forward weights (Hebbian): W_forward += lr * h^T @ x
        2. Backward weights (error correction): W_backward += lr * error^T @ h
        3. Homeostatic bias: b += lr * (target_activity - current_activity)
        
        Args:
            x: Input tensor (uses last_input if None)
            h: Output activations (uses last_output if None)
            error: Prediction error (computes if None)
        """
        # Use stored values if not provided
        if x is None:
            x = self.last_input
        if h is None:
            h = self.last_output
        if error is None and x is not None and h is not None:
            error = self.compute_prediction_error(x, h)
        
        if x is None or h is None or error is None:
            raise ValueError("Cannot update weights: missing required tensors")
        
        batch_size = x.shape[0]
        
        with torch.no_grad():
            # 1. Update forward weights (Hebbian encoding)
            # ΔW_forward = (1/batch_size) * h^T @ x
            # This strengthens connections between co-active input-output pairs
            delta_W_forward = torch.matmul(h.T, x) / batch_size
            self.W_forward += self.lr_forward * delta_W_forward
            
            # 2. Update backward weights (prediction error minimization)
            # ΔW_backward = (1/batch_size) * error^T @ h
            # This improves the ability to reconstruct inputs
            delta_W_backward = torch.matmul(error.T, h) / batch_size
            self.W_backward += self.lr_backward * delta_W_backward
            
            # 3. Homeostatic bias adjustment
            # Maintains stable activity levels across neurons
            current_activity = self.activity_tracker.get_activity()
            bias_adjustment = self.target_activity - current_activity
            self.bias += self.lr_homeostatic * bias_adjustment
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive layer statistics for monitoring.
        
        Returns:
            Dictionary containing activity, sparsity, and error statistics
        """
        stats = {}
        
        if self.last_output is not None:
            stats['output_mean'] = self.last_output.mean().item()
            stats['output_std'] = self.last_output.std().item()
            stats['active_fraction'] = (self.last_output > 0).float().mean().item()
            stats['actual_sparsity'] = 1.0 - stats['active_fraction']
        
        if self.prediction_error is not None:
            stats['prediction_error_mean'] = self.prediction_error.abs().mean().item()
            stats['prediction_error_std'] = self.prediction_error.std().item()
        
        stats['avg_activity'] = self.activity_tracker.get_activity().mean().item()
        stats['bias_mean'] = self.bias.mean().item()
        stats['bias_std'] = self.bias.std().item()
        
        # Weight statistics
        stats['W_forward_norm'] = self.W_forward.norm().item()
        stats['W_backward_norm'] = self.W_backward.norm().item()
        
        return stats
    
    def reset_activity(self):
        """Reset activity tracker (useful between epochs)."""
        self.activity_tracker.reset()
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
                f'sparsity={self.sparsity:.2f}, target_activity={self.target_activity:.2f}')
