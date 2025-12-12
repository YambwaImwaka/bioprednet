"""
BioPredNet Network Implementation

Implements a multi-layer hierarchical network using BioPredNet layers.
Orchestrates predictive coding across layers with hierarchical credit assignment.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
import numpy as np

from .layer import BioPredNetLayer


class BioPredNetNetwork(nn.Module):
    """
    Multi-layer BioPredNet network with hierarchical predictive coding.
    
    The network consists of multiple BioPredNet layers organized hierarchically:
    - Lower layers learn basic features
    - Higher layers learn abstract representations
    - Each layer independently minimizes its prediction error
    
    Key features:
    - Layer-parallel learning (can update all layers simultaneously)
    - Hierarchical credit assignment (each layer has local objective)
    - No backpropagation required
    - Sparse activation throughout
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        sparsity: float = 0.15,
        lr_forward: float = 0.001,
        lr_backward: float = 0.001,
        lr_homeostatic: float = 0.0001,
        device: str = 'cpu',
        init_method: str = 'xavier'
    ):
        """
        Initialize BioPredNet network.
        
        Args:
            layer_sizes: List of layer dimensions, e.g., [784, 512, 256, 10]
            sparsity: Target sparsity level for all layers
            lr_forward: Learning rate for forward weights
            lr_backward: Learning rate for backward weights
            lr_homeostatic: Learning rate for homeostatic regulation
            device: Device to run on
            init_method: Weight initialization method
        """
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.sparsity = sparsity
        self.device = device
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BioPredNetLayer(
                input_dim=layer_sizes[i],
                output_dim=layer_sizes[i + 1],
                sparsity=sparsity,
                lr_forward=lr_forward,
                lr_backward=lr_backward,
                lr_homeostatic=lr_homeostatic,
                device=device,
                init_method=init_method
            )
            self.layers.append(layer)
        
        # Storage for activations and errors
        self.activations = []
        self.prediction_errors = []
        
    def forward(self, x: torch.Tensor, return_all_activations: bool = False) -> torch.Tensor:
        """
        Forward pass through all layers with sparse activation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_all_activations: Whether to return activations from all layers
            
        Returns:
            Output activations from final layer, or list of all activations
        """
        # Clear previous activations
        self.activations = []
        
        # Store input as first activation
        h = x
        self.activations.append(h)
        
        # Forward pass through all layers
        for layer in self.layers:
            h = layer(h)
            self.activations.append(h)
        
        if return_all_activations:
            return self.activations
        return h
    
    def compute_all_prediction_errors(
        self,
        target: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Compute prediction errors for all layers.
        
        For hidden layers: ε_l = h_l - ĥ_l where ĥ_l = W_backward @ h_{l+1}
        For output layer: ε_out = target - h_out (supervised error)
        
        Args:
            target: Target output for supervised learning (required for output layer)
            
        Returns:
            List of prediction errors for each layer
        """
        self.prediction_errors = []
        
        # Compute errors for hidden layers (bottom-up)
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            h_current = self.activations[i + 1]  # Current layer activation
            
            # Predict current activation from next layer
            h_next = self.activations[i + 2]  # Next layer activation
            h_pred = torch.matmul(h_next, layer.W_backward.T)
            
            # Prediction error
            error = h_current - h_pred
            self.prediction_errors.append(error)
        
        # Output layer error (supervised)
        if target is not None:
            output_error = target - self.activations[-1]
            self.prediction_errors.append(output_error)
        else:
            # If no target, use zero error
            self.prediction_errors.append(torch.zeros_like(self.activations[-1]))
        
        return self.prediction_errors
    
    def update_all_weights(
        self,
        target: Optional[torch.Tensor] = None,
        parallel: bool = False
    ):
        """
        Update weights for all layers using local learning rules.
        
        This is where the magic happens - each layer updates independently
        based on its local prediction error, without needing global gradients.
        
        Args:
            target: Target output for supervised learning
            parallel: If True, updates can be done in parallel (future optimization)
        """
        # Ensure we have prediction errors
        if len(self.prediction_errors) == 0:
            self.compute_all_prediction_errors(target)
        
        # Update each layer
        for i, layer in enumerate(self.layers):
            x = self.activations[i]  # Input to this layer
            h = self.activations[i + 1]  # Output of this layer
            
            # For hidden layers, use prediction error
            # For output layer, use supervised error
            if i < len(self.layers) - 1:
                # Hidden layer: error is discrepancy between actual and predicted
                error = self.prediction_errors[i]
            else:
                # Output layer: use supervised error
                error = self.prediction_errors[i]
            
            # Update weights using local rule
            layer.update_weights(x, h, error)
    
    def train_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train on a single batch using BioPredNet algorithm.
        
        Algorithm:
        1. Forward pass with sparse activation
        2. Compute prediction errors for all layers
        3. Update weights using local learning rules
        
        Args:
            x: Input batch of shape (batch_size, input_dim)
            y: Target labels of shape (batch_size, num_classes)
            
        Returns:
            Dictionary of training statistics
        """
        # 1. Forward pass
        output = self.forward(x)
        
        # 2. Compute prediction errors
        self.compute_all_prediction_errors(y)
        
        # 3. Update all weights
        self.update_all_weights(target=y)
        
        # Compute statistics
        stats = self.get_training_statistics(y)
        
        return stats
    
    def get_training_statistics(self, target: torch.Tensor) -> Dict[str, float]:
        """
        Collect comprehensive training statistics.
        
        Args:
            target: Target tensor for accuracy computation
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Overall network statistics
        if len(self.activations) > 0:
            output = self.activations[-1]
            
            # Classification accuracy
            if target.dim() == 1:
                # Target is class indices
                predictions = output.argmax(dim=1)
                accuracy = (predictions == target).float().mean().item()
            else:
                # Target is one-hot or probabilities
                predictions = output.argmax(dim=1)
                target_classes = target.argmax(dim=1)
                accuracy = (predictions == target_classes).float().mean().item()
            
            stats['accuracy'] = accuracy
        
        # Prediction errors
        if len(self.prediction_errors) > 0:
            total_error = sum(e.abs().mean().item() for e in self.prediction_errors)
            stats['total_prediction_error'] = total_error
            stats['avg_prediction_error'] = total_error / len(self.prediction_errors)
            
            # Per-layer errors
            for i, error in enumerate(self.prediction_errors):
                stats[f'layer_{i}_error'] = error.abs().mean().item()
        
        # Per-layer statistics
        for i, layer in enumerate(self.layers):
            layer_stats = layer.get_statistics()
            for key, value in layer_stats.items():
                stats[f'layer_{i}_{key}'] = value
        
        return stats
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Output predictions
        """
        with torch.no_grad():
            output = self.forward(x)
        return output
    
    def reset_activities(self):
        """Reset activity trackers for all layers."""
        for layer in self.layers:
            layer.reset_activity()
    
    def get_layer_representations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get sparse representations from all layers for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            List of activations from each layer
        """
        with torch.no_grad():
            activations = self.forward(x, return_all_activations=True)
        return activations
    
    def get_reconstruction(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Reconstruct input from a specific layer's representation.
        
        Useful for visualizing what a layer has learned.
        
        Args:
            x: Input tensor
            layer_idx: Which layer to reconstruct from
            
        Returns:
            Reconstructed input
        """
        with torch.no_grad():
            # Forward pass to get activations
            activations = self.forward(x, return_all_activations=True)
            
            # Get representation at specified layer
            h = activations[layer_idx + 1]
            
            # Reconstruct using backward weights
            layer = self.layers[layer_idx]
            reconstruction = layer.predict_input(h)
        
        return reconstruction
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (f'layer_sizes={self.layer_sizes}, sparsity={self.sparsity:.2f}, '
                f'num_layers={self.num_layers}')
