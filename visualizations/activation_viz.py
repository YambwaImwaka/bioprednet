"""
Activation Visualization Tools

Functions for visualizing sparse activation patterns and neuron activity.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


def plot_sparse_activations(
    activations: torch.Tensor,
    title: str = "Sparse Activation Pattern",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
):
    """
    Visualize sparse activation pattern as a heatmap.
    
    Args:
        activations: Activation tensor of shape (batch_size, num_neurons)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    # Convert to numpy
    act_np = activations.detach().cpu().numpy()
    
    # Create binary activation map
    binary_act = (act_np > 0).astype(float)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Binary activation pattern
    im1 = axes[0].imshow(binary_act, aspect='auto', cmap='binary', interpolation='nearest')
    axes[0].set_title('Binary Activation (Active/Inactive)')
    axes[0].set_xlabel('Neuron Index')
    axes[0].set_ylabel('Sample Index')
    plt.colorbar(im1, ax=axes[0])
    
    # Activation values (for active neurons)
    im2 = axes[1].imshow(act_np, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title('Activation Values')
    axes[1].set_xlabel('Neuron Index')
    axes[1].set_ylabel('Sample Index')
    plt.colorbar(im2, ax=axes[1])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_neuron_activity(
    activations: torch.Tensor,
    title: str = "Neuron Activity Distribution",
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None
):
    """
    Plot neuron activity statistics.
    
    Args:
        activations: Activation tensor of shape (num_samples, num_neurons)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    act_np = activations.detach().cpu().numpy()
    
    # Compute statistics
    firing_rates = (act_np > 0).mean(axis=0)  # Fraction of time each neuron fires
    avg_activation = act_np.mean(axis=0)  # Average activation value
    max_activation = act_np.max(axis=0)  # Max activation
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Firing rate distribution
    axes[0, 0].hist(firing_rates, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(firing_rates.mean(), color='red', linestyle='--', 
                       label=f'Mean: {firing_rates.mean():.3f}')
    axes[0, 0].set_xlabel('Firing Rate')
    axes[0, 0].set_ylabel('Number of Neurons')
    axes[0, 0].set_title('Firing Rate Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Average activation distribution
    axes[0, 1].hist(avg_activation, bins=50, color='seagreen', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(avg_activation.mean(), color='red', linestyle='--',
                       label=f'Mean: {avg_activation.mean():.3f}')
    axes[0, 1].set_xlabel('Average Activation')
    axes[0, 1].set_ylabel('Number of Neurons')
    axes[0, 1].set_title('Average Activation Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Neuron usage (sorted firing rates)
    sorted_rates = np.sort(firing_rates)[::-1]
    axes[1, 0].plot(sorted_rates, linewidth=2, color='coral')
    axes[1, 0].axhline(firing_rates.mean(), color='red', linestyle='--', 
                      label=f'Mean: {firing_rates.mean():.3f}')
    axes[1, 0].set_xlabel('Neuron Rank')
    axes[1, 0].set_ylabel('Firing Rate')
    axes[1, 0].set_title('Neurons Sorted by Activity')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Sparsity over samples
    sparsity_per_sample = 1.0 - (act_np > 0).mean(axis=1)
    axes[1, 1].plot(sparsity_per_sample, linewidth=1, alpha=0.7, color='purple')
    axes[1, 1].axhline(sparsity_per_sample.mean(), color='red', linestyle='--',
                      label=f'Mean Sparsity: {sparsity_per_sample.mean():.3f}')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Sparsity')
    axes[1, 1].set_title('Sparsity Across Samples')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_activity_over_time(
    activity_history: List[float],
    target_activity: float,
    title: str = "Homeostatic Regulation",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot how neuron activity evolves over training (homeostatic regulation).
    
    Args:
        activity_history: List of activity values over time
        target_activity: Target activity level
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    plt.plot(activity_history, linewidth=2, label='Actual Activity', color='steelblue')
    plt.axhline(target_activity, color='red', linestyle='--', 
               linewidth=2, label=f'Target: {target_activity:.3f}')
    
    plt.xlabel('Batch/Iteration')
    plt.ylabel('Average Activity')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_layer_activations_comparison(
    layer_activations: List[torch.Tensor],
    layer_names: Optional[List[str]] = None,
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None
):
    """
    Compare activation patterns across multiple layers.
    
    Args:
        layer_activations: List of activation tensors, one per layer
        layer_names: Optional layer names
        figsize: Figure size
        save_path: Path to save figure
    """
    num_layers = len(layer_activations)
    
    if layer_names is None:
        layer_names = [f'Layer {i}' for i in range(num_layers)]
    
    fig, axes = plt.subplots(1, num_layers, figsize=figsize)
    
    if num_layers == 1:
        axes = [axes]
    
    for i, (act, name) in enumerate(zip(layer_activations, layer_names)):
        act_np = act.detach().cpu().numpy()
        binary_act = (act_np > 0).astype(float)
        
        # Show first N samples
        n_samples = min(100, act_np.shape[0])
        im = axes[i].imshow(binary_act[:n_samples], aspect='auto', 
                           cmap='binary', interpolation='nearest')
        
        # Compute sparsity
        sparsity = 1.0 - binary_act.mean()
        
        axes[i].set_title(f'{name}\nSparsity: {sparsity:.2%}')
        axes[i].set_xlabel('Neuron Index')
        
        if i == 0:
            axes[i].set_ylabel('Sample Index')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
