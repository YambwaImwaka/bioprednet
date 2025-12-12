"""
Training Visualization Tools

Functions for visualizing training progress, prediction errors, and convergence.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: List[str] = ['accuracy', 'avg_prediction_error'],
    title: str = "Training Progress",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train' and 'val' subdicts containing metric lists
        metrics: List of metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training curve
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], linewidth=2, 
                   label='Train', marker='o', markersize=4)
        
        # Plot validation curve if available
        val_key = f'val_{metric}' if not metric.startswith('val_') else metric
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], linewidth=2,
                   label='Validation', marker='s', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_errors(
    error_history: Dict[str, List[float]],
    title: str = "Prediction Errors by Layer",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot prediction errors for each layer over training.
    
    Args:
        error_history: Dict with keys like 'layer_0_prediction_error'
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    # Extract layer errors
    layer_errors = {}
    for key, values in error_history.items():
        if 'layer_' in key and 'prediction_error' in key:
            layer_errors[key] = values
    
    # Sort by layer number
    sorted_keys = sorted(layer_errors.keys(), 
                        key=lambda x: int(x.split('_')[1]))
    
    # Plot each layer
    for key in sorted_keys:
        layer_num = key.split('_')[1]
        epochs = range(1, len(layer_errors[key]) + 1)
        plt.plot(epochs, layer_errors[key], linewidth=2, 
                marker='o', markersize=3, label=f'Layer {layer_num}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Error')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sparsity_levels(
    sparsity_history: Dict[str, List[float]],
    target_sparsity: float,
    title: str = "Sparsity Levels by Layer",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot sparsity levels for each layer.
    
    Args:
        sparsity_history: Dict with keys like 'layer_0_actual_sparsity'
        target_sparsity: Target sparsity level
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    # Extract sparsity metrics
    layer_sparsity = {}
    for key, values in sparsity_history.items():
        if 'layer_' in key and 'sparsity' in key:
            layer_sparsity[key] = values
    
    # Sort by layer
    sorted_keys = sorted(layer_sparsity.keys(),
                        key=lambda x: int(x.split('_')[1]))
    
    # Plot each layer
    for key in sorted_keys:
        layer_num = key.split('_')[1]
        epochs = range(1, len(layer_sparsity[key]) + 1)
        plt.plot(epochs, layer_sparsity[key], linewidth=2,
                marker='o', markersize=3, label=f'Layer {layer_num}')
    
    # Target line
    plt.axhline(target_sparsity, color='red', linestyle='--',
               linewidth=2, label=f'Target: {target_sparsity:.2f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Sparsity')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_training_dashboard(
    history: Dict[str, List[float]],
    target_sparsity: float = 0.15,
    figsize: tuple = (18, 12),
    save_path: Optional[str] = None
):
    """
    Create comprehensive training dashboard.
    
    Args:
        history: Training history dictionary
        target_sparsity: Target sparsity level
        figsize: Figure size
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy curves
    ax1 = fig.add_subplot(gs[0, :2])
    if 'accuracy' in history:
        epochs = range(1, len(history['accuracy']) + 1)
        ax1.plot(epochs, history['accuracy'], linewidth=2, 
                marker='o', markersize=4, label='Train')
    if 'val_accuracy' in history:
        epochs = range(1, len(history['val_accuracy']) + 1)
        ax1.plot(epochs, history['val_accuracy'], linewidth=2,
                marker='s', markersize=4, label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Classification Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Overall prediction error
    ax2 = fig.add_subplot(gs[0, 2])
    if 'avg_prediction_error' in history:
        epochs = range(1, len(history['avg_prediction_error']) + 1)
        ax2.plot(epochs, history['avg_prediction_error'], 
                linewidth=2, color='coral', marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.set_title('Avg Prediction Error')
    ax2.grid(alpha=0.3)
    
    # 3. Per-layer prediction errors
    ax3 = fig.add_subplot(gs[1, :])
    layer_errors = {k: v for k, v in history.items() 
                   if 'layer_' in k and 'prediction_error' in k}
    sorted_keys = sorted(layer_errors.keys(), 
                        key=lambda x: int(x.split('_')[1]))
    for key in sorted_keys:
        layer_num = key.split('_')[1]
        epochs = range(1, len(layer_errors[key]) + 1)
        ax3.plot(epochs, layer_errors[key], linewidth=2,
                marker='o', markersize=3, label=f'Layer {layer_num}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Prediction Error')
    ax3.set_title('Prediction Errors by Layer')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Sparsity levels
    ax4 = fig.add_subplot(gs[2, :])
    layer_sparsity = {k: v for k, v in history.items()
                     if 'layer_' in k and 'actual_sparsity' in k}
    sorted_keys = sorted(layer_sparsity.keys(),
                        key=lambda x: int(x.split('_')[1]))
    for key in sorted_keys:
        layer_num = key.split('_')[1]
        epochs = range(1, len(layer_sparsity[key]) + 1)
        ax4.plot(epochs, layer_sparsity[key], linewidth=2,
                marker='o', markersize=3, label=f'Layer {layer_num}')
    ax4.axhline(target_sparsity, color='red', linestyle='--',
               linewidth=2, label=f'Target: {target_sparsity:.2f}')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Sparsity')
    ax4.set_title('Sparsity Levels by Layer')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_ylim([0, 1])
    
    fig.suptitle('BioPredNet Training Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comparison_bars(
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    title: str = "BioPredNet vs Baseline",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Compare BioPredNet with baseline (e.g., backprop).
    
    Args:
        metrics: BioPredNet metrics
        baseline_metrics: Baseline metrics
        title: Plot title
        figsize: Figure size  
        save_path: Path to save figure
    """
    # Get common metrics
    common_keys = set(metrics.keys()) & set(baseline_metrics.keys())
    
    metric_names = list(common_keys)
    biopred_values = [metrics[k] for k in metric_names]
    baseline_values = [baseline_metrics[k] for k in metric_names]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, biopred_values, width, 
                   label='BioPredNet', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, baseline_values, width,
                   label='Baseline', color='coral', alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace('_', ' ').title() for k in metric_names],
                       rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
