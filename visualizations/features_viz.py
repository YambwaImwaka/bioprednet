"""
Feature Visualization Tools

Functions for visualizing learned features (weights) from BioPredNet layers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_learned_features(
    weights: torch.Tensor,
    img_shape: Optional[Tuple[int, int]] = None,
    n_features: int = 64,
    title: str = "Learned Features",
    figsize: tuple = (15, 15),
    save_path: Optional[str] = None
):
    """
    Visualize learned features from weight matrix.
    
    Args:
        weights: Weight matrix of shape (n_neurons, input_dim)
        img_shape: Shape to reshape weights to (height, width) for visualization
        n_features: Number of features to display
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    weights_np = weights.detach().cpu().numpy()
    
    # Determine number of features to show
    n_features = min(n_features, weights_np.shape[0])
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(n_features)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(n_features):
        feature = weights_np[i]
        
        # Normalize for visualization
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
        
        if img_shape is not None:
            # Reshape to image
            feature = feature.reshape(img_shape)
        
        # Plot
        if img_shape is not None and len(img_shape) == 2:
            axes[i].imshow(feature, cmap='gray', interpolation='nearest')
        else:
            axes[i].plot(feature)
        
        axes[i].axis('off')
        axes[i].set_title(f'F{i}', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    n_samples: int = 10,
    img_shape: Optional[Tuple[int, int]] = (28, 28),
    title: str = "Original vs Reconstruction",
    figsize: tuple = (15, 6),
    save_path: Optional[str] = None
):
    """
    Compare original inputs with their reconstructions.
    
    Args:
        originals: Original inputs
        reconstructions: Reconstructed inputs
        n_samples: Number of samples to show
        img_shape: Shape to reshape to
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    orig_np = originals.detach().cpu().numpy()
    recon_np = reconstructions.detach().cpu().numpy()
    
    n_samples = min(n_samples, orig_np.shape[0])
    
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    for i in range(n_samples):
        # Original
        orig = orig_np[i]
        if img_shape is not None:
            orig = orig.reshape(img_shape)
        
        axes[0, i].imshow(orig, cmap='gray', interpolation='nearest')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        
        # Reconstruction
        recon = recon_np[i]
        if img_shape is not None:
            recon = recon.reshape(img_shape)
        
        axes[1, i].imshow(recon, cmap='gray', interpolation='nearest')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstruction', fontsize=12)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_weight_statistics(
    weights: torch.Tensor,
    title: str = "Weight Statistics",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None
):
    """
    Plot statistics of weight matrix.
    
    Args:
        weights: Weight matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    weights_np = weights.detach().cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Histogram
    axes[0].hist(weights_np, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Weight Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Weight Distribution')
    axes[0].grid(alpha=0.3)
    
    # Absolute values
    abs_weights = np.abs(weights_np)
    axes[1].hist(abs_weights, bins=100, color='seagreen', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('|Weight Value|')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Absolute Weight Distribution')
    axes[1].grid(alpha=0.3)
    
    # Summary statistics
    stats_text = f"""
    Mean: {weights_np.mean():.4f}
    Std:  {weights_np.std():.4f}
    Min:  {weights_np.min():.4f}
    Max:  {weights_np.max():.4f}
    
    L1 Norm: {np.abs(weights_np).sum():.2f}
    L2 Norm: {np.sqrt((weights_np**2).sum()):.2f}
    """
    
    axes[2].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center')
    axes[2].axis('off')
    axes[2].set_title('Statistics')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_similarity(
    features: torch.Tensor,
    title: str = "Feature Similarity Matrix",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot pairwise feature similarity.
    
    Args:
        features: Feature matrix of shape (n_features, feature_dim)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    # Normalize features
    features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity
    similarity = (features_norm @ features_norm.T).detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    im = plt.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar(im, label='Cosine Similarity')
    plt.title(title)
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
