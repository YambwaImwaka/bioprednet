"""
BioPredNet Core Module

This module implements the core architecture of BioPredNet, a biologically-inspired
neural network training algorithm based on predictive coding, sparse activation,
and local learning rules.
"""

from .layer import BioPredNetLayer
from .network import BioPredNetNetwork
from .utils import (
    sparse_topk,
    initialize_weights,
    compute_sparsity,
    track_activity
)

__all__ = [
    'BioPredNetLayer',
    'BioPredNetNetwork',
    'sparse_topk',
    'initialize_weights',
    'compute_sparsity',
    'track_activity'
]
