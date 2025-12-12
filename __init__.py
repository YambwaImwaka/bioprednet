"""
BioPredNet: Biologically-Inspired Predictive Neural Networks

A novel neural network training algorithm that eliminates backpropagation
by implementing biological learning principles.
"""

__version__ = '1.0.0'
__author__ = 'BioPredNet Team'
__license__ = 'MIT'

from core.network import BioPredNetNetwork
from core.layer import BioPredNetLayer
from training.trainer import BioPredNetTrainer
from training.metrics import MetricsTracker, BiologicalMetrics

__all__ = [
    'BioPredNetNetwork',
    'BioPredNetLayer',
    'BioPredNetTrainer',
    'MetricsTracker',
    'BiologicalMetrics'
]
