"""
BioPredNet Trainer

Main training class that orchestrates the BioPredNet training algorithm,
including data handling, optimization, checkpointing, and monitoring.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
from tqdm import tqdm
import os
import json

from core.network import BioPredNetNetwork
from .metrics import MetricsTracker, compute_metrics, BiologicalMetrics


class BioPredNetTrainer:
    """
    Trainer for BioPredNet networks.
    
    Handles:
    - Training loop with BioPredNet algorithm
    - Validation and testing
    - Metrics tracking and logging
    - Model checkpointing
    - Early stopping
    """
    
    def __init__(
        self,
        model: BioPredNetNetwork,
        device: str = 'cpu',
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: BioPredNetNetwork to train
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
        # Create checkpoint directory
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def prepare_target(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Convert labels to target format for BioPredNet.
        
        For classification, we use one-hot encoding as the target.
        
        Args:
            labels: Class labels (batch_size,)
            num_classes: Number of classes
            
        Returns:
            One-hot encoded targets (batch_size, num_classes)
        """
        if labels.dim() == 1:
            # Convert to one-hot
            targets = torch.zeros(labels.shape[0], num_classes, device=self.device)
            targets.scatter_(1, labels.unsqueeze(1), 1.0)
        else:
            # Already in correct format
            targets = labels.to(self.device)
        
        return targets
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        num_classes: int,
        epoch: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            num_classes: Number of output classes
            epoch: Current epoch number
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        
        # Progress bar
        if verbose:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        else:
            pbar = train_loader
        
        for batch_idx, (data, labels) in enumerate(pbar):
            # Move to device
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Flatten images if needed
            if data.dim() > 2:
                data = data.view(data.shape[0], -1)
            
            # Prepare targets
            targets = self.prepare_target(labels, num_classes)
            
            # Train on batch
            batch_stats = self.model.train_batch(data, targets)
            
            # Update metrics
            self.train_metrics.update_batch(batch_stats)
            
            # Update progress bar
            if verbose:
                recent_metrics = self.train_metrics.get_latest_batch_metrics(window=10)
                pbar.set_postfix({
                    'acc': f"{recent_metrics.get('accuracy', 0):.3f}",
                    'err': f"{recent_metrics.get('avg_prediction_error', 0):.3f}"
                })
        
        # Compute epoch metrics
        self.train_metrics.end_epoch()
        epoch_metrics = self.train_metrics.get_epoch_metrics()
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        num_classes: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            num_classes: Number of output classes
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_errors = []
        
        if verbose:
            pbar = tqdm(val_loader, desc='Validation')
        else:
            pbar = val_loader
        
        for data, labels in pbar:
            # Move to device
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Flatten if needed
            if data.dim() > 2:
                data = data.view(data.shape[0], -1)
            
            # Forward pass
            output = self.model.forward(data)
            
            # Prepare targets
            targets = self.prepare_target(labels, num_classes)
            
            # Compute prediction errors
            errors = self.model.compute_all_prediction_errors(targets)
            
            # Store for metrics
            all_predictions.append(output)
            all_targets.append(targets)
            all_errors.append([e.clone() for e in errors])
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)
        
        # Add average prediction error
        avg_errors = []
        for layer_idx in range(len(all_errors[0])):
            layer_errors = [batch_errors[layer_idx] for batch_errors in all_errors]
            avg_error = torch.cat(layer_errors, dim=0).abs().mean().item()
            avg_errors.append(avg_error)
            metrics[f'layer_{layer_idx}_prediction_error'] = avg_error
        
        metrics['avg_prediction_error'] = sum(avg_errors) / len(avg_errors)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        num_classes: int = 10,
        early_stopping_patience: Optional[int] = None,
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            num_classes: Number of output classes
            early_stopping_patience: Stop if no improvement for N epochs
            save_best: Whether to save best model
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        if verbose:
            print(f"\nTraining BioPredNet on {self.device}")
            print(f"Model: {self.model.layer_sizes}")
            print(f"Epochs: {num_epochs}")
            print(f"Sparsity: {self.model.sparsity:.2f}")
            print("="*60)
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, num_classes, epoch, verbose
            )
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader, num_classes, verbose)
                self.val_metrics.update_batch(val_metrics)
                self.val_metrics.end_epoch()
                
                val_accuracy = val_metrics['accuracy']
                
                # Check for improvement
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    if save_best and self.checkpoint_dir:
                        self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                
                # Print epoch summary
                if verbose:
                    print(f"\nEpoch {epoch}/{num_epochs}")
                    print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, "
                          f"Error: {train_metrics['avg_prediction_error']:.4f}")
                    print(f"  Val   - Acc: {val_accuracy:.4f}, "
                          f"Error: {val_metrics['avg_prediction_error']:.4f}")
                    print(f"  Best Val Acc: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")
                
                # Early stopping
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch}")
                    break
            else:
                # No validation, just print training metrics
                if verbose:
                    print(f"\nEpoch {epoch}/{num_epochs}")
                    print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, "
                          f"Error: {train_metrics['avg_prediction_error']:.4f}")
        
        # Training complete
        if verbose:
            print("\n" + "="*60)
            print("Training Complete!")
            if val_loader is not None:
                print(f"Best Validation Accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")
            print("="*60 + "\n")
        
        # Return history
        history = {
            'train_metrics': self.train_metrics.epoch_metrics,
            'val_metrics': self.val_metrics.epoch_metrics if val_loader else {},
            'best_epoch': self.best_epoch,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        return history
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt'):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified")
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.current_epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'model_config': {
                'layer_sizes': self.model.layer_sizes,
                'sparsity': self.model.sparsity
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pt'):
        """
        Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified")
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.best_epoch = checkpoint['best_epoch']
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Epoch: {self.current_epoch}, Best Val Acc: {self.best_val_accuracy:.4f}")
    
    def save_history(self, filename: str = 'history.json'):
        """
        Save training history to JSON.
        
        Args:
            filename: History filename
        """
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified")
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        history = {
            'train_metrics': {k: [float(v) for v in vals] 
                            for k, vals in self.train_metrics.epoch_metrics.items()},
            'val_metrics': {k: [float(v) for v in vals] 
                          for k, vals in self.val_metrics.epoch_metrics.items()},
            'best_epoch': int(self.best_epoch),
            'best_val_accuracy': float(self.best_val_accuracy)
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"History saved: {filepath}")
    
    def analyze_biological_properties(
        self,
        data_loader: DataLoader,
        num_samples: int = 1000
    ) -> Dict[str, any]:
        """
        Analyze biological properties of the trained network.
        
        Args:
            data_loader: Data loader for analysis
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary of biological metrics
        """
        self.model.eval()
        
        # Collect activations from all layers
        all_layer_activations = [[] for _ in range(len(self.model.layers))]
        
        samples_collected = 0
        
        with torch.no_grad():
            for data, _ in data_loader:
                if samples_collected >= num_samples:
                    break
                
                data = data.to(self.device)
                if data.dim() > 2:
                    data = data.view(data.shape[0], -1)
                
                # Get activations from all layers
                activations = self.model.get_layer_representations(data)
                
                for i, act in enumerate(activations[1:]):  # Skip input
                    all_layer_activations[i].append(act)
                
                samples_collected += data.shape[0]
        
        # Concatenate activations
        layer_activations = [torch.cat(acts, dim=0) for acts in all_layer_activations]
        
        # Compute biological metrics
        bio_metrics = {
            'num_samples_analyzed': samples_collected
        }
        
        for i, activations in enumerate(layer_activations):
            layer_stats = {
                'sparsity': BiologicalMetrics.compute_sparsity(activations),
                'lifetime_sparsity': BiologicalMetrics.compute_lifetime_sparsity(activations),
                'population_sparsity': BiologicalMetrics.compute_population_sparsity(activations),
                'utilization': BiologicalMetrics.compute_neuron_utilization(activations),
                'efficiency': BiologicalMetrics.compute_energy_efficiency(activations)
            }
            
            bio_metrics[f'layer_{i}'] = layer_stats
        
        return bio_metrics
