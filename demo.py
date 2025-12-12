"""
Quick Demo Script

Demonstrates BioPredNet on a small MNIST subset for quick testing.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.network import BioPredNetNetwork
from training.trainer import BioPredNetTrainer

def quick_demo():
    """Run a quick demo on a small subset of MNIST."""
    
    print("="*60)
    print(" BioPredNet Quick Demo")
    print("="*60)
    print("\nThis demo trains on a small MNIST subset (2000 samples)")
    print("for 10 epochs to demonstrate the algorithm.\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load MNIST (small subset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading MNIST...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Use small subset for quick demo
    train_subset = Subset(train_dataset, range(2000))
    test_subset = Subset(test_dataset, range(500))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # Create model
    print("Creating BioPredNet...")
    model = BioPredNetNetwork(
        layer_sizes=[784, 256, 128, 10],
        sparsity=0.15,
        lr_forward=0.01,
        lr_backward=0.01,
        lr_homeostatic=0.001,
        device=device
    )
    
    print(f"Architecture: [784, 256, 128, 10]")
    print(f"Sparsity: 15%")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    trainer = BioPredNetTrainer(model, device=device)
    
    print("Training for 10 epochs...\n")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=None,
        num_epochs=10,
        num_classes=10,
        verbose=True
    )
    
    # Test
    print("\nEvaluating on test set...")
    test_metrics = trainer.validate(test_loader, num_classes=10, verbose=False)
    
    print("\n" + "="*60)
    print(" Results")
    print("="*60)
    print(f"Final Training Accuracy: {history['train_metrics']['accuracy'][-1]:.2%}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Average Prediction Error: {test_metrics['avg_prediction_error']:.4f}")
    print("="*60)
    
    print("\n✅ Demo complete! For full experiments, run:")
    print("   python experiments/mnist.py")
    print("\n")

if __name__ == '__main__':
    quick_demo()
