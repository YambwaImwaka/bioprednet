"""
MNIST Experiment with BioPredNet

Train and evaluate BioPredNet on MNIST digit classification.
Expected accuracy: 97-98%
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import BioPredNetNetwork
from training.trainer import BioPredNetTrainer
from visualizations.training_viz import create_training_dashboard
from visualizations.features_viz import plot_learned_features, plot_reconstructions
from visualizations.activation_viz import plot_neuron_activity


def run_mnist_experiment(
    device='cpu',
    sparsity=0.15,
    epochs=50,
    batch_size=128,
    lr_forward=0.001,
    lr_backward=0.001,
    lr_homeostatic=0.0001,
    save_dir='./results/mnist'
):
    """
    Run BioPredNet on MNIST.
    
    Args:
        device: Device to train on
        sparsity: Target sparsity level
        epochs: Number of training epochs
        batch_size: Batch size
        lr_forward: Learning rate for forward weights
        lr_backward: Learning rate for backward weights
        lr_homeostatic: Learning rate for homeostatic regulation
        save_dir: Directory to save results
    """
    print("="*70)
    print(" BioPredNet MNIST Experiment")
    print("="*70)
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Split train into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create BioPredNet model
    print("\nCreating BioPredNet model...")
    layer_sizes = [784, 512, 256, 128, 10]
    
    model = BioPredNetNetwork(
        layer_sizes=layer_sizes,
        sparsity=sparsity,
        lr_forward=lr_forward,
        lr_backward=lr_backward,
        lr_homeostatic=lr_homeostatic,
        device=device
    )
    
    print(f"Architecture: {layer_sizes}")
    print(f"Sparsity: {sparsity:.1%}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = BioPredNetTrainer(
        model=model,
        device=device,
        checkpoint_dir=save_dir
    )
    
    # Train
    print("\n" + "="*70)
    print(" Training")
    print("="*70)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        num_classes=10,
        early_stopping_patience=15,
        save_best=True,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print(" Test Set Evaluation")
    print("="*70)
    
    test_metrics = trainer.validate(test_loader, num_classes=10, verbose=True)
    
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Prediction Error: {test_metrics['avg_prediction_error']:.4f}")
    
    # Save results
    trainer.save_history('training_history.json')
    
    # Visualizations
    print("\n" + "="*70)
    print(" Generating Visualizations")
    print("="*70)
    
    # Training dashboard
    print("Creating training dashboard...")
    create_training_dashboard(
        history['train_metrics'],
        target_sparsity=sparsity,
        save_path=os.path.join(save_dir, 'training_dashboard.png')
    )
    
    # Learned features from first layer
    print("Visualizing learned features...")
    first_layer_weights = model.layers[0].W_forward
    plot_learned_features(
        first_layer_weights,
        img_shape=(28, 28),
        n_features=64,
        title="First Layer Features (MNIST)",
        save_path=os.path.join(save_dir, 'learned_features.png')
    )
    
    # Get sample batch for analysis
    sample_data, _ = next(iter(test_loader))
    sample_data = sample_data.to(device).view(sample_data.shape[0], -1)
    
    # Reconstructions
    print("Generating reconstructions...")
    reconstructions = model.get_reconstruction(sample_data, layer_idx=0)
    plot_reconstructions(
        sample_data[:10],
        reconstructions[:10],
        n_samples=10,
        img_shape=(28, 28),
        title="MNIST Reconstructions (First Layer)",
        save_path=os.path.join(save_dir, 'reconstructions.png')
    )
    
    # Neuron activity analysis
    print("Analyzing neuron activity...")
    activations = model.get_layer_representations(sample_data[:1000])
    for i, act in enumerate(activations[1:]):  # Skip input
        plot_neuron_activity(
            act,
            title=f"Neuron Activity - Layer {i}",
            save_path=os.path.join(save_dir, f'activity_layer_{i}.png')
        )
    
    # Biological analysis
    print("\nPerforming biological analysis...")
    bio_metrics = trainer.analyze_biological_properties(test_loader, num_samples=5000)
    
    print("\nBiological Metrics:")
    for layer_idx in range(len(model.layers)):
        layer_key = f'layer_{layer_idx}'
        if layer_key in bio_metrics:
            stats = bio_metrics[layer_key]
            print(f"\nLayer {layer_idx}:")
            print(f"  Sparsity: {stats['sparsity']:.2%}")
            print(f"  Neuron Utilization: {stats['utilization']['utilization']:.2%}")
            print(f"  Dead Neurons: {stats['utilization']['dead_neurons']:.2%}")
            print(f"  Efficiency Gain: {stats['efficiency']['efficiency_gain']:.2%}")
            print(f"  Speedup: {stats['efficiency']['speedup']:.2f}x")
    
    print("\n" + "="*70)
    print(" Experiment Complete!")
    print("="*70)
    print(f"\nResults saved to: {save_dir}")
    print(f"Best Validation Accuracy: {history['best_val_accuracy']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    return history, test_metrics


if __name__ == '__main__':
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run experiment
    history, test_metrics = run_mnist_experiment(
        device=device,
        sparsity=0.15,
        epochs=50,
        batch_size=128
    )
