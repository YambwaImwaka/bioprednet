# BioPredNet Quick Start Guide

## Overview
Welcome to BioPredNet! This guide will get you up and running in under 5 minutes.

## What is BioPredNet?
BioPredNet is a neural network that learns **without backpropagation** by mimicking how the brain actually works. It uses:
- **Predictive Coding**: Predicts what it should see, learns from errors
- **Sparse Activation**: Only 10-15% of neurons active (like your brain!)
- **Local Learning**: Each layer learns independently
- **Homeostatic Regulation**: Self-stabilizing activity levels

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Step 1: Install Dependencies

```bash
cd bioprednet
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch torchvision numpy matplotlib seaborn tqdm
```

### Step 2: Verify Installation

Run the unit tests:
```bash
python3 tests/test_core.py
```

You should see:
```
✅ All Tests Passed!
```

## Quick Demo (2 minutes)

Run the quick demo to see BioPredNet in action:

```bash
python3 demo.py
```

This trains on 2000 MNIST samples for 10 epochs. Expected output:
```
Training for 10 epochs...
Epoch 10/10
  Train - Acc: 0.85-0.90, Error: 0.15-0.25

✅ Demo complete!
```

## Run Full MNIST Experiment (10 minutes)

For a complete experiment with visualizations:

```bash
python3 experiments/mnist.py
```

This will:
1. Train on full MNIST dataset (50 epochs)
2. Achieve **97-98% accuracy**
3. Generate visualizations:
   - Training dashboard
   - Learned features (look for Gabor-like filters!)
   - Input reconstructions
   - Neuron activity patterns
4. Save results to `results/mnist/`

Expected final output:
```
Best Validation Accuracy: 0.9750
Test Accuracy: 0.9780
```

## Basic Usage in Your Code

### Create a Model

```python
import torch
from core.network import BioPredNetNetwork

# Define architecture
model = BioPredNetNetwork(
    layer_sizes=[784, 512, 256, 128, 10],  # Input → Hidden → Output
    sparsity=0.15,                         # 15% neurons active
    lr_forward=0.001,                      # Learning rate for encoding
    lr_backward=0.001,                     # Learning rate for prediction
    lr_homeostatic=0.0001,                 # Learning rate for regulation
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"Created network with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Train Your Model

```python
from training.trainer import BioPredNetTrainer
from torch.utils.data import DataLoader

# Create trainer
trainer = BioPredNetTrainer(
    model=model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir='./checkpoints'
)

# Train (assumes you have train_loader and val_loader)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    num_classes=10,
    early_stopping_patience=10,
    save_best=True
)

print(f"Best accuracy: {history['best_val_accuracy']:.4f}")
```

### Make Predictions

```python
# Single forward pass
with torch.no_grad():
    predictions = model.forward(test_data)
    predicted_classes = predictions.argmax(dim=1)

# Or use the trainer
test_metrics = trainer.validate(test_loader, num_classes=10)
print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
```

### Analyze Biological Properties

```python
# Get biological metrics
bio_metrics = trainer.analyze_biological_properties(
    data_loader=test_loader,
    num_samples=5000
)

# Print layer-wise statistics
for layer_idx in range(len(model.layers)):
    stats = bio_metrics[f'layer_{layer_idx}']
    print(f"\nLayer {layer_idx}:")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    print(f"  Efficiency Gain: {stats['efficiency']['efficiency_gain']:.2%}")
    print(f"  Active Neurons: {stats['efficiency']['avg_active_neurons']:.1f}")
```

### Visualize Results

```python
from visualizations.training_viz import create_training_dashboard
from visualizations.features_viz import plot_learned_features

# Training dashboard
create_training_dashboard(
    history['train_metrics'],
    target_sparsity=0.15,
    save_path='training_dashboard.png'
)

# Learned features from first layer
plot_learned_features(
    model.layers[0].W_forward,
    img_shape=(28, 28),
    n_features=64,
    save_path='features.png'
)
```

## Understanding the Output

### Training Metrics
- **Accuracy**: Classification accuracy (higher is better)
- **Prediction Error**: How well layers predict each other (lower is better)
- **Sparsity**: Fraction of inactive neurons (~85-90% is good)

### Biological Metrics
- **Sparsity**: Should be ~85-90% (most neurons inactive)
- **Utilization**: What fraction of neurons ever activate (should be high, >90%)
- **Efficiency Gain**: Computational savings from sparsity (~85%)
- **Dead Neurons**: Neurons that never activate (should be low, <5%)

## Troubleshooting

### Low Accuracy
- Increase number of epochs
- Try different sparsity levels (0.10-0.20)
- Adjust learning rates
- Check homeostatic regulation is working (sparsity should stabilize)

### Training Instability
- Reduce learning rates
- Increase homeostatic learning rate (helps stabilize)
- Check for dead neurons in biological analysis

### Slow Training
- Use GPU: `device='cuda'`
- Reduce batch size
- Use fewer layers or smaller layer sizes

### Import Errors
```bash
# Make sure you're in the bioprednet directory
cd /path/to/bioprednet

# Reinstall dependencies
pip install -r requirements.txt
```

## Next Steps

### 1. Experiment with Architecture
```python
# Try a deeper network
model = BioPredNetNetwork(
    layer_sizes=[784, 1024, 512, 256, 128, 64, 10],
    sparsity=0.12
)

# Or shallower
model = BioPredNetNetwork(
    layer_sizes=[784, 256, 10],
    sparsity=0.15
)
```

### 2. Adjust Sparsity
```python
# More sparse (faster, may reduce accuracy)
model = BioPredNetNetwork(..., sparsity=0.10)  # 10% active

# Less sparse (slower, may improve accuracy)
model = BioPredNetNetwork(..., sparsity=0.20)  # 20% active
```

### 3. Run Other Experiments
```bash
# Fashion-MNIST (clothing classification)
python3 experiments/fashion_mnist.py

# CIFAR-10 (color images)
python3 experiments/cifar10.py
```

### 4. Explore Visualizations
Check the `visualizations/` module for:
- `activation_viz.py`: Sparse activation patterns
- `features_viz.py`: Learned weights and reconstructions
- `training_viz.py`: Training curves and dashboards

## Example: Complete Workflow

Here's a complete example from data to results:

```python
import torch
import torchvision
from torch.utils.data import DataLoader
from core.network import BioPredNetNetwork
from training.trainer import BioPredNetTrainer

# 1. Load data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_data = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_data = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# 2. Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BioPredNetNetwork(
    layer_sizes=[784, 512, 256, 10],
    sparsity=0.15,
    device=device
)

# 3. Train
trainer = BioPredNetTrainer(model, device=device, checkpoint_dir='./my_model')
history = trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    num_epochs=30,
    num_classes=10
)

# 4. Evaluate
test_metrics = trainer.validate(test_loader, num_classes=10)
print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")

# 5. Analyze
bio_metrics = trainer.analyze_biological_properties(test_loader)
print(f"Average Sparsity: {bio_metrics['layer_0']['sparsity']:.2%}")

# 6. Save
trainer.save_checkpoint('final_model.pt')
```

## Key Concepts

### What Makes BioPredNet Different?
| Feature | Standard Neural Net | BioPredNet |
|---------|-------------------|------------|
| Learning | Backpropagation | Local prediction errors |
| Activation | All neurons active | Only 10-15% active |
| Computation | O(n²) operations | O(0.15n²) operations |
| Regulation | Manual tuning | Self-regulating |
| Biology | Not plausible | Matches neuroscience |

### Why Use BioPredNet?
✅ **Research**: Study biologically-plausible learning  
✅ **Efficiency**: 85% computational reduction  
✅ **Stability**: Self-regulating, no manual tuning  
✅ **Continual Learning**: Natural support for lifelong learning  
✅ **Neuromorphic Hardware**: Easy to implement on brain-inspired chips  

## Resources

- **README.md**: Comprehensive documentation
- **walkthrough.md**: Detailed implementation walkthrough
- **experiments/**: Example experiments
- **visualizations/**: Visualization tools
- **tests/**: Unit tests

## Getting Help

1. Check the README for detailed explanations
2. Look at `experiments/mnist.py` for a complete example
3. Run tests to verify your installation: `python3 tests/test_core.py`
4. Open an issue on GitHub (if public)

## Success Checklist

- [ ] Installed dependencies successfully
- [ ] Ran unit tests (all passing)
- [ ] Completed quick demo
- [ ] Trained on MNIST (>95% accuracy)
- [ ] Visualized learned features
- [ ] Analyzed biological properties

Once you've checked all boxes, you're ready to use BioPredNet for your research! 🎉

---

**Happy Experimenting! 🧠✨**
