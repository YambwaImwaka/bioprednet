"""
Unit Tests for BioPredNet Core Components

Tests the fundamental components of BioPredNet to ensure correctness.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.layer import BioPredNetLayer
from core.network import BioPredNetNetwork
from core.utils import sparse_topk, compute_sparsity, ActivityTracker


def test_sparse_topk():
    """Test sparse top-k selection."""
    print("Testing sparse_topk...")
    
    x = torch.randn(10, 100)
    sparse_x, indices = sparse_topk(x, k=0.1)
    
    # Check sparsity
    sparsity = compute_sparsity(sparse_x)
    assert 0.85 <= sparsity <= 0.95, f"Expected ~90% sparsity, got {sparsity:.2%}"
    
    # Check that top values are preserved
    for i in range(10):
        top_vals = torch.topk(x[i], 10)[0]
        sparse_vals = sparse_x[i][sparse_x[i] > 0]
        assert torch.allclose(top_vals, sparse_vals.sort(descending=True)[0]), \
            "Top-k values not preserved"
    
    print("✅ sparse_topk test passed")


def test_activity_tracker():
    """Test activity tracking."""
    print("Testing ActivityTracker...")
    
    tracker = ActivityTracker(num_neurons=100, momentum=0.9)
    
    # Generate random activations
    for _ in range(10):
        activations = torch.rand(32, 100)
        avg_activity = tracker.update(activations)
    
    # Check that activity is tracked
    assert avg_activity.shape == (100,), "Wrong activity shape"
    assert 0 <= avg_activity.min() <= 1, "Activity should be in [0, 1]"
    assert 0 <= avg_activity.max() <= 1, "Activity should be in [0, 1]"
    
    print("✅ ActivityTracker test passed")


def test_bioprednet_layer():
    """Test BioPredNetLayer."""
    print("Testing BioPredNetLayer...")
    
    layer = BioPredNetLayer(
        input_dim=784,
        output_dim=256,
        sparsity=0.15,
        device='cpu'
    )
    
    # Forward pass
    x = torch.randn(32, 784)
    h = layer(x)
    
    # Check output shape
    assert h.shape == (32, 256), f"Wrong output shape: {h.shape}"
    
    # Check sparsity
    sparsity = compute_sparsity(h)
    assert 0.80 <= sparsity <= 0.90, f"Expected ~85% sparsity, got {sparsity:.2%}"
    
    # Test prediction
    x_pred = layer.predict_input(h)
    assert x_pred.shape == x.shape, "Wrong prediction shape"
    
    # Test weight update
    error = layer.compute_prediction_error(x, h)
    assert error.shape == x.shape, "Wrong error shape"
    
    layer.update_weights(x, h, error)
    
    # Get statistics
    stats = layer.get_statistics()
    assert 'output_mean' in stats, "Missing statistics"
    
    print("✅ BioPredNetLayer test passed")


def test_bioprednet_network():
    """Test BioPredNetNetwork."""
    print("Testing BioPredNetNetwork...")
    
    model = BioPredNetNetwork(
        layer_sizes=[784, 512, 256, 10],
        sparsity=0.15,
        device='cpu'
    )
    
    # Forward pass
    x = torch.randn(32, 784)
    output = model.forward(x)
    
    assert output.shape == (32, 10), f"Wrong output shape: {output.shape}"
    
    # Get all activations
    activations = model.get_layer_representations(x)
    assert len(activations) == 4, "Wrong number of activation tensors"
    
    # Compute prediction errors
    y = torch.randint(0, 10, (32,))
    y_onehot = torch.zeros(32, 10)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    
    errors = model.compute_all_prediction_errors(y_onehot)
    assert len(errors) == 3, "Wrong number of error tensors"
    
    # Train on batch
    stats = model.train_batch(x, y_onehot)
    assert 'accuracy' in stats, "Missing accuracy in stats"
    assert 'avg_prediction_error' in stats, "Missing prediction error in stats"
    
    print("✅ BioPredNetNetwork test passed")


def test_reconstruction():
    """Test input reconstruction."""
    print("Testing reconstruction...")
    
    model = BioPredNetNetwork(
        layer_sizes=[100, 50, 10],
        sparsity=0.2,
        device='cpu'
    )
    
    # Create input
    x = torch.randn(16, 100)
    
    # Get reconstruction from first layer
    reconstruction = model.get_reconstruction(x, layer_idx=0)
    
    assert reconstruction.shape == x.shape, "Wrong reconstruction shape"
    
    # Reconstruction should be somewhat similar to input
    # (though not perfect since the network is untrained)
    
    print("✅ Reconstruction test passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print(" BioPredNet Unit Tests")
    print("="*60)
    print()
    
    try:
        test_sparse_topk()
        test_activity_tracker()
        test_bioprednet_layer()
        test_bioprednet_network()
        test_reconstruction()
        
        print()
        print("="*60)
        print(" ✅ All Tests Passed!")
        print("="*60)
        return True
        
    except Exception as e:
        print()
        print("="*60)
        print(f" ❌ Tests Failed: {str(e)}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
