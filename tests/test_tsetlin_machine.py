import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from tsetlin_machine import TsetlinMachine

@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data"""
    np.random.seed(42)
    # Create synthetic binary data
    n_samples = 1000
    n_features = 20
    
    # Generate random binary features
    X = np.random.randint(2, size=(n_samples, n_features), dtype=np.uint8)
    # Generate binary targets (0 or 1)
    y = np.random.randint(2, size=n_samples, dtype=np.uint32)
    
    # Split into train and test
    train_size = int(0.8 * n_samples)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    return X_train, y_train, X_test, y_test

@pytest.fixture
def basic_tsetlin():
    """Create a basic TsetlinMachine instance"""
    return TsetlinMachine(n_clauses=10, s=3.0, threshold=10)

def test_initialization(basic_tsetlin):
    """Test proper initialization of TsetlinMachine"""
    assert basic_tsetlin.n_clauses == 10
    assert basic_tsetlin.s == 3.0
    assert basic_tsetlin.threshold == 10
    assert basic_tsetlin.C is None
    assert basic_tsetlin.W is None

def test_data_validation(basic_tsetlin):
    """Test data validation checks"""
    invalid_x = np.random.rand(100, 10)  # Float data instead of binary
    invalid_y = np.random.randint(2, size=50)  # Mismatched samples
    valid_x = np.random.randint(2, size=(100, 10), dtype=np.uint8)
    valid_y = np.random.randint(2, size=100, dtype=np.uint32)
    
    # Test invalid data type
    with pytest.raises(ValueError):
        basic_tsetlin.set_train_data(invalid_x, valid_y)
    
    # Test mismatched samples
    with pytest.raises(ValueError):
        basic_tsetlin.set_train_data(valid_x, invalid_y)

def test_memory_allocation(basic_tsetlin, sample_binary_data):
    """Test proper memory allocation"""
    X_train, y_train, _, _ = sample_binary_data
    
    # Test allocation without setting data
    with pytest.raises(ValueError):
        basic_tsetlin.allocate_memory()
    
    # Test proper allocation
    basic_tsetlin.set_train_data(X_train, y_train)
    basic_tsetlin.allocate_memory()
    
    expected_literal_count = X_train.shape[1] * 2  # Due to feature negation
    assert basic_tsetlin.C.shape == (basic_tsetlin.n_clauses, expected_literal_count)
    assert basic_tsetlin.W.shape == (basic_tsetlin.n_outputs, basic_tsetlin.n_clauses)

def test_training_performance(basic_tsetlin, sample_binary_data):
    """Test training performance and convergence"""
    X_train, y_train, X_test, y_test = sample_binary_data
    
    basic_tsetlin.set_train_data(X_train, y_train)
    basic_tsetlin.set_eval_data(X_test, y_test)
    
    # Train for a few epochs
    results = basic_tsetlin.train(
        training_epochs=5,
        eval_freq=1,
        hide_progress_bar=True
    )
    
    # Verify training results structure
    assert 'train_time' in results
    assert 'eval_acc' in results
    
    # Test prediction functionality
    predictions = basic_tsetlin.predict(X_test)
    assert predictions.shape == y_test.shape
    assert np.all((predictions == 0) | (predictions == 1))  # Binary predictions
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    assert 0 <= accuracy <= 1  # Verify accuracy is in valid range

def test_reset_functionality(basic_tsetlin, sample_binary_data):
    """Test reset functionality"""
    X_train, y_train, X_test, y_test = sample_binary_data
    
    basic_tsetlin.set_train_data(X_train, y_train)
    basic_tsetlin.set_eval_data(X_test, y_test)
    basic_tsetlin.allocate_memory()
    
    # Store initial state
    initial_C = basic_tsetlin.C.copy()
    initial_W = basic_tsetlin.W.copy()
    
    # Train for a few epochs
    basic_tsetlin.train(training_epochs=2, hide_progress_bar=True)
    
    # Verify state changed
    assert not np.array_equal(initial_C, basic_tsetlin.C)
    assert np.array_equal(initial_W, basic_tsetlin.W)  # W should not change during training
    
    # Reset
    basic_tsetlin.reset()
    assert basic_tsetlin.C is None
    assert basic_tsetlin.W is None
    assert basic_tsetlin.x_train is None
    assert basic_tsetlin.y_train is None

def test_early_stopping(basic_tsetlin, sample_binary_data):
    """Test early stopping functionality"""
    X_train, y_train, X_test, y_test = sample_binary_data
    
    basic_tsetlin.set_train_data(X_train, y_train)
    basic_tsetlin.set_eval_data(X_test, y_test)
    
    # Train with early stopping at 60% accuracy
    results = basic_tsetlin.train(
        training_epochs=100,
        eval_freq=1,
        hide_progress_bar=True,
        early_stop_at=60.0
    )
    
    # Verify training stopped early if accuracy threshold was reached
    assert len(results['train_time']) < 100 or 'best_eval_acc' not in results or results['best_eval_acc'] < 60.0