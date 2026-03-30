"""
quick_test.py
~~~~~~~~~~~~~

A quick test to verify the Python 3 migration works correctly.
Tests with just 3 epochs to verify functionality.
"""

import mnist_loader
import network

print("=" * 60)
print("Quick Test - Python 3 Neural Network")
print("=" * 60)

# Load the MNIST data
print("\n1. Loading MNIST data...")
try:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(f"   ✓ Training data: {len(training_data)} examples")
    print(f"   ✓ Validation data: {len(validation_data)} examples")
    print(f"   ✓ Test data: {len(test_data)} examples")
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    exit(1)

# Create a network
print("\n2. Creating neural network [784, 30, 10]...")
try:
    net = network.Network([784, 30, 10])
    print("   ✓ Network created successfully")
    print(f"   - Number of layers: {net.num_layers}")
    print(f"   - Layer sizes: {net.sizes}")
except Exception as e:
    print(f"   ✗ Error creating network: {e}")
    exit(1)

# Test feedforward
print("\n3. Testing feedforward...")
try:
    test_input = training_data[0][0]
    output = net.feedforward(test_input)
    print(f"   ✓ Feedforward works (output shape: {output.shape})")
except Exception as e:
    print(f"   ✗ Error in feedforward: {e}")
    exit(1)

# Train for just 3 epochs as a quick test
print("\n4. Training for 3 epochs (quick test)...")
print("-" * 60)
try:
    net.SGD(training_data, 3, 10, 3.0, test_data=test_data)
    print("-" * 60)
    print("   ✓ Training completed successfully!")
except Exception as e:
    print(f"   ✗ Error during training: {e}")
    exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("The Python 3 migration is working correctly.")
print("=" * 60)
