"""
save_load_network.py
~~~~~~~~~~~~~~~~~~~~

Utility functions to save and load trained neural networks.
This lets you train once and reuse the network many times.
"""

import pickle
import network
import mnist_loader
import os

def save_network(net, filename="trained_network.pkl"):
    """
    Save a trained network to a file.
    
    Args:
        net: The Network object to save
        filename: Name of the file to save to
    """
    data = {
        'sizes': net.sizes,
        'weights': net.weights,
        'biases': net.biases
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Network saved to '{filename}'")
    return filename

def load_network(filename="trained_network.pkl"):
    """
    Load a trained network from a file.
    
    Args:
        filename: Name of the file to load from
        
    Returns:
        Network object with loaded weights and biases
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Network file '{filename}' not found!")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    net = network.Network(data['sizes'])
    net.weights = data['weights']
    net.biases = data['biases']
    
    print(f"✓ Network loaded from '{filename}'")
    return net

def train_and_save(epochs=30, filename="trained_network.pkl"):
    """
    Train a new network and save it.
    
    Args:
        epochs: Number of training epochs
        filename: File to save the trained network to
    """
    print("=" * 60)
    print(f"Training New Network ({epochs} epochs)")
    print("=" * 60)
    
    print("\nLoading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    print("Creating neural network [784, 30, 10]...")
    net = network.Network([784, 30, 10])
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    net.SGD(training_data, epochs, 10, 3.0, test_data=test_data)
    print("-" * 60)
    
    # Save the trained network
    save_network(net, filename)
    
    return net

def test_saved_network(filename="trained_network.pkl"):
    """
    Load a saved network and test its accuracy.
    
    Args:
        filename: File containing the saved network
    """
    print("\n" + "=" * 60)
    print("Testing Saved Network")
    print("=" * 60)
    
    # Load network
    net = load_network(filename)
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_data = mnist_loader.load_data_wrapper()
    
    # Evaluate
    print("Evaluating network on test data...")
    accuracy = net.evaluate(test_data)
    total = len(test_data)
    percentage = (accuracy / total) * 100
    
    print(f"\n✓ Accuracy: {accuracy} / {total} ({percentage:.2f}%)")
    
    return net


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network Save/Load Utility")
    print("=" * 60)
    
    print("\nWhat would you like to do?")
    print("1. Train a new network and save it (30 epochs)")
    print("2. Train quickly and save it (5 epochs)")
    print("3. Load an existing network and test it")
    print("4. Load a network and show some predictions")
    
    choice = input("\nChoice (1/2/3/4): ").strip()
    
    if choice == '1':
        net = train_and_save(epochs=30, filename="trained_network_30.pkl")
        print("\n✓ Done! You can now use this network without retraining.")
        
    elif choice == '2':
        net = train_and_save(epochs=5, filename="trained_network_5.pkl")
        print("\n✓ Done! Network saved (quick training).")
        
    elif choice == '3':
        filename = input("Enter filename (default: trained_network.pkl): ").strip()
        if not filename:
            filename = "trained_network.pkl"
        
        try:
            net = test_saved_network(filename)
        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
            print("\nTip: Train and save a network first (option 1 or 2)")
    
    elif choice == '4':
        filename = input("Enter filename (default: trained_network.pkl): ").strip()
        if not filename:
            filename = "trained_network.pkl"
        
        try:
            net = load_network(filename)
            
            # Show some predictions
            print("\nLoading test data...")
            _, _, test_data = mnist_loader.load_data_wrapper()
            
            import random
            import numpy as np
            import matplotlib.pyplot as plt
            
            print("Showing 10 random predictions...")
            
            samples = random.sample(list(test_data), 10)
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            fig.suptitle('Network Predictions', fontsize=16)
            
            for idx, (image, label) in enumerate(samples):
                row = idx // 5
                col = idx % 5
                ax = axes[row, col]
                
                output = net.feedforward(image)
                prediction = np.argmax(output)
                
                image_2d = image.reshape(28, 28)
                ax.imshow(image_2d, cmap='gray')
                
                color = 'green' if prediction == label else 'red'
                ax.set_title(f'Pred: {prediction}, True: {label}', color=color)
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
    
    else:
        print("Invalid choice!")
