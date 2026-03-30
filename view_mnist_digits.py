"""
view_mnist_digits.py
~~~~~~~~~~~~~~~~~~~~

Visualize some digits from the MNIST dataset.
This helps you understand what the neural network is learning from.
"""

import mnist_loader
import matplotlib.pyplot as plt
import numpy as np

def show_digit(image_data):
    """Display a single digit image."""
    # Reshape from (784, 1) to (28, 28)
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

def show_mnist_samples(n=10):
    """Show n random samples from the MNIST dataset."""
    print("Loading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Take n random samples from test data
    import random
    samples = random.sample(list(test_data), n)
    
    # Create a grid to display digits
    rows = 2
    cols = n // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4))
    fig.suptitle('Sample Digits from MNIST Dataset', fontsize=16)
    
    for idx, (image, label) in enumerate(samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Reshape and display
        image_2d = image.reshape(28, 28)
        ax.imshow(image_2d, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'mnist_samples.png'")
    plt.show()

def show_prediction_examples(net):
    """Show some test images with network predictions."""
    print("\nLoading test data...")
    _, _, test_data = mnist_loader.load_data_wrapper()
    
    # Take 10 random samples
    import random
    samples = random.sample(list(test_data), 10)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Network Predictions vs True Labels', fontsize=16)
    
    for idx, (image, label) in enumerate(samples):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # Get network prediction
        output = net.feedforward(image)
        prediction = np.argmax(output)
        
        # Display
        image_2d = image.reshape(28, 28)
        ax.imshow(image_2d, cmap='gray')
        
        # Color title based on correctness
        color = 'green' if prediction == label else 'red'
        ax.set_title(f'Pred: {prediction}\nTrue: {label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved predictions to 'predictions.png'")
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("MNIST Digit Visualization")
    print("=" * 60)
    
    show_mnist_samples(10)
    
    print("\nWould you like to see network predictions?")
    print("(You need to train a network first)")
    response = input("Load and test a network? (y/n): ").strip().lower()
    
    if response == 'y':
        import network
        print("\nCreating and training network (this takes a moment)...")
        print("Training with 3 epochs for quick demo...")
        
        training_data, _, test_data = mnist_loader.load_data_wrapper()
        net = network.Network([784, 30, 10])
        net.SGD(training_data, 3, 10, 3.0, test_data=test_data)
        
        show_prediction_examples(net)
