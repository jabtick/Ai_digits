"""
simple_demo.py
~~~~~~~~~~~~~~

A simple command-line demo showing the network in action.
No GUI required - just shows random predictions.
"""

import network
import mnist_loader
import numpy as np
import random

def show_digit_ascii(image_data):
    """Display a digit as ASCII art."""
    image = image_data.reshape(28, 28)
    
    # ASCII characters from dark to light
    chars = " .:-=+*#%@"
    
    print("\n" + "=" * 56)
    for row in image:
        line = ""
        for pixel in row:
            char_index = int(pixel * (len(chars) - 1))
            line += chars[char_index] * 2  # Double width for better aspect ratio
        print(line)
    print("=" * 56)

def interactive_demo():
    """Interactive demo showing predictions on test data."""
    print("\n" + "=" * 60)
    print("Neural Network Digit Recognition - Interactive Demo")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Create and train network
    print("Creating neural network...")
    net = network.Network([784, 30, 10])
    
    print("\nTraining network (3 epochs for quick demo)...")
    print("For better accuracy, train with 30 epochs (takes longer)")
    print("-" * 60)
    net.SGD(training_data, 3, 10, 3.0, test_data=test_data)
    print("-" * 60)
    
    print("\n✓ Network ready! Starting interactive demo...\n")
    
    test_list = list(test_data)
    
    while True:
        input("\nPress Enter to see a random prediction (or Ctrl+C to quit)...")
        
        # Pick random test image
        image, true_label = random.choice(test_list)
        
        # Get prediction
        output = net.feedforward(image)
        prediction = np.argmax(output)
        confidence = output[prediction][0] * 100
        
        # Show ASCII art of digit
        show_digit_ascii(image)
        
        # Show prediction
        print(f"\n🤖 Network prediction: {prediction}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"✓  True label: {true_label}")
        
        if prediction == true_label:
            print("   Result: ✓ CORRECT!")
        else:
            print(f"   Result: ✗ WRONG (should be {true_label})")
        
        # Show all activations
        print("\n   All output activations:")
        for digit in range(10):
            activation = output[digit][0]
            bar_length = int(activation * 40)
            bar = "█" * bar_length
            marker = " ← PREDICTION" if digit == prediction else ""
            print(f"   {digit}: {bar} {activation:.3f}{marker}")

def batch_test():
    """Test the network on multiple images and show statistics."""
    print("\n" + "=" * 60)
    print("Batch Test - Testing Network Accuracy")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    
    # Create and train network
    print("Creating neural network...")
    net = network.Network([784, 30, 10])
    
    print("\nTraining network (5 epochs)...")
    print("-" * 60)
    net.SGD(training_data, 5, 10, 3.0, test_data=test_data)
    print("-" * 60)
    
    # Test on random samples
    print("\nTesting on 20 random samples...\n")
    
    test_list = list(test_data)
    samples = random.sample(test_list, 20)
    
    correct = 0
    for idx, (image, true_label) in enumerate(samples, 1):
        output = net.feedforward(image)
        prediction = np.argmax(output)
        confidence = output[prediction][0] * 100
        
        is_correct = prediction == true_label
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{idx:2d}. True: {true_label}  Pred: {prediction}  "
              f"Confidence: {confidence:5.1f}%  {status}")
    
    accuracy = (correct / len(samples)) * 100
    print(f"\n{'=' * 60}")
    print(f"Results: {correct}/{len(samples)} correct ({accuracy:.1f}% accuracy)")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network Digit Recognition Demo")
    print("=" * 60)
    
    print("\nChoose a demo:")
    print("1. Interactive demo (see ASCII digits and predictions)")
    print("2. Batch test (test on 20 random samples)")
    
    choice = input("\nChoice (1 or 2, default=1): ").strip()
    
    try:
        if choice == '2':
            batch_test()
        else:
            interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo ended. Goodbye!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
