"""
test_network.py
~~~~~~~~~~~~~~~

A simple script to test the neural network on MNIST data.
Make sure you have the mnist.pkl.gz file in a ../data/ directory.

Usage:
    python test_network.py
"""

import mnist_loader
import network

# Load the MNIST data
print("Loading MNIST data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create a network with 784 input neurons, 30 hidden neurons, and 10 output neurons
print("Creating neural network [784, 30, 10]...")
net = network.Network([784, 30, 10])

# Train the network using stochastic gradient descent
print("Training the network...")
print("Parameters: 30 epochs, mini-batch size 10, learning rate 3.0")
print("-" * 60)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

print("-" * 60)
print("Training complete!")
