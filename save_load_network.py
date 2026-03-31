"""
save_load_network.py
~~~~~~~~~~~~~~~~~~~~

Utility functions to save and load trained neural networks.
"""

import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network


def save_network(net, filename="trained_network.pkl"):
    """Save the network parameters together with useful training metadata."""
    _, _, test_data = mnist_loader.load_data_wrapper()
    accuracy = net.evaluate(test_data)
    total = len(test_data)
    percentage = (accuracy / total) * 100

    data = {
        "sizes": net.sizes,
        "weights": net.weights,
        "biases": net.biases,
        "accuracy": percentage,
        "hidden_activation": getattr(net, "hidden_activation", "sigmoid"),
        "output_activation": getattr(net, "output_activation", "sigmoid"),
        "cost": getattr(net, "cost", "cross_entropy"),
    }

    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved network to '{filename}'")
    print(f"Saved accuracy: {percentage:.2f}%")


def load_network(filename="trained_network.pkl", show_accuracy=True):
    """Load a saved network.

    Older saved files do not contain activation metadata, so they are restored
    with the original all-sigmoid behavior for backward compatibility.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Network file '{filename}' not found!")

    with open(filename, "rb") as f:
        data = pickle.load(f)

    hidden_activation = data.get("hidden_activation", "sigmoid")
    output_activation = data.get("output_activation", "sigmoid")
    cost = data.get("cost", "cross_entropy")

    net = network.Network(
        data["sizes"],
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        cost=cost,
    )
    net.weights = data["weights"]
    net.biases = data["biases"]

    print(f"Loaded network from '{filename}'")
    if "accuracy" in data:
        print(f"Saved accuracy: {data['accuracy']:.2f}%")
    print(f"Architecture: {net.sizes}")
    print(f"Hidden activation: {net.hidden_activation}")
    print(f"Output activation: {net.output_activation}")

    if show_accuracy:
        print("\nEvaluating network...")
        _, _, test_data = mnist_loader.load_data_wrapper()
        accuracy = net.evaluate(test_data)
        total = len(test_data)
        percentage = (accuracy / total) * 100
        print(f"Current accuracy: {accuracy} / {total} ({percentage:.2f}%)")

    return net


def train_and_save(epochs=12, filename="trained_network.pkl"):
    """Train an upgraded dense network and save it."""
    print("=" * 60)
    print(f"Training New Network ({epochs} epochs)")
    print("=" * 60)

    print("\nLoading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    print("Creating neural network [784, 128, 64, 10] with ReLU hidden layers...")
    net = network.Network(
        [784, 128, 64, 10],
        hidden_activation="relu",
        output_activation="sigmoid",
        cost="cross_entropy",
    )

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    net.SGD(
        training_data,
        epochs,
        20,
        0.2,
        test_data=test_data,
        validation_data=validation_data,
        lmbda=0.0001,
    )
    print("-" * 60)

    save_network(net, filename)
    return net


def test_saved_network(filename="trained_network.pkl"):
    """Load a saved network and print its current test accuracy."""
    print("\n" + "=" * 60)
    print("Testing Saved Network")
    print("=" * 60)

    net = load_network(filename)

    print("\nLoading test data...")
    _, _, test_data = mnist_loader.load_data_wrapper()

    print("Evaluating network on test data...")
    accuracy = net.evaluate(test_data)
    total = len(test_data)
    percentage = (accuracy / total) * 100

    print(f"\nAccuracy: {accuracy} / {total} ({percentage:.2f}%)")
    return net


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network Save/Load Utility")
    print("=" * 60)

    print("\nWhat would you like to do?")
    print("1. Train a new network and save it (12 epochs)")
    print("2. Train quickly and save it (5 epochs)")
    print("3. Load an existing network and test it")
    print("4. Load a network and show some predictions")

    choice = input("\nChoice (1/2/3/4): ").strip()

    if choice == "1":
        net = train_and_save(epochs=12, filename="trained_network_12.pkl")
        print("\nDone! You can now use this network without retraining.")

    elif choice == "2":
        net = train_and_save(epochs=5, filename="trained_network_5.pkl")
        print("\nDone! Network saved (quick training).")

    elif choice == "3":
        filename = input("Enter filename (default: trained_network.pkl): ").strip()
        if not filename:
            filename = "trained_network.pkl"

        try:
            net = test_saved_network(filename)
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nTip: Train and save a network first (option 1 or 2)")

    elif choice == "4":
        filename = input("Enter filename (default: trained_network.pkl): ").strip()
        if not filename:
            filename = "trained_network.pkl"

        try:
            net = load_network(filename)

            print("\nLoading test data...")
            _, _, test_data = mnist_loader.load_data_wrapper()

            print("Showing 10 random predictions...")
            samples = random.sample(list(test_data), 10)
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            fig.suptitle("Network Predictions", fontsize=16)

            for idx, (image, label) in enumerate(samples):
                row = idx // 5
                col = idx % 5
                ax = axes[row, col]

                output = net.feedforward(image)
                prediction = np.argmax(output)

                image_2d = image.reshape(28, 28)
                ax.imshow(image_2d, cmap="gray")

                color = "green" if prediction == label else "red"
                ax.set_title(f"Pred: {prediction}, True: {label}", color=color)
                ax.axis("off")

            plt.tight_layout()
            plt.show()

        except FileNotFoundError as e:
            print(f"\nError: {e}")

    else:
        print("Invalid choice!")
