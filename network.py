"""
network.py
~~~~~~~~~~

A feedforward neural network trained with mini-batch stochastic gradient
descent and backpropagation.

This version keeps the original project's spirit, but upgrades the core math
so deeper dense networks train more reliably:
- ReLU hidden layers are supported
- weight initialization is scaled to the layer size
- optional L2 weight decay is built into SGD
- older saved sigmoid networks still remain compatible
"""

# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):
    def __init__(
        self,
        sizes,
        hidden_activation="relu",
        output_activation="sigmoid",
        cost="cross_entropy",
    ):
        """Create a feedforward network with configurable activations.

        Args:
            sizes: Layer sizes such as [784, 256, 128, 10]
            hidden_activation: Activation for every hidden layer
            output_activation: Activation for the output layer
            cost: Cost function name used during backpropagation
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.cost = cost

        # Biases stay centered near zero. The weights use a variance scaled to the
        # incoming fan-in so deeper networks keep activations in a healthier range.
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [self._initialize_weights(x, y) for x, y in zip(sizes[:-1], sizes[1:])]

    def _initialize_weights(self, input_size, output_size):
        """Initialize weights with a scale matched to the chosen activation."""
        if self.hidden_activation == "relu":
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)
        return np.random.randn(output_size, input_size) * scale

    def _activation_name_for_layer(self, layer_index):
        """Return the activation used at a specific layer."""
        # Layer numbering here starts at 2 because layer 1 is the input layer.
        # That means the final output layer has index ``self.num_layers``.
        if layer_index == self.num_layers:
            return self.output_activation
        return self.hidden_activation

    def _activation(self, z, activation_name):
        """Apply the configured activation function."""
        if activation_name == "sigmoid":
            return sigmoid(z)
        if activation_name == "relu":
            return relu(z)
        if activation_name == "softmax":
            return softmax(z)
        raise ValueError(f"Unsupported activation: {activation_name}")

    def _activation_prime(self, z, activation_name):
        """Return the derivative of the configured activation."""
        if activation_name == "sigmoid":
            return sigmoid_prime(z)
        if activation_name == "relu":
            return relu_prime(z)
        raise ValueError(f"Derivative not supported for activation: {activation_name}")

    def feedforward(self, a):
        """Return the network output for input ``a``."""
        for layer_index, (b, w) in enumerate(zip(self.biases, self.weights), start=2):
            z = np.dot(w, a) + b
            a = self._activation(z, self._activation_name_for_layer(layer_index))
        return a

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        test_data=None,
        lmbda=0.0,
        validation_data=None,
    ):
        """Train the network using mini-batch stochastic gradient descent.

        Args:
            training_data: List of (x, y) pairs
            epochs: Number of full passes through the training set
            mini_batch_size: Number of samples per update
            eta: Learning rate
            test_data: Optional held-out data for progress reporting
            lmbda: L2 weight decay strength
            validation_data: Optional validation set for additional monitoring
        """
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        if validation_data:
            validation_data = list(validation_data)
            n_validation = len(validation_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            message = f"Epoch {j}"
            if validation_data:
                validation_correct = self.evaluate(validation_data)
                validation_pct = 100.0 * validation_correct / n_validation
                message += f" | validation: {validation_correct} / {n_validation} ({validation_pct:.2f}%)"
            if test_data:
                test_correct = self.evaluate(test_data)
                test_pct = 100.0 * test_correct / n_test
                message += f" | test: {test_correct} / {n_test} ({test_pct:.2f}%)"
            if not validation_data and not test_data:
                message += " complete"
            print(message)

    def update_mini_batch(self, mini_batch, eta, lmbda, training_size):
        """Update weights and biases using one mini-batch.

        The weight update includes L2 decay:
            W <- (1 - eta * lambda / n) * W - eta/m * dC/dW
        where n is the full training set size and m is the mini-batch size.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        mini_batch_scale = eta / len(mini_batch)
        weight_decay = 1 - eta * (lmbda / training_size)
        self.weights = [
            weight_decay * w - mini_batch_scale * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - mini_batch_scale * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """Return gradients for the cost with respect to biases and weights."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Store every activation and weighted input so the chain rule can walk
        # backward through the network during backpropagation.
        activation = x
        activations = [x]
        zs = []
        activation_names = []

        for layer_index, (b, w) in enumerate(zip(self.biases, self.weights), start=2):
            z = np.dot(w, activation) + b
            activation_name = self._activation_name_for_layer(layer_index)
            activation = self._activation(z, activation_name)
            zs.append(z)
            activations.append(activation)
            activation_names.append(activation_name)

        delta = self._output_delta(zs[-1], activations[-1], y, activation_names[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # For hidden layers, each error term is the next layer's error projected
        # backward through the weights and then scaled by the local derivative.
        for l in range(2, self.num_layers):
            z = zs[-l]
            activation_name = activation_names[-l]
            sp = self._activation_prime(z, activation_name)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def _output_delta(self, z, output_activations, y, activation_name):
        """Return the output-layer error term used by backpropagation."""
        # For cross-entropy with sigmoid or softmax output, the derivative of the
        # output activation cancels algebraically with the loss derivative.
        if self.cost == "cross_entropy" and activation_name in {"sigmoid", "softmax"}:
            return output_activations - y
        return self.cost_derivative(output_activations, y) * self._activation_prime(z, activation_name)

    def evaluate(self, test_data):
        """Return the number of correctly classified test inputs."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        r"""Return \partial C / \partial a for the output activations."""
        return output_activations - y


def sigmoid(z):
    """The sigmoid activation squashes values into the range (0, 1)."""
    clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_prime(z):
    """Derivative of sigmoid."""
    sig = sigmoid(z)
    return sig * (1 - sig)


def relu(z):
    """ReLU keeps positive values and clips negative values to zero."""
    return np.maximum(0.0, z)


def relu_prime(z):
    """Derivative of ReLU: 1 on the positive side, 0 on the negative side."""
    return (z > 0).astype(np.float64)


def softmax(z):
    """Softmax converts logits into normalized class probabilities."""
    shifted = z - np.max(z)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)
