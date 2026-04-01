"""
draw_and_predict.py
~~~~~~~~~~~~~~~~~~~

Draw your own digit and test it with the trained neural network!

Usage:
    python draw_and_predict.py

This will:
1. Train a neural network (or load a saved one)
2. Open a drawing window
3. Let you draw a digit
4. Show the network's prediction
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter
import os
import network
import mnist_loader
import save_load_network

CANVAS_SIZE = 280
MNIST_IMAGE_SIZE = 28
TARGET_DIGIT_SIZE = 20
DRAW_THRESHOLD = 0.10
LOW_CONFIDENCE_THRESHOLD = 75.0

class DigitDrawer:
    def __init__(self, net):
        self.net = net
        self.window = tk.Tk()
        self.window.title("Draw a Digit (0-9)")
        
        # Create canvas for drawing
        self.canvas_width = CANVAS_SIZE
        self.canvas_height = CANVAS_SIZE
        self.canvas = tk.Canvas(
            self.window, 
            width=self.canvas_width, 
            height=self.canvas_height,
            bg='white',
            cursor='cross'
        )
        self.canvas.pack()
        
        # Create image for drawing
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_coords)
        self.old_x = None
        self.old_y = None
        
        # Buttons frame
        button_frame = tk.Frame(self.window)
        button_frame.pack()
        
        # Predict button
        predict_btn = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict,
            bg='green',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=20
        )
        predict_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear,
            bg='red',
            fg='white',
            font=('Arial', 14, 'bold'),
            padx=20
        )
        clear_btn.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Prediction label
        self.result_label = tk.Label(
            self.window,
            text="Draw a digit and click Predict",
            font=('Arial', 16, 'bold'),
            fg='blue'
        )
        self.result_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(
            self.window,
            text="Draw with mouse. Try to center the digit and make it large.",
            font=('Arial', 10),
            fg='gray'
        )
        instructions.pack()
        
    def paint(self, event):
        """Draw on canvas."""
        paint_width = 20
        if self.old_x and self.old_y:
            # Draw on canvas
            self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y,
                width=paint_width, fill='black',
                capstyle=tk.ROUND, smooth=tk.TRUE
            )
            # Draw on image
            self.draw.line(
                [self.old_x, self.old_y, event.x, event.y],
                fill='black', width=paint_width
            )
        self.old_x = event.x
        self.old_y = event.y
    
    def reset_coords(self, event):
        """Reset drawing coordinates."""
        self.old_x = None
        self.old_y = None
    
    def clear(self):
        """Clear the canvas."""
        self.canvas.delete('all')
        self.image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(
            text="Draw a digit and click Predict",
            fg='blue'
        )

    def _shift_to_center_of_mass(self, img_array):
        """Shift the digit so its center of mass sits near the image center."""
        coords = np.indices(img_array.shape)
        total_weight = img_array.sum()
        if total_weight <= 0:
            return img_array

        center_y = float((coords[0] * img_array).sum() / total_weight)
        center_x = float((coords[1] * img_array).sum() / total_weight)

        shift_y = int(round((MNIST_IMAGE_SIZE - 1) / 2 - center_y))
        shift_x = int(round((MNIST_IMAGE_SIZE - 1) / 2 - center_x))

        centered = np.zeros_like(img_array)

        src_y_start = max(0, -shift_y)
        src_y_end = min(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE - shift_y)
        dst_y_start = max(0, shift_y)
        dst_y_end = min(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE + shift_y)

        src_x_start = max(0, -shift_x)
        src_x_end = min(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE - shift_x)
        dst_x_start = max(0, shift_x)
        dst_x_end = min(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE + shift_x)

        centered[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img_array[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]
        return centered

    def preprocess_image(self):
        """Convert the drawing into a more MNIST-like 28x28 grayscale image."""
        # Work at the original drawing resolution first so we do not lose stroke details
        # before cropping and centering the digit.
        img_array = 1.0 - (np.array(self.image, dtype=np.float32) / 255.0)
        coords = np.argwhere(img_array > DRAW_THRESHOLD)

        if coords.size == 0:
            blank = np.zeros((MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), dtype=np.float32)
            return blank.reshape(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE, 1), blank

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Pad the crop a little so drawn strokes keep some breathing room after scaling.
        pad = 20
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(img_array.shape[0] - 1, y_max + pad)
        x_max = min(img_array.shape[1] - 1, x_max + pad)

        cropped = self.image.crop((x_min, y_min, x_max + 1, y_max + 1))
        cropped = cropped.filter(ImageFilter.GaussianBlur(radius=0.6))

        width, height = cropped.size
        scale = TARGET_DIGIT_SIZE / max(width, height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cropped.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        # Place the resized digit into the middle of a 28x28 canvas, which mirrors the
        # layout used by MNIST much better than a direct full-frame resize.
        mnist_canvas = Image.new('L', (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), 'white')
        offset_x = (MNIST_IMAGE_SIZE - resized_width) // 2
        offset_y = (MNIST_IMAGE_SIZE - resized_height) // 2
        mnist_canvas.paste(resized, (offset_x, offset_y))

        processed = 1.0 - (np.array(mnist_canvas, dtype=np.float32) / 255.0)
        processed[processed < 0.05] = 0.0
        processed = np.clip(processed * 1.15, 0.0, 1.0)
        processed = self._shift_to_center_of_mass(processed)

        # Reshape to (784, 1) for the network.
        img_vector = processed.reshape(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE, 1)

        img_array = processed.reshape(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
        return img_vector, img_array.reshape(28, 28)
    
    def predict(self):
        """Get network prediction for drawn digit."""
        # Preprocess the image
        input_vector, processed_image = self.preprocess_image()
        
        # Get network output
        output = self.net.feedforward(input_vector)
        prediction = np.argmax(output)
        confidence = output[prediction][0] * 100

        confidence_note = ""
        label_color = 'green'
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            confidence_note = " - low confidence, try drawing it larger/cleaner"
            label_color = 'dark orange'

        # Update label
        self.result_label.config(
            text=f"Prediction: {prediction} (Confidence: {confidence:.1f}%){confidence_note}",
            fg=label_color
        )
        
        # Show what the network sees
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        
        # Original drawing
        ax1.imshow(self.image, cmap='gray')
        ax1.set_title('Your Drawing')
        ax1.axis('off')
        
        # Processed (what network sees)
        ax2.imshow(processed_image, cmap='gray')
        ax2.set_title(f'Network Input (28x28)\nPrediction: {prediction}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show all output activations
        self.show_all_outputs(output)
    
    def show_all_outputs(self, output):
        """Show the activation of all output neurons."""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        digits = range(10)
        activations = [output[i][0] for i in digits]
        
        colors = ['green' if act == max(activations) else 'blue' for act in activations]
        bars = ax.bar(digits, activations, color=colors, alpha=0.7)
        
        ax.set_xlabel('Digit', fontsize=12)
        ax.set_ylabel('Activation', fontsize=12)
        ax.set_title('Network Output Activations for Each Digit', fontsize=14)
        ax.set_xticks(digits)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, act in zip(bars, activations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{act:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def run(self):
        """Start the drawing interface."""
        self.window.mainloop()


def choose_default_network_file():
    """Return the strongest saved model available in the project."""
    candidates = [
        "trained_network_improved.pkl",
        "test_network1",
        "trained_network.pkl",
        "trained_network_30.pkl",
    ]

    for filename in candidates:
        if os.path.exists(filename):
            return filename

    return "trained_network.pkl"


def train_or_load_network():
    print("\n" + "=" * 60)
    print("Neural Network Setup")
    print("=" * 60)

    print("\n1. Load existing trained network")
    print("2. Train new network and save it")

    choice = input("\nChoice (1/2): ").strip()

    if choice == '1':
        default_filename = choose_default_network_file()
        filename = input(f"Enter filename (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename

        try:
            net = save_load_network.load_network(filename)

            return net
        except FileNotFoundError:
            print("No saved network found. Training a new one instead...\n")

    # Train new network
    print("\nLoading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    epoch_input = input("Enter number of epochs (default: 12): ").strip()
    epochs = int(epoch_input) if epoch_input else 12

    # This remains a fully connected network, but it is kept compact enough to
    # train in a reasonable time with this handwritten NumPy implementation.
    print("Creating neural network [784, 128, 64, 10] with ReLU hidden layers...")
    net = network.Network(
        [784, 128, 64, 10],
        hidden_activation="relu",
        output_activation="sigmoid",
        cost="cross_entropy",
    )

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    summary = net.SGD(
        training_data,
        epochs,
        20,
        0.2,
        test_data=test_data,
        validation_data=validation_data,
        lmbda=0.0001,
    )
    print("-" * 60)

    if summary["best_epoch"] is not None:
        best_validation_pct = 100.0 * summary["best_validation_correct"] / len(validation_data)
        print(
            f"Best validation checkpoint: epoch {summary['best_epoch']} "
            f"({summary['best_validation_correct']} / {len(validation_data)} = {best_validation_pct:.2f}%)"
        )

    # Save after training
    save_choice = input("\nSave this network? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_mode = "f"
        if summary["best_weights"] is not None:
            save_mode = input("Save best checkpoint or final model? (b/f, default: b): ").strip().lower()
            if not save_mode:
                save_mode = "b"

        if save_mode == "b" and summary["best_weights"] is not None:
            net.weights = [w.copy() for w in summary["best_weights"]]
            net.biases = [b.copy() for b in summary["best_biases"]]

        filename = input("Filename (default: trained_network_improved.pkl): ").strip()
        if not filename:
            filename = "trained_network_improved.pkl"
        save_load_network.save_network(net, filename)

    return net

if __name__ == "__main__":
    print("=" * 60)
    print("Draw Your Own Digit - Neural Network Tester")
    print("=" * 60)
    
    try:
        # Train or load network
        net = train_or_load_network()
        
        print("\n✓ Network ready!")
        print("\nOpening drawing window...")
        print("Instructions:")
        print("  - Draw a digit (0-9) with your mouse")
        print("  - Try to make it large and centered")
        print("  - Click 'Predict' to see what the network thinks")
        print("  - Click 'Clear' to draw again")
        print("\n" + "=" * 60)
        
        # Start drawing interface
        app = DigitDrawer(net)
        app.run()
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
