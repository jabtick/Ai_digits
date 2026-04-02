"""
draw_and_predict_double.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Draw a two-digit number and test it with the trained neural network.

Usage:
    python draw_and_predict_double.py

This will:
1. Train a neural network (or load a saved one)
2. Open a drawing window split into two drawing zones
3. Let you draw one digit on each side
4. Show the combined two-digit prediction
"""

import os
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import mnist_loader
import network
import save_load_network

CANVAS_SIZE = 280
DOUBLE_CANVAS_WIDTH = CANVAS_SIZE * 2
MNIST_IMAGE_SIZE = 28
TARGET_DIGIT_SIZE = 20
DRAW_THRESHOLD = 0.10
LOW_CONFIDENCE_THRESHOLD = 75.0


class DigitDrawer:
    def __init__(self, net):
        self.net = net
        self.window = tk.Tk()
        self.window.title("Draw a Two-Digit Number (00-99)")

        # Create a wider canvas so each digit has its own drawing region.
        self.canvas_width = DOUBLE_CANVAS_WIDTH
        self.canvas_height = CANVAS_SIZE
        self.canvas = tk.Canvas(
            self.window,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            cursor="cross",
        )
        self.canvas.pack()

        # Store the drawn strokes in a grayscale image that matches the canvas size.
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self._draw_divider()

        # Bind mouse events.
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coords)
        self.old_x = None
        self.old_y = None

        button_frame = tk.Frame(self.window)
        button_frame.pack()

        predict_btn = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict,
            bg="green",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=20,
        )
        predict_btn.pack(side=tk.LEFT, padx=5, pady=10)

        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear,
            bg="red",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=20,
        )
        clear_btn.pack(side=tk.LEFT, padx=5, pady=10)

        self.result_label = tk.Label(
            self.window,
            text="Draw two digits and click Predict",
            font=("Arial", 16, "bold"),
            fg="blue",
        )
        self.result_label.pack(pady=10)

        instructions = tk.Label(
            self.window,
            text="Draw one digit on the left and one digit on the right.",
            font=("Arial", 10),
            fg="gray",
        )
        instructions.pack()

    def _draw_divider(self):
        """Draw the visual guide that splits the board into two halves."""
        divider_x = CANVAS_SIZE
        self.canvas.create_line(
            divider_x,
            0,
            divider_x,
            self.canvas_height,
            fill="gray",
            width=3,
            dash=(8, 6),
        )
        self.canvas.create_text(
            CANVAS_SIZE // 2,
            20,
            text="Left digit",
            fill="gray",
            font=("Arial", 11, "bold"),
        )
        self.canvas.create_text(
            CANVAS_SIZE + (CANVAS_SIZE // 2),
            20,
            text="Right digit",
            fill="gray",
            font=("Arial", 11, "bold"),
        )

    def paint(self, event):
        """Draw on canvas."""
        paint_width = 20
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=paint_width,
                fill="black",
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
            )
            self.draw.line(
                [self.old_x, self.old_y, event.x, event.y],
                fill="black",
                width=paint_width,
            )
        self.old_x = event.x
        self.old_y = event.y

    def reset_coords(self, event):
        """Reset drawing coordinates."""
        self.old_x = None
        self.old_y = None

    def clear(self):
        """Clear the canvas and redraw the divider."""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self._draw_divider()
        self.result_label.config(
            text="Draw two digits and click Predict",
            fg="blue",
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

    def _preprocess_single_digit(self, digit_image):
        """Convert one half of the board into a MNIST-like 28x28 image."""
        img_array = 1.0 - (np.array(digit_image, dtype=np.float32) / 255.0)
        coords = np.argwhere(img_array > DRAW_THRESHOLD)

        if coords.size == 0:
            blank = np.zeros((MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), dtype=np.float32)
            return blank.reshape(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE, 1), blank

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        pad = 20
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(img_array.shape[0] - 1, y_max + pad)
        x_max = min(img_array.shape[1] - 1, x_max + pad)

        cropped = digit_image.crop((x_min, y_min, x_max + 1, y_max + 1))
        cropped = cropped.filter(ImageFilter.GaussianBlur(radius=0.6))

        width, height = cropped.size
        scale = TARGET_DIGIT_SIZE / max(width, height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cropped.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        mnist_canvas = Image.new("L", (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), "white")
        offset_x = (MNIST_IMAGE_SIZE - resized_width) // 2
        offset_y = (MNIST_IMAGE_SIZE - resized_height) // 2
        mnist_canvas.paste(resized, (offset_x, offset_y))

        processed = 1.0 - (np.array(mnist_canvas, dtype=np.float32) / 255.0)
        processed[processed < 0.05] = 0.0
        processed = np.clip(processed * 1.15, 0.0, 1.0)
        processed = self._shift_to_center_of_mass(processed)

        img_vector = processed.reshape(MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE, 1)
        return img_vector, processed.reshape(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)

    def preprocess_image(self):
        """Split the board into left and right halves and preprocess both."""
        left_image = self.image.crop((0, 0, CANVAS_SIZE, CANVAS_SIZE))
        right_image = self.image.crop((CANVAS_SIZE, 0, DOUBLE_CANVAS_WIDTH, CANVAS_SIZE))

        left_vector, left_processed = self._preprocess_single_digit(left_image)
        right_vector, right_processed = self._preprocess_single_digit(right_image)
        return (left_vector, left_processed), (right_vector, right_processed)

    def predict(self):
        """Predict the left and right digits with the same network and join them."""
        (left_vector, left_processed), (right_vector, right_processed) = self.preprocess_image()

        left_output = self.net.feedforward(left_vector)
        right_output = self.net.feedforward(right_vector)

        left_prediction = int(np.argmax(left_output))
        right_prediction = int(np.argmax(right_output))

        left_confidence = float(left_output[left_prediction][0] * 100)
        right_confidence = float(right_output[right_prediction][0] * 100)
        average_confidence = (left_confidence + right_confidence) / 2.0

        combined_prediction = f"{left_prediction}{right_prediction}"

        confidence_note = ""
        label_color = "green"
        if average_confidence < LOW_CONFIDENCE_THRESHOLD:
            confidence_note = " - low confidence, try drawing both digits larger/cleaner"
            label_color = "dark orange"

        self.result_label.config(
            text=(
                f"Prediction: {combined_prediction} "
                f"(Confidence: {average_confidence:.1f}%){confidence_note}"
            ),
            fg=label_color,
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(self.image, cmap="gray")
        axes[0].set_title("Your Drawing")
        axes[0].axis("off")

        axes[1].imshow(left_processed, cmap="gray")
        axes[1].set_title("Left Network Input (28x28)")
        axes[1].axis("off")

        axes[2].imshow(right_processed, cmap="gray")
        axes[2].set_title("Right Network Input (28x28)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

        self.show_all_outputs(left_output, right_output)

    def show_all_outputs(self, left_output, right_output):
        """Show output activations for the left and right digit predictions."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        digits = range(10)

        panels = [
            (axes[0], left_output, "Left Digit Activations"),
            (axes[1], right_output, "Right Digit Activations"),
        ]

        for ax, output, title in panels:
            activations = [output[i][0] for i in digits]
            colors = ["green" if act == max(activations) else "blue" for act in activations]
            bars = ax.bar(digits, activations, color=colors, alpha=0.7)

            ax.set_xlabel("Digit", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_xticks(list(digits))
            ax.grid(axis="y", alpha=0.3)

            for bar, act in zip(bars, activations):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{act:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        axes[0].set_ylabel("Activation", fontsize=12)
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

    if choice == "1":
        default_filename = choose_default_network_file()
        filename = input(f"Enter filename (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename

        try:
            net = save_load_network.load_network(filename)
            return net
        except FileNotFoundError:
            print("No saved network found. Training a new one instead...\n")

    print("\nLoading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    epoch_input = input("Enter number of epochs (default: 12): ").strip()
    epochs = int(epoch_input) if epoch_input else 12

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

    save_choice = input("\nSave this network? (y/n): ").strip().lower()
    if save_choice == "y":
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
    print("Draw Your Own Two-Digit Number - Neural Network Tester")
    print("=" * 60)

    try:
        net = train_or_load_network()

        print("\nNetwork ready!")
        print("\nOpening drawing window...")
        print("Instructions:")
        print("  - Draw the left digit in the left half")
        print("  - Draw the right digit in the right half")
        print("  - Click 'Predict' to see the combined two-digit guess")
        print("  - Click 'Clear' to draw again")
        print("\n" + "=" * 60)

        app = DigitDrawer(net)
        app.run()

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
