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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageDraw
import network
import mnist_loader

class DigitDrawer:
    def __init__(self, net):
        self.net = net
        self.window = tk.Tk()
        self.window.title("Draw a Digit (0-9)")
        
        # Create canvas for drawing
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(
            self.window, 
            width=self.canvas_width, 
            height=self.canvas_height,
            bg='white',
            cursor='cross'
        )
        self.canvas.pack()
        
        # Create image for drawing
        self.image = Image.new('L', (280, 280), 'white')
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
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(
            text="Draw a digit and click Predict",
            fg='blue'
        )
    
    def preprocess_image(self):
        """Convert drawn image to MNIST format (28x28, inverted)."""
        # Resize to 28x28
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img_resized)
        
        # Invert (MNIST has white digits on black background)
        img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Reshape to (784, 1) for network
        img_vector = img_array.reshape(784, 1)
        
        return img_vector, img_array.reshape(28, 28)
    
    def predict(self):
        """Get network prediction for drawn digit."""
        # Preprocess the image
        input_vector, processed_image = self.preprocess_image()
        
        # Get network output
        output = self.net.feedforward(input_vector)
        prediction = np.argmax(output)
        confidence = output[prediction][0] * 100
        
        # Update label
        self.result_label.config(
            text=f"Prediction: {prediction} (Confidence: {confidence:.1f}%)",
            fg='green'
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


def train_or_load_network():
    """Train a new network or load an existing one."""
    print("\n" + "=" * 60)
    print("Neural Network Setup")
    print("=" * 60)
    num_of_epochs=0
    choice = input("\n1. Train a new network (3 epochs, ~1 minute)\n"
                  "2. Train for better accuracy (user selected epochs, ~10 minutes)\n"
                  "3. Quick test (use partially trained network)\n"
                  "\nChoice (1/2/3): ").strip()
    
    print("\nLoading MNIST data...")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    num_of_epochs = int(input("\nEnter number of epochs: "))
    print("Creating neural network [784, user selected epochs, 10]...")
    net = network.Network([784, num_of_epochs, 10])
    
    if choice == '2':
        print(f"\nTraining network for {num_of_epochs} epochs...")
        print("This will take several minutes. Please wait...")
        print("-" * 60)
        net.SGD(training_data, num_of_epochs, 10, 3.0, test_data=test_data)
        print("-" * 60)
    elif choice == '3':
        print("\nQuick test mode - network is untrained!")
        print("(Predictions will be random)")
    else:  # Default to 3 epochs
        print("\nTraining network for 3 epochs...")
        print("-" * 60)
        net.SGD(training_data, 3, 10, 3.0, test_data=test_data)
        print("-" * 60)
    
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
