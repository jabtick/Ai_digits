# 🚀 Getting Started with Neural Network Digit Recognition

## Complete Setup in 5 Minutes

### Step 1: Install Required Software

First, make sure you have Python 3 installed:
```bash
python --version  # Should show Python 3.x
```

Install required packages:
```bash
pip install numpy matplotlib pillow
```

### Step 2: Get the Code and Data

Create a project folder:
```bash
mkdir digit-recognition
cd digit-recognition
```

Download the files I provided:
- network.py
- mnist_loader.py
- test_network.py
- quick_test.py
- draw_and_predict.py
- view_mnist_digits.py
- save_load_network.py

Get the MNIST data:
```bash
mkdir data
cd data
# Download from: https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz
# Or use wget:
wget https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz
cd ..
```

### Step 3: Verify Everything Works

Run the quick test:
```bash
python quick_test.py
```

You should see:
```
✓ Training data: 50000 examples
✓ Validation data: 10000 examples
✓ Test data: 10000 examples
...
Epoch 0: 9075 / 10000
...
All tests passed! ✓
```

---

## 🎨 Now The Fun Part - What You Can Do!

### 1️⃣ View MNIST Training Images

See what the network is learning from:
```bash
python view_mnist_digits.py
```

This will show you random digits from the dataset.

### 2️⃣ Train Your Own Network

Full training (10-15 minutes, ~95% accuracy):
```bash
python test_network.py
```

Or edit the file to train with different parameters:
```python
net = network.Network([784, 30, 10])  # Try [784, 100, 10] for better accuracy
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```

### 3️⃣ **Draw Your Own Digits!** (The Cool Part)

This is what you want!
```bash
python draw_and_predict.py
```

This will:
1. Train a network (or you can skip to quick test mode)
2. Open a drawing window
3. Let you draw digits with your mouse
4. Show you what the network predicts!

**How to use the drawing interface:**
- Draw a digit (0-9) with your mouse
- Make it large and centered
- Click "Predict" to see the result
- Click "Clear" to draw another

You'll see:
- Your drawing
- What the network "sees" (28x28 processed image)
- The prediction and confidence
- A bar chart of all output neuron activations

### 4️⃣ Save Your Trained Network

Don't want to wait 10 minutes every time? Save it!
```bash
python save_load_network.py
```

Options:
1. Train and save a network (30 epochs)
2. Quick train and save (5 epochs)
3. Load and test a saved network
4. Load and show predictions

Example workflow:
```python
# Save a network
python save_load_network.py
# Choose option 1 (train 30 epochs)
# Wait ~10 minutes
# Network saved to 'trained_network_30.pkl'

# Later, use it instantly:
from save_load_network import load_network
net = load_network('trained_network_30.pkl')
# Now you can use net immediately!
```

---

## 📊 Understanding the Results

### What do the numbers mean?

When training:
```
Epoch 0: 9129 / 10000
```
- Epoch 0 = First pass through training data
- 9129 / 10000 = Got 9,129 correct out of 10,000 test images
- That's 91.29% accuracy!

By Epoch 29, you should see ~95% accuracy.

### What makes a good digit drawing?

For best results when drawing:
- ✅ **Large** - Fill most of the canvas
- ✅ **Centered** - Middle of the canvas
- ✅ **Clear** - Thick, bold lines
- ✅ **One stroke** - Connect all parts
- ❌ Too small
- ❌ In the corner
- ❌ Very thin lines

---

## 🎯 Quick Reference

### Run Scripts

| Script | What it does | Time |
|--------|-------------|------|
| `quick_test.py` | Verify setup works | 30 sec |
| `view_mnist_digits.py` | See training data | 5 sec |
| `test_network.py` | Full training | 10 min |
| `draw_and_predict.py` | ⭐ Draw & test digits | 1-10 min |
| `save_load_network.py` | Save/load networks | varies |

### Common Commands

```bash
# Quick verification
python quick_test.py

# See what the network learns from
python view_mnist_digits.py

# Train a network (be patient!)
python test_network.py

# The fun part - draw your own digits!
python draw_and_predict.py
```

---

## 🐛 Troubleshooting

### "No module named numpy"
```bash
pip install numpy matplotlib pillow
```

### "No such file or directory: 'data/mnist.pkl.gz'"
Make sure you have the data file in a `data/` folder:
```bash
mkdir data
cd data
wget https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz
```

### Drawing window doesn't open
Make sure you have tkinter installed:
```bash
# On Ubuntu/Debian:
sudo apt-get install python3-tk

# On macOS:
brew install python-tk

# On Windows: Usually included with Python
```

### Network accuracy is low
- Make sure you trained for at least 10 epochs
- Try more hidden neurons: `Network([784, 100, 10])`
- Try different learning rates: `net.SGD(..., eta=1.0)`

---

## 🎓 Next Steps

Once you're comfortable:
1. **Experiment with architecture**: Try different numbers of hidden neurons
2. **Adjust hyperparameters**: Change learning rate, mini-batch size, epochs
3. **Visualize weights**: See what patterns each neuron learned
4. **Try other datasets**: CIFAR-10, Fashion-MNIST, etc.
5. **Build a web interface**: Use Flask to make it accessible online

---

## 📚 Learning More

- **Book**: http://neuralnetworksanddeeplearning.com/
  - Chapter 1: This code! 
  - Chapter 2: How backpropagation works
  - Chapter 3: Improving the network
  
- **Modern alternatives**: 
  - TensorFlow/Keras
  - PyTorch
  - But this simple version helps you understand the fundamentals!

---

## ✅ Your First Session Checklist

- [ ] Installed numpy, matplotlib, pillow
- [ ] Downloaded all .py files
- [ ] Got mnist.pkl.gz in data/ folder
- [ ] Ran quick_test.py successfully
- [ ] Viewed sample MNIST digits
- [ ] Trained a network (at least 3 epochs)
- [ ] Drew your own digit and got a prediction!

**Have fun experimenting! 🎉**
