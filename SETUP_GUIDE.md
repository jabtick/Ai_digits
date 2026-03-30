# Complete Setup Guide - Neural Network Digit Recognition

## Step 1: Create Your Project Folder

```bash
# Create a new folder for your project
mkdir mnist-neural-network
cd mnist-neural-network
```

## Step 2: Get the Files

You have two options:

### Option A: Use the files I provided (Recommended)
1. Download the 6 files I created:
   - network.py
   - mnist_loader.py
   - test_network.py
   - quick_test.py
   - README.md
   - MIGRATION_GUIDE.md

2. Put them all in your `mnist-neural-network` folder

### Option B: Clone the original repo and use my updated files
```bash
git clone https://github.com/mnielsen/neural-networks-and-deep-learning.git
# Then replace the .py files with my Python 3 versions
```

## Step 3: Get the MNIST Data

You need the training data file. Here's how:

```bash
# If you cloned the repo in Option B:
cp neural-networks-and-deep-learning/data/mnist.pkl.gz ./data/

# OR download directly:
mkdir data
cd data
wget https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz
cd ..
```

## Step 4: Install Dependencies

```bash
pip install numpy
pip install matplotlib  # For visualizing digits
pip install pillow      # For drawing your own digits
```

## Step 5: Test It Works

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
Epoch 1: 9217 / 10000
Epoch 2: 9262 / 10000
...
All tests passed! ✓
```

## Your Project Structure Should Look Like:

```
mnist-neural-network/
├── data/
│   └── mnist.pkl.gz          # The MNIST dataset
├── network.py                # Neural network code
├── mnist_loader.py           # Data loader
├── test_network.py           # Full training script
├── quick_test.py             # Quick validation
├── README.md                 # Documentation
├── MIGRATION_GUIDE.md        # Migration details
└── my_digit_tester.py        # (We'll create this next!)
```

## Next Steps

Once this is working, I'll show you how to:
1. **Draw your own digits** and test them
2. **Visualize** what the network learned
3. **Save and load** trained networks
4. **Create a simple GUI** for drawing

Ready to continue? Let me know!
