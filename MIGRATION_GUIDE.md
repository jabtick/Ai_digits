# Python 2 to Python 3 Migration Guide

## Detailed List of Changes

### 1. network.py

#### Import Statements
No changes needed - `random` and `numpy` work the same in both versions.

#### Line 56: xrange → range
```python
# Python 2.7
for j in xrange(epochs):

# Python 3
for j in range(epochs):
```

#### Line 60: xrange → range
```python
# Python 2.7
for k in xrange(0, n, mini_batch_size)

# Python 3
for k in range(0, n, mini_batch_size)
```

#### Line 64-67: print statement → print function
```python
# Python 2.7
print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
print "Epoch {0} complete".format(j)

# Python 3
print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
print("Epoch {0} complete".format(j))
```

#### Line 112: xrange → range
```python
# Python 2.7
for l in xrange(2, self.num_layers):

# Python 3
for l in range(2, self.num_layers):
```

#### New Lines 53-58: Converting zip iterators to lists
```python
# Python 3 addition (needed because SGD method shuffles the data)
training_data = list(training_data)
n = len(training_data)

if test_data:
    test_data = list(test_data)
    n_test = len(test_data)
```

---

### 2. mnist_loader.py

#### Line 13: cPickle → pickle
```python
# Python 2.7
import cPickle

# Python 3
import pickle
```

#### Line 43: pickle.load with encoding
```python
# Python 2.7
training_data, validation_data, test_data = cPickle.load(f)

# Python 3
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
```

**Note:** The `encoding='latin1'` parameter is crucial for reading pickled files 
that were created with Python 2. Without it, you'll get decoding errors.

#### Lines 71, 73, 75: zip → list(zip())
```python
# Python 2.7
training_data = zip(training_inputs, training_results)
validation_data = zip(validation_inputs, va_d[1])
test_data = zip(test_inputs, te_d[1])

# Python 3
training_data = list(zip(training_inputs, training_results))
validation_data = list(zip(validation_inputs, va_d[1]))
test_data = list(zip(test_inputs, te_d[1]))
```

**Why?** In Python 3, `zip()` returns an iterator, not a list. We need actual 
lists because:
1. The data gets shuffled in the SGD method
2. We need to calculate `len()` on the data
3. The data might be iterated multiple times

---

## Why These Changes Were Necessary

### xrange → range
- Python 2 had two functions: `range()` (creates list) and `xrange()` (creates iterator)
- Python 3 removed `xrange()` and made `range()` return an iterator
- The new `range()` behaves like the old `xrange()`

### Print Function
- Python 2: `print` was a statement
- Python 3: `print()` is a function
- Requires parentheses in Python 3

### cPickle → pickle
- Python 2 had `pickle` (slow, in Python) and `cPickle` (fast, in C)
- Python 3 merged them: the `pickle` module automatically uses the C implementation
- `cPickle` doesn't exist in Python 3

### pickle.load encoding
- Python 2 pickled strings as bytes by default
- Python 3 distinguishes between bytes and strings
- Need to specify encoding when loading Python 2 pickled files

### zip() Returns Iterator
- Python 2: `zip()` returned a list
- Python 3: `zip()` returns an iterator (more memory efficient)
- If you need a list (for shuffling, len(), or multiple iterations), wrap with `list()`

---

## Testing the Migration

Run the test script to verify everything works:

```bash
python test_network.py
```

Expected behavior:
- No syntax errors
- Network trains successfully
- Achieves ~95% accuracy after 30 epochs

## Common Issues and Solutions

### Issue 1: "NameError: name 'xrange' is not defined"
**Solution:** Replace `xrange` with `range`

### Issue 2: "TypeError: 'zip' object is not subscriptable"
**Solution:** Wrap `zip()` calls with `list()`

### Issue 3: "UnicodeDecodeError" when loading pickle file
**Solution:** Add `encoding='latin1'` to `pickle.load()`

### Issue 4: "SyntaxError: invalid syntax" on print statements
**Solution:** Change `print "text"` to `print("text")`

### Issue 5: "ModuleNotFoundError: No module named 'cPickle'"
**Solution:** Change `import cPickle` to `import pickle`
