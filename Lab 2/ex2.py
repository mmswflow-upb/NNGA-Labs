"""
Perceptron learning algorithm implemented with vectorized NumPy operations
instead of explicit Python loops over individual samples.
"""

import numpy as np

# Target labels for 5 training examples (0 or 1), shape (5,1) or 5x1
T = np.array([0, 0, 1, 0, 1])

# Input feature matrix: each row = one example, each column = a feature, it's of order 5 x 2
x = np.array([
    [1, 0],
    [2, 0],
    [3, 0],
    [0, 1],
    [0, 2]
])

# Add a bias column of ones to X: np.ones((n,1)) creates a (5×1) column of 1s
a = np.ones((len(x), 1))
# np.concatenate(..., axis=1) stacks the bias as a third column → X has shape (5,3)
X = np.concatenate((x, a), axis=1)

# Initialize weight vector (feature1 weight, feature2 weight, bias weight), its shape is (3,)
w = np.array([1, 3, 2])

# Perceptron update loop: repeats until there are no classification errors
while True:
    # Compute raw linear outputs: X.dot(w) → shape (5,), one value per sample, w is treated as a column vector in this dot product
    # So result is of shape 5x1
    n = X.dot(w) 

    # Step activation: (n >= 0) yields a Boolean array, .astype(int) converts True→1, False→0
    # So for each sample or row in n, it gives a new row with one element representing the class of the sample which was on that row
    y = (n >= 0).astype(int) # this is still 5x1

    # Error vector for all examples: target minus predicted (shape (5,))
    e = T - y
    print('e =', e)

    """
    Illustration of np.tile(e, (X.shape[1], 1)):
    
    Suppose X has shape (5, 3) — 5 samples and 3 weights (2 features + bias):
    
        X = [[1, 0, 1],
             [2, 0, 1],
             [3, 0, 1],
             [0, 1, 1],
             [0, 2, 1]]
    
    And the error vector for these 5 samples is:
    
        e = np.array([ 1, -1, 2,  0, -2 ])   # shape (5,)
    
    We want E to have the same number of columns as X (5) and the same number of rows as there are weights (3), so each weight update aligns with each feature across all samples.
    
    The tuple `(X.shape[1], 1)` passed to `np.tile` specifies:
    
        • First element (X.shape[1] == 3): repeat e into 3 rows  
        • Second element (1): do not duplicate columns (keep the original 5 elements in each row) (gives how many times to copy into columns, in this case only once per row)
    
    Resulting E has shape (3, 5):
    
        E = [[ 1, -1,  2,  0, -2],
             [ 1, -1,  2,  0, -2],
             [ 1, -1,  2,  0, -2]]
    
    
    If you call np.tile(e, (X.shape[1],)) instead of np.tile(e, (X.shape[1], 1)), NumPy treats the tuple as specifying only the number of repeats — not the shape of each axis. 

    Given:
        e = np.array([ 1, -1, 2,  0, -2 ])   # shape (5,)
        X.shape[1] == 3
    
    Then:
    
        E_wrong = np.tile(e, (X.shape[1],))
        print(E_wrong.shape)   # → (15,)
        print(E_wrong)
        # [ 1 -1  2  0 -2  1 -1  2  0 -2  1 -1  2  0 -2]
    
    Instead of a 2‑D (3×5) array, you get a single 1‑D array of length 15 — e repeated end‑to‑end three times. That isn’t aligned for elementwise multiplication with X. 
    
    Later we transpose E to shape (5, 3) so elementwise multiplication with X produces per-sample, per-weight contributions for the update.
    """

    E = np.tile(e, (X.shape[1], 1))
    print('E =', E)

    # Global error: sum of absolute errors across all elements (scalar)
    eGlobal = np.sum(np.abs(E))
    print('\nGlobal error now is...', eGlobal)

    # Stop when there are zero errors
    if eGlobal == 0:
        break

    # Transpose E back to shape (5,3) so it aligns elementwise with X
    E = E.T

    # Weight update: elementwise multiply X and E → shape (5,3), then sum rows (axis=0)
    dw = np.sum(X * E, axis=0) # So here we get a shape (3,), each entry containing the accumulated changes in the weight that must be made for ith feature

    # Update weight vector
    w += dw

print('w =', w)
