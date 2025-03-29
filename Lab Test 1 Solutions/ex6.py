"""


robot that decides whether to charge its battery based on two sensor inputs
"""

import numpy as np

x = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]] # Input data for sensor A and B
T = np.array([[1], [0], [0], [0], [0], [0]])  #expected output

no_samples = np.size(x, 0)  # no. of samples = 6 rows
no_weights = np.size(x, 1) + 1  # no. of features + 1 (for bias) = 2 + 1 = 3

# Extend input with bias term (X[:,2] = 1), so we have a column of 1s for bias too
X = np.ones([no_samples, no_weights])  
X[:, :-1] = np.array(x)  # Copy x values but keep the last column as 1 (bias)

#the starting weights for sensor A, sensor B and bias
W = np.array([[1], [-1], [1]])


error = 1 

# Update weights until global error is zero
while error != 0:
    error = 0
    for sample in range(no_samples): # For each sample update the weights (if necessary) and find the accumulated global error at the end
        n = np.dot(X[sample, :], W)  # weghted sum
        y = n >= 0  # Activation function of perceptron if n >= 0 then y = 1 if not then y = 0
        e = T[sample] - y  # The error of the output for this  sample
        dW = np.reshape(X[sample, :] * e, [no_weights, 1])  # Compute the change in the weights, which is proportional to the input and the error
        W = W + dW  # Update the weights
        error += np.abs(e)  # Accumulate the global error for this iteration, use abs because some errors can be negative but we only care about magnitude
        
print("Final Weights:\n", W)
print("Predictions:", (np.dot(X, W) >= 0).astype(int).flatten())