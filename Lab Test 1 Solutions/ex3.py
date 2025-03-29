"""
ex3

Fighter A
"""

import numpy as np

x = np.array( [ [1,0], [2,0], [3,0], [0,1], [0,2] ] ) # the three unches and the two kicks

T = np.array( [ [0],[0],[1],[0],[1] ] ) # the expected output

W = np.array([ [1],[2],[2] ])

no_samples=np.size(x,0) # => 5 samples
no_weights=np.size(x,1)+1 # 2 features + 1 for bias


X=np.ones([no_samples,no_weights]) # generate a matrix of ones of shape 5x3
X[:,:-1]=x # [all lines, all columns except the last one] = [5, 2] (the matrix of sampled features)



def weightedSum(W,X): #function for the weighted sum
    N=np.dot(X,W)  # P*wp + K*wk + b = n >= 0
    return(N>=0)



error=1 

# Perceptron keeps updating the weights until the global error is zero
while error!=0:
    error=0
    for sample in range(no_samples):
        n=np.dot(X[sample,:],W) #the weighted sum for one sample at a time
        y=n>=0
        e=T[sample]-y # Possible values: 0 → Correct classification (no update needed), 1 → False negative (increase weights), -1 → False positive (decrease weights)
        dW=np.reshape(X[sample,:]*e,[no_weights,1]) # computes the weight update in the perceptron algorithm and reshapes it into a column vector
        W=W+dW # update the weights
        print(W) # print updated weights on each iteration
        print('-------------')
        error=error+np.abs(e) # Accumulate global error (take abs value of local error, because we care about magnitudes only)
    print('===============')
    
print(weightedSum(W,X)) #returns true or false on each line that it computes the sum on