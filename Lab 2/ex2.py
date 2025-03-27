"""
Perceptron algorithm again, but implemented using matrices instead of 
going through each element one by one with a for loop
"""

import numpy as np

T = np.array([0,0,1,0,1])
x = np.array([ [1,0], [2,0], [3,0], [0,1], [0,2] ])
a = np.array([ np.ones(len(x)) ])
X = np.concatenate((x,a.T), axis= 1)
w = np.array([1,3,2])

#perceptron learning algorithm
while 1:
    n = np.matmul(X,w) #matrix multiplication
    
    y = (n>=0) * 1 # gets converted directly to 0 or 1 from boolean
    
    #e = T[i] - y
    e = T - y #doing it directly because we are working with arrays
    print('e = ', e)
    
    #matrix of errors
    #np.tile(v, num) requires a vector or an array and a number
    #It repeats v the amount of num times
    #np.size(X) gets the size of the matrix 
    E = np.tile(e, ( np.size(X,1), 1 ) )
    print(' E = ', E)
    
    #its the sum of individual errors to quantify how big the error is
    #its for describing the general error considering all cases
    eGlobal = np.sum(np.abs(E))
    print('\n\nGlobal error now is...', eGlobal)
    
    if eGlobal == 0:
        break
    
    #dw = X[i] * e
    #due to matrix multiplication rules we need to transpose the error matrix
    E = E.T
    dw = X * E
    dw = sum(dw)
    w = w + dw
    
print('w = ', w)
    
    