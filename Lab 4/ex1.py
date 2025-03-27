
"""
Batch gradient descent, it applies widrow-hoff on 
a batch of inputs, so there's only one weight for the
whole set of inputs and one weight for the bias
"""


import numpy as np
import matplotlib.pyplot as plt

x = np.array([ [1], [0], [2] ])

"""
Inputs x (one feature) and a constant bias term are concatenated into a 3×2 matrix:
    |1 1|
x = |0 1|
    |2 1|
"""
x = np.concatenate( (x, np.ones([len(x),1])), axis = 1)

"""
The true label vector (target values)

    |0|
T = |2|
    |2|
"""

T = np.array([ [0], [2], [2] ])

"""
in this case we only have 2 weights, we're doing batch training on the whole set of inputs with one weight, 
and one weight is for the bias

w = |1|
    |2|


Learning rate is 0.1
Number of epochs is 30 (run this algorithm 30 times)
Smallest allowed error is 1e-3 (0.001)
"""
w = np.array([ [1], [2] ])
alpha = .1

maxE = 30
eps = 10 ** (-3) # 10 ** (-3) = 1e-3


for i in range (maxE):
    """
    define Y = x * w
    
    x: 3x2, w: 2x1 , Y: 3x1 (3 outputs in one column vector) after multiplying x with w
    
    """
    Y = np.matmul(x, w)
    """
    defining E = T - Y (difference between target values and computed ones 'Y' )
    
        |e1|
    E = |e2|
        |e3|
    """
    E = T - Y
    
    """
    Update rule: dw = learning rate x input vector x error vector
    x: 3x2 , E: 3x1, dw: 3x2 , how?
    Well, numpy broadcasts E's single column across the 2 columns of x
    so you get:
        
        
                            |x1*e1 b1*e1|
    alpha * x * E = alpha * |x2*e2 b2*e2|
                            |x3*e3 b3*e3|
                            
    Now we have weight updates proportional to the learning rate and the inputs,
    but we need to apply them on 2 weights no more, so how do we do that?
    
    Well we want to find the accumulated change in the two weights, each column in the previously mentioend matrix
    represents the weight changes which were supposed to be applied on the individual weights, if we wouldn't have had batch
    weighting (for each input there would be a weight like before), so now we need to sum up these individual input weights into
    one batch weight, same goes for the bias weights
    """
    
    dw = alpha * x * E
    print('dw=', dw)
    
    """
    This is what np.sum does here, it sums in each column the numbers stored on each row, so the matrix collapses to a 1x2 matrix
    or a row vector of 2 columns, each column is the change in one of the two weights (batch and bias weight changes)
    
    Suppose you had this:
    
         |dw1 db1|   |0.2 0.1|
    dw = |dw2 db2| = |0.0 0.2| --> applying that sum with axis = 0 (sum elements on each row) --> dw =  |(0.2 + 0.0 + 0.4)  (0.1 + 0.2 + 0.1)|
         |dw3 db3|   |0.4 0.1|
    
    ==> dw = |0.6 0.4|
    """
    
    dw = np.sum(dw, axis = 0)
    print('dw=', dw)
    w = w + dw # here we just apply the change to the weights vector
    
    """
    E**2 just squares the matrix element-wise (each element in the matrix is squared) the matrix doesnt get multiplied with itself
    then we sum all elements and divide by 2 (LMS function), and we get eGlobal, 
    which is the computed LMS (Least Mean Square) loss (total error squared)
    we compare it with minimum allowed error, if it's under it then we stop, if not we keep going until we 
    reach max num of epochs (maxE)
    """
    
    eGlobal = 0.5 * np.sum(E**2)
    print('Global error: ', eGlobal)
    
    if eGlobal < eps:
        break
    
print('=========================================')
print('w=', w)

plt.close('all')
plt.plot(x[:,0], T, 'og')
plt.plot(x[:,0], Y, 'd-r')
plt.grid('on')