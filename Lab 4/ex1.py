
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
    """
    
    dw = alpha * x * E
    print('dw=', dw)
    
    """
    
    """
    
    dw = np.sum(dw, axis = 0)
    print('dw=', dw)
    w = w + dw
    
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