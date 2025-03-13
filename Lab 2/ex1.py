"""
This program simulates the perceptron, a type of neuron that tries to learn 
how to group things into 'classes'
"""

import numpy as np

t = np.array([0,0,1,0,1])

"""
True label vector or 't'

not randomly picked, as we can see for the x axis, the threshold value seems to be 3, so x >= 3 will be classified as 
being part of class '1', the same for the y coords, the threshold value seems to be 2, so y>= 2 will be classified as
part of class '1', the perceptron will try to identify this pattern on each iteration trying to lower its global error
until it will be able to output the True label vector 
"""


#tuple = [param1, param2] 
X = np.array([ [1,0], [2,0], [3,0], [0,1], [0,2] ])

"""
X is the 4x2 matrix that contains the pairs of (x,y) coords to be classified by this perceptron

X = [
 [1., 0.] ,
 [2., 0.] ,
 [3., 0.] ,
 [0., 1.] ,
 [0., 2.] ]

There are cases when a perceptron can't classify a set of data, thats what we call linearly inseperable data, but in this
case we have linearly separable data, i.e you can draw a line and the points will be on either side of the line while having the correct class 
according to the true label vector (each side of the line defines the area of a class)

"""

a = np.array([ np.ones(len(X)) ]) #create an array populated with ones with the same num of rows as X

"""
'a' is the vector of biases, it gives freedom to the perceptron to make its approximated lines not
always pass through the origin, it is usually initialized with 1s
"""



"""
lower-case variables will most likely be arrays or vectors
upper-case variables are most likely matrices

We can concatenate vectors to create matrices
a.T gives you the transpose (from row to column vector in this case):

   
                   [1,
                    1,
 [1,1,1,1,1] ->     1,
                    1,
                    1]
"""
Xh = np.concatenate((X,a.T), axis= 1) # axis = 1 tells numpy that the concatenation happens horizontally (side-by-side, joining columns)

"""
Now the matrix Xh will look like this:

Xh = [
 [1., 0., 1.] ,
 [2., 0., 1.] ,
 [3., 0., 1.] ,
 [0., 1., 1.] ,
 [0., 2., 1.] ]

This allows us (just like in Computer graphics in 2D with homogenous coords) to apply transformations on this matrix
such as translation of origin in 2D, because we dont want the line to only pass through the origin.
In order to get the best approximation we have to make it able to pass through different points other than origin
"""

w = np.array([1,3,2])
"""
weight - how relevant are the values in the tuples for drawing the approximated line
they're three values because they correspond to 3 columns or "features", so for the x coord the weight is 1, for 
y coord the weight is 3, and for the bias or the 1's column the weight is 2

of course these are randomly picked values, the perceptron will have to change them as it tries to
lower the global error more and more until it reaches zero, but the idea of giving these starting values
to the perceptron is to create a somewhat neutral, unbiased starting point
"""



#Run till break out of loop (globalError becomes zero)
while 1:
    
    #global error
    eGlobal = 0
    
    #loop for each row in matrix Xh
    for i in range(len(Xh)):
        
        n = np.sum(Xh[i] * w) #dot product between ith row of Xh and weights vector
        # n = np.dot(Xh[i], w) equivalent of what is above
        
        """
        the core concept of the perceptron, the way it computes its output for a given input is the following: 
        n_i = dot_product(inp_i  w), inp_i is the vector of inputs, including the bias, that's an input vector because a perceptron might 
        have multiple lines or 'connections' that receive input values
        
        the structure of inp_i = (x_i -> ith x coord ,y_i -> ith y coord, b_i -> ith bias)
        
        this will result into: n_i = x_i * w1 + y_i * w2 + b_i * w3
       
        

        """
        
        if n>= 0:
            y = 1
        else:
            y = 0
    
        """

        Here we compute y based on the sum above, the perceptron is trying to figure out
        if 'n_i' is bigger than zero, then that means that the ith guess 'y' is 1, so the ith input's class is 1, else its class is zero
       
        But why specifically zero?
        
        normally, the equation of a line is y = a*x + b
        or it's: y/a - x -b/a = 0 -> y*a_prime + x*b_prime + c = 0 (c is the bias that allows shifting the line away from origin), 
        so 'n' is equal to this, then what? well, 'n' must also be equal to zero for this equation to properly define a line
        
        this line that the perceptron is trying to find is actually the boundary that separates the two classes
        of inputs, class 1 and 0, if a point (x,y) sits on one side of this boundary (or line), then it's in class 1, if 
        a point sits on the other, then it's class 0.
        
        conventionally, if a point sits right on the boundary, then it will be considered as being a member of group/class 1
        
        the perceptron is constantly trying to figure out the correct weights which will determine the boundary/line that makes 
        the inputs fall into the correct classes according to the True label vector or 't'
        
        why compute the dot product though? the dot product between the weights vector and the inputs vector tells us
        how aligned are the inputs with the weights, basically it's a measure of aligned are the inputs to a certain class
        and it tells us how far away are the inputs from the boundary separating the classes, the further away in positive direction (n > 0),
        the more it 'resembles' class 1, the further in negative direction (n < 0), the more it resembles class 0
        
        Thats why the weights are used, to determine the line that perfectly defines the areas of two classes, such that all inputs
        fall into the correct groups/sides/classes according to what we have defined in the true label vector
        
        """
        
        e = t[i] - y
        
        
        """
        Here we check the local error, aka the difference between the
        value of the true label vector at index i (associated to the ith row in X or ith pair of coords (x,y)) and 
        our current output y, the perceptron is trying to find the correct weights in order
        to have y always match our true label vector
        """
        
        
        eGlobal = eGlobal + abs(e)
        
        """
        The global error is the sum of all local errors, we have to take the absolute value of each local error, 
        because at some point we might get negative errors (the actual output 'y' is bigger than the expected one t[i])
        so we just take into consideration their magnitudes, not their signs, only how big they are
        """
        
        #d-variable = derivative of that function/variable
        #derivative of the weight
        dw = Xh[i] * e
        w = w + dw
    
        """
        Here we make changes to our weights, to correct the errors in our predictions
        
        If the perceptron guessed correctly (e = 0), then dw = 0 → no update needed.
        If the perceptron guessed 0 but the true class was 1 (e = 1), we increase w in the direction of X[i] (move towards the correct class).
        If the perceptron guessed 1 but the true class was 0 (e = -1), we decrease w in the direction of X[i] (move away from the incorrect class).
        
        """
    
   
    if eGlobal == 0:
        break
    """
    When the global error reaches zero, it means that the perceptron has managed to find
    the correct weights which allow it to correctly classify the pairs of coordinates fed into it
    according to the true label vector, so it stops running
    """
# print final results (final weights)
print('w = ', w)