import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as nn


"""
predict the XOR function, this can be done only if we have multiple layers of perceptrons, since it isn't
linearly separable...

x1       x2      XOR(x1, x2)
0        0            0
0        1            1
1        0            1
1        1            0



From this table we determine that the true label vector is:
          
XOR = T = [0, 1, 1, 0]  we didn't nest square brackets so its not even a row vector or a column vector, 
                        just a normal array
                        
!! each column is a feature or an input, i.e feature 1 = x1 ...etc and each row is a sample (set of inputs)

And the matrix of inputs is: 

column_T = np.array([[0],[1],[1],[0]]) this on the other hand would've been a column vector
row_T = np.array([ [0,1,1,0] ]) and this would've been a row vector

    |0 0|
x = |0 1|
    |1 0|
    |1 1|
"""

x = np.array([ [0,0], [0,1], [1,0], [1,1] ])
T = np.array([0,1,1,0])


"""
MLP (Multiple Layer Perceptron) Classifier

it creates a neural network of 2 layers with one hidden layer of 4 neurons in this case, and it makes
it run for 4000 times at most in order to find the optimal solution. Like mentioned before, for the XOR function
we need a hidden layer, not just one layer with 1 or 2 perceptrons..

the hidden_layer_sizes=() parameter takes a tuple of numbers, ith number 
in the tuple represents number of neurons or perceptrons in ith layer

Make sure that when you add a hidden layer its size is between: (no_features + no_classes)/2  and  2*(no_features) 

then we fit the data (train the model) and we predict

model.score gives how accurate are the results in percentage, we're obviously aiming for 100% accuracy,
i.e. all elements of the y vector match the true label vector values
"""
model = nn.MLPClassifier(hidden_layer_sizes=(4,), random_state= 1, max_iter= 4000)
model.fit(x, T)
Y = model.predict(x)

print('T=', T)
print('Y=', Y)

Z = model.score(x, T)
print('Z=', Z)