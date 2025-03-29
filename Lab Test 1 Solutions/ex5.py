import numpy as np
import sklearn.neural_network as nn


"""
predict the XOR function, this can be done only if we have multiple layers of perceptrons, since it isn't
linearly separable...

x1       x2      AND(x1, x2)
0        0            0
0        1            0
1        0            0
1        1            1


F(x1,x2) = x1x2

From this table we determine that the true label vector is:
          
AND = T = [0, 0, 0, 1]  
                       

And the matrix of inputs is: 



    |0 0|
x = |0 1|
    |1 0|
    |1 1|
"""

x = np.array([ [0,0], [0,1], [1,0], [1,1] ])
T = np.array([0,0,0,1])


"""
MLP (Multiple Layer Perceptron) Classifier


the hidden_layer_sizes=() parameter takes a tuple of numbers, ith number 
in the tuple represents number of neurons in ith layer, in this case we don't need a hidden layer for the AND function


then we fit the data (train the model) and we predict

model.score gives how accurate are the results in percentage, we're obviously aiming for 100% accuracy,
i.e. all elements of the y vector match the true label vector values
"""
model = nn.MLPClassifier(hidden_layer_sizes=(), random_state= 1, max_iter= 2000)
model.fit(x, T)
Y = model.predict(x)

print('T=', T)
print('Y=', Y)

print('Weights: ', model.coefs_)
print('Biases: ', model.intercepts_)

Z = model.score(x, T)
print('Score=', Z)