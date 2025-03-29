"""

    logical func with 3 inputs 
    
    The folmula for the logical function (not minimized)
    F(x,y,z) = x`y`z + x`yz` + xy`z` 

"""

import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([
    [0, 0, 0],  
    [0, 0, 1], 
    [0, 1, 0],  
    [0, 1, 1],  
    [1, 0, 0],  
    [1, 0, 1], 
    [1, 1, 0],  
    [1, 1, 1],  
])

T = np.array([0, 1, 1, 0, 1, 0, 0, 0])  # expected output

"""
We use multilayered perceptron with one hidden layer containing 3 neurons, the activation function is the logistic function and the
random state is set to 1 to seed the same random numbers for the weights every time we run the code
max num of iterations is set to 2000 and its sufficient, since I got the score of 1.0
"""

model = MLPClassifier(hidden_layer_sizes=(3,), max_iter=2000, activation='logistic', solver='lbfgs', random_state=1)
model.fit(X, T)

predictions = model.predict(X)

print("Expected Outputs:", T)
print("MLP Predictions:", predictions)
print("Model Accuracy:", model.score(X, T))
