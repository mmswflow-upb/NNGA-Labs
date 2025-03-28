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

#
# other_T is just T with extra steps

#MLP (Multiple Layer Perceptron) Classifier
model = nn.MLPClassifier(hidden_layer_sizes=(), random_state= 1, max_iter= 500)
model.fit(x, T)
Y = model.predict(x)

print('T=', T)
print('Y=', Y)

Z = model.score(x, T)
print('Z=', Z)