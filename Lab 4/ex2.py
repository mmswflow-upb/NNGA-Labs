
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as nn

# predict the XOR function
#   x1       x2      XOR(x1, x2)
#   0        0            0
#   0        1            1
#   1        0            1
#   1        1            0

x = np.array([ [0,0], [0,1], [1,0], [1,1] ])
T = np.array([0,1,1,0])
#other_T = np.array([[0],[1],[1],[0]])
# other_T is just T with extra steps

#MLP (Multiple Layer Perceptron) Classifier
model = nn.MLPClassifier(hidden_layer_sizes=(), random_state= 1, max_iter= 500)
model.fit(x, T)
Y = model.predict(x)

print('T=', T)
print('Y=', Y)

Z = model.score(x, T)
print('Z=', Z)