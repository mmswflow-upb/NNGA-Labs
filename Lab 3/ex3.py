from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt


X, T = load_digits(return_X_y=True) # return ranges from 0 to 9

# 10 ** (-14) == 1e-14

model = Perceptron(tol=1e-3, random_state=0 )
model.fit(X,T)
Y = model.predict(X)


for i in range(10):
    im = np.reshape(X[i, :], (8,8) ) # image in an 8x8 matrix
    plt.figure()
    plt.imshow(15- im, cmap='grey') # color map in grayscale
    plt.title(str(T[i]) + '-' + str(Y[i]))
    