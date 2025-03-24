from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt


"""
sklearn stores a data set of greyscaled images of digits from 0 to 9

it has about 1800 input images that go into the perceptron for it to learn how to detect digits

each image is 8x8 pixels so 64 pixels per image

load_digits returns two things: firstly X

X is a matrix of 1797 ~ 1800 rows and 64 columns, on each row we have a new 'image' which is then described by the 64 pixels stored in each column

the pixels data is just their intensity since we're using greyscale instead of rgb, the values go from 0 to 16 in this case due to the encoding of the images (not so relevant)

on the other hand, T is the array of labels that tells the perceptron which digit does the ith image in X represent (ith row is a whole image in X) 

so if you had for example X[2] = (an image of 64 pixels), and T[2] = 3, this means that the image stored at index 2 in X represents the digit '3' 
"""

X, T = load_digits(return_X_y=True) # return ranges from 0 to 9

# 10 ** (-14) == 1e-14

"""
Perceptron(...) constructs a linear classifier that will learn to separate each digit class from the others.

tol=1e-3 tells it to stop iterating once improvement in the loss falls below 0.001.

When you set random_state=0 in scikit‑learn’s Perceptron, you’re giving its internal pseudo‑random number generator a fixed “seed.” 
Although the generator produces values that appear random, seeding it ensures those values follow the exact same 
deterministic sequence every time you run the code. Since the Perceptron uses these values to initialize its weight vector before training, 
using the same seed guarantees that it always starts from identical weights. 
As a result, training the model repeatedly under the same conditions yields exactly the same learned parameters, 
predictions, and performance metrics — making your experiments fully reproducible rather than subject to random variation.

model.fit(X, T) trains the perceptron on all 1,797 flattened‑image samples in X, using their true labels in T

predict(X) applies the trained perceptron to every row of X and returns an array Y of length 1797.

Each element Y[i] is the model’s predicted digit (0–9) for the corresponding image in X[i].
"""

model = Perceptron(tol=1e-3, random_state=0 )
model.fit(X,T)
Y = model.predict(X)


"""
The for loop runs through the first ten rows of X, and for each row it takes the 64‑element flattened pixel vector 
and reshapes it into an 8×8 array that represents the original image.
 
Calls plt.figure() to open a new, blank plotting window so each digit appears in its own figure.

Uses plt.imshow(15 - im, cmap='gray') to display the image in grayscale—subtracting from 15 inverts the pixel intensities so the handwritten strokes appear dark against a light background (original values range 0–16).

Sets the plot’s title to "<True Label> - <Predicted Label>" (via str(T[i]) + '-' + str(Y[i])), letting you immediately see whether the perceptron classified that particular digit correctly.
"""

for i in range(10):
    im = np.reshape(X[i, :], (8,8) ) # image in an 8x8 matrix
    plt.figure()
    plt.imshow(15- im, cmap='grey') # color map in grayscale
    plt.title(str(T[i]) + '-' + str(Y[i])) # Add title (number) of the figure
    