import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 2 * np.pi + 1, 0.1) # Creates a range of numbers with given start, end and step

plt.close('all') # Closes all the windows
plt.figure('My first Plot', facecolor='yellow') # Creates a new window with the title 'My first Plot' and yellow background
plt.subplot(2,1,1) # Creates a new plot in the window with 2 rows and 1 column, in the first row
plt.title('Sine Function') # Gives title to the previously mentioned plot
plt.grid('on') #Turn on the grid


f_sine = np.sin(t) # Creates a sine wave
plt.plot(t, f_sine, 'r') # Plots the sine wave with red color (mapping values from t to f_sine)
plt.subplot(2,1,2) # Creates a new plot in the window with 2 rows and 1 column, in the second row

f_cos = np.cos(t) # same as above but with cosine
plt.plot(t, f_cos, 'b')
plt.title('Cosine Function')
plt.grid('on') #Turn on the grid

#Creating a circle with sine and cosine
plt.figure('Circle', facecolor= 'red') # Make a new figure (window)
plt.axis('equal') # Make axis of the same length

plt.plot(f_cos, f_sine, 'green')
plt.show()
