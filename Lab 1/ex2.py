import numpy as np
import matplotlib.pyplot as plt

# f(x) = 7*x^2 + -3*x + 2

x = np.arange(-10,10,1)
fx = 7*x**2 - 3*x +2
plt.close('all')
plt.figure('Polynomial of x', facecolor='green')
plt.plot(x, fx, '.')
#plt.plot(x, fx, 'r')
plt.title('Quadratic Polynomial')


plt.show()
