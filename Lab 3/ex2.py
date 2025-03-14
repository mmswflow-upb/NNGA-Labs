# Estimate the sqrt of n

import numpy as np


import matplotlib.pyplot as plt

n = int(input("Give me a value for n = "))

step = 0.1

def f(x):
    return x**2 - n

x = np.arange(-n, n+0.1, step)

# Array of zeros
z = np.zeros(len(x))


plt.close('all')
plt.plot(x, f(x))
plt.plot(x, z)
plt.grid('on')


A = 0
B = n

plt.plot(A, 0, 'ok') # 'ok' makes a circle appear in that point
plt.plot(B, 0, 'ok') 

C = (A+B)/2
eps = 10**(-14) # Threshold, this is the allowed error

while 1:
    plt.plot(C, 0, '.r')
    if f(C) * f(A) < 0:
        B = C
    else:
        A = C
    
    if(np.abs(f(C)) < eps):
        break
    else:
        C = (A+B)/2        

print(C)
