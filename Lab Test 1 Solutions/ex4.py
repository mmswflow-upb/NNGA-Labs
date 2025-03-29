# Estimate the sqrt of n

import numpy as np


import matplotlib.pyplot as plt

n = 1000

step = 0.01 # step of the interval  (difference between each number in the interval)

"""
We define the function
"""
def f(x):
    return np.sin(x) + x**2/5 + x/2 - 1.5

"""
We just create the range of numbers here (x-axis) to plot the function
"""
x = np.arange(-n, n+0.1, step)

"""
Create an empty array of zeroes of the same length as the x-axis range in order to plot the line  y = 0 (x-axis)
this line shows us where the graph of the function intersects with the x-axis
"""
z = np.zeros(len(x))


"""
We start at A
"""

A = -n
B = n


C = (A+B)/2

"""
The closer |f(c)| is to zero the closer C is to the real root, because as we go down the graph or up towards the intersection between the
x-axis and the graph of the function, the closer we are to the root
"""


eps = 10**(-14) 

"""
Threshold, this is value under which the function evaluated at c (f(C)) has to get for the 
algorithm to stop, in other words this is how close f(C) must be on the y-axis to the x axis (y = 0)
"""

while 1:

    
    """
    We use binary search  algorithm (in a way) to find the root, if the functions value changes between A and C, then B has to be shifted
    to the left where C is, because the root is 100% between A and C since thats where the function changes its value, else the root is on the right
    between C and B
    """
    if f(C) * f(A) < 0:
        B = C
    else:
        A = C
    
    """
    If f(C) got close enough to zero on the y-axis (x-axis: y = 0) (smaller than eps), then stop, otherwise keep going to get a more accurate result
    it's in the form abs(f(C)) because the function evaluated at point C might also be under the x-axis (if the function goes under x-axis thats natural)
    """
    
    if(np.abs(f(C)) < eps):
        break
    else:
        C = (A+B)/2        

print(C)
