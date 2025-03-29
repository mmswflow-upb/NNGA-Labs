"""
The ADALINE (Windrow-Hoff) algorithm
"""

#input
p = 1
k = 0

# weight
wp = 1
wk = 2

# Bias and Target

b = 2
T = 1



# Output function of the perceptron
def Y():
    return p * wp + k * wk + b

# Average error squared
def E2():
    return 0.5 * (T-Y() ) ** 2


"""
the error squared is basically the integral of the normal error, if we differentiate it we get back the same formula and 
it's useful for the next functions in which we need to update the weights using the normal error
we will try to find the minimum of this function so that we get as close as possible to the target, we wont necessarily look for a 
perfect solution, because this algorithm can be used on non-linearly separable data sets, we will run a certain number of epochs 
"""


"""
Derivative of error squared with respect to the weight of p

why do we multiply with -p? Because we want to compute the change in the weight of the p input
which must be proportional with the error (which is the difference between the true value and the computed Y value), just like 
in the 2nd lab

You can also try to differentiate it and you will get that result 
"""

def dE2dwp():
    return ( T - Y() ) * (-p)

"""
Same thing here as above, this time for input k though
"""

def dE2dwk():
    return ( T - Y() ) * (-k)

"""
Same thing as above with a single difference, usually bias is just 1 
so we just change the sign and we multiply with the error  (try also differentiating it)
"""
def dE2db():
    return ( T - Y() ) * (-1) 

"""
Teaching or Learning rate

It's used to control how big are the updates on the weights, too small and the learning process will take a very long time
too big and the algorithm might overshoot and wont reach the minimum of the error function very easily and might cause instability
"""
a = .1

"""
This algorithm doesnt necessarily stop somewhere in case there's a non-linearly separable data set, instead we tell it when to stop by
giving it a number of epochs, a threshold for the error, or threshold for changes in weights ||Δw||
"""

for i in range(10): 
    print("Epoch ", i, " : ", Y(), E2())
    wp -= a*dE2dwp()
    wk -= a*dE2dwk()
    b -= a*dE2db()
    