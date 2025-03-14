#input
p = 1
k = 0

# weight
wp = 1
wk = 1

# Bias and Target

b = 1
T = 1

def Y():
    return p * wp + k * wk + b

# Average error squared

def E2():
    return 0.5 * (T-Y() ) ** 2


"""
the error squared is basically the integral of the normal error, if we differentiate it we get back the same formula and 
it's useful for the next functions in which we need to update the weights using the normal error
"""

# derivative of error squared of derivative of the weight of p
def dE2dwp():
    return ( T - Y() ) * (-p)

# derivative of error squared of derivative of the weight of k
def dE2dwk():
    return ( T - Y() ) * (-k)

# derivative of error squared of derivative of the bias
def dE2db():
    return ( T - Y() ) * (-1) # usually bias is just 1 so we just change is value from positive to negative instead

# Teaching rate
# .1 == 0.1
a = .1

for i in range(10):
    print("Epoch ", i, " : ", Y(), E2())
    wp -= a*dE2dwp()
    wk -= a*dE2dwk()
    b -= a*dE2db()
    