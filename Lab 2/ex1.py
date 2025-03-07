import numpy as np


#True label vector
T = np.array([0,0,1,0,1])

#tuple = [param1, param2] 
x = np.array([ [1,0], [2,0], [3,0], [0,1], [0,2] ])

#create an array populated with ones with the same length as x
a = np.array([ np.ones(len(x)) ])

#lower-case variables will most likely be arrays or vectors
#upper-case variables are most likely matrices

#We can concatenate vectors to create matrices
#a.T gives you the transpose
X = np.concatenate((x,a.T), axis= 1)

#weight - how relevant are the values in the tuples for drawing the approximated line
w = np.array([1,3,2])

while 1:
    #global error
    eGlobal = 0
    
    
    for i in range(len(x)):
        n = np.sum(X[i] * w)
        
        if n>= 0:
            y = 1
        else:
            y = 0
    
        #local error
        e = T[i] - y
        eGlobal = eGlobal + abs(e)
        
        #d-variable = derivative of that function/variable
        #derivative of the weight
        dw = X[i] * e
        w = w + dw
        
    if eGlobal == 0:
        break
    
print('w = ', w)