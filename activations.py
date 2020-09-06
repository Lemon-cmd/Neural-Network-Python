import numpy as np

def relu(Z):
    return np.maximum(Z, 0.0)

def drelu(O):
    return np.heaviside(O, 1.0) 

def linear(Z):
    return Z

def dlinear(O):
    return O

def tanh(Z):
    return np.tanh(Z)
    
def dtanh(O):
    return 1.0 - np.multiply(O,O)  

def softmax(Z):
    expA = np.exp(Z)
    return expA / expA.sum()

def dsoftmax(O):
    return np.multiply(O, 1.0 - O)

def sigmoid(Z):
    #Z = np.heaviside(Z, Z.min())
    #return 1.0 / (1.0 + np.exp(-Z))
    return np.exp(-np.logaddexp(0, -Z))

def dsigmoid(O):
    return np.multiply(O, 1.0 - O)
