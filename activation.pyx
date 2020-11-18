from numpy import maximum, heaviside, tanh, sin, cos, multiply, divide, exp, log, power, max as npmax
from math import pi 

def softmax(Z):
    num = exp(Z - npmax(Z))
    out = divide(num, num.sum())
    return out, multiply(out, 1 - out)

def relu(Z):
    out = maximum(Z, 0.0)
    return out, heaviside(out, 1.0)

def sigmoid(Z):
    out = divide(1, 1 + exp(-Z))
    return out, multiply(out, 1 - out)

def ztanh(Z):
    out = tanh(Z)
    return out, multiply(out, 1 - out)

def teq2(Z):
    e_pi = exp(-pi)
    return sin(e_pi * Z) + cos(e_pi), e_pi * -cos(e_pi * Z)

def teq(Z):
    e_pi = exp(-pi) 
    e_pi_z = e_pi * Z 
    return sin(e_pi_z), e_pi * cos(e_pi_z)
