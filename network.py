import numpy as np
from data import *
from activations import *
import random

class network():
    class layer():
        def __init__(self, nodes, activation):
            self.W = None 
            self.I = None 
            self.O = None 
            self.B = None 
            self.delta = None
            self.act = activation
            self.nhiddens = nodes 

    def __init__(self, ins, lr=0.1):
        self._nins = ins 
        self._lrate = lr 
        self.accuracy = 0.0

        self.loss = 0.0
        self.dloss = 0.0
        
        self.__activate = {'sigmoid' : sigmoid, 'tanh': tanh, 'softmax': softmax, 
        'linear': linear, 'relu': relu}
        self.__dactivate = {'sigmoid' : dsigmoid, 'tanh': dtanh, 'softmax': dsoftmax, 
        'linear': dlinear, 'relu': drelu}

        self.__net = []
        self.__size = 0

    def summary(self):
        for layer in self.__net:
            print(layer.W)

    def addLayer(self, nodes, activation = 'linear'):
        item = self.layer(nodes, activation)
        if (self.__size == 0):
            item.W = np.random.rand(nodes, self._nins) * (2/self._nins)
        else:
            item.W = np.random.rand(nodes, self.__net[-1].nhiddens) * (2/self._nins)

        item.O = np.zeros((1, nodes), dtype=float)
        item.dO = np.zeros((1, nodes), dtype=float)
        item.B = np.ones((1, nodes), dtype=float) * 0.001

        self.__size += 1
        self.__net.append(item)
    
    def forward(self, X, Y):
        X = np.matrix(X, dtype=float)

        for j in range(self.__size):
            if (j == 0):
                self.__net[j].I = X 
            else:
                self.__net[j].I = self.__net[j - 1].O

            self.__net[j].O = np.dot(self.__net[j].I, self.__net[j].W.T) + self.__net[j].B 
            self.__net[j].O = self.__activate[self.__net[j].act](self.__net[j].O)
            self.__net[j].dO = self.__dactivate[self.__net[j].act](self.__net[j].O)
            self.__net[j].dO = np.clip(-10, 10, self.__net[j].dO)

        if (np.argmax(self.__net[-1].O) == np.argmax(Y)):
            self.accuracy += 1.0

        #if (np.round(self.__net[-1].O) == Y):
        #    self.accuracy += 1.0

        self.loss = np.sum(np.multiply(-Y, np.log(self.__net[-1].O)))
        self.dloss = self.__net[-1].O - Y
        self.dloss = np.clip(-10, 10, self.dloss)
        #self.loss = np.sum(np.square(self.__net[-1].O - Y))
        #self.dloss = np.sum(self.__net[-1].O - Y)

    def backpropagation(self, Y):    
        #update output layer
        self.__net[-1].W -= self._lrate * np.dot(self.dloss.T, self.__net[-1].I)
        self.__net[-1].B -= self._lrate * self.dloss
        self.__net[-1].delta = self.dloss

        for j in range(self.__size - 2, -1, -1):
            self.__net[j].delta = self._lrate * np.multiply(np.dot(self.__net[j + 1].delta, self.__net[j + 1].W), self.__net[j].dO)
            self.__net[j].W -= np.dot(self.__net[j].delta.T, self.__net[j].I)
            self.__net[j].B -= self.__net[j].delta
            
    def train(self, X, Y, epochs=100):
        size = len(X)
        x, y = X, Y
        for e in range(epochs):
            z = random.randrange(0, size//2)
            z1 = random.randrange(0, size)
            x, y = shuffle(X, Y, random.randrange(1, 2555))
            for i in range(size):
                self.forward(x[i], y[i])

                if (i >= z and i <= z1):
                    self.backpropagation(Y[i])
                
            print("Loss: {0} Accuracy: {1}%".format(self.loss, self.accuracy/size * 100))
            self.accuracy = 0.0

    def test(self, X, Y):
        for i in range(len(X)):
            self.forward(X[i], Y[i])

        print("Accuracy: {0}%".format(self.accuracy/len(X) * 100))


net = network(3, 0.0001)
net.addLayer(20, 'relu')
net.addLayer(3, 'softmax')

net.train(X[:len(X)*3//4], Y[:len(Y)*3//4], 200)
net.test(X[len(X)*3//4:], Y[len(X)*3//4:])

