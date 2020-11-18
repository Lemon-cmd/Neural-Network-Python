from layer import * 
from batch_gen import * 
from numpy import allclose, round as npround, clip, argmax, exp
from fastrand import pcg32bounded

class seq_net():
    def __init__(self, ins, cf="mean_square_error", lr=0.001):
        self.__size, self.__loss, self.__dloss, self.__lrate = 0, 0.0, 0.0, lr 
        self.__cf = cf
        self.__ins = ins
        self.__net = []

    def summary(self):
        print("************************SUMMARY***********************")
        print("Num. of layers: {0} Cost Function: {1}\nOptimizer: {2}\n".format(self.__size, self.__cf, "adam"))
        for j in range(self.__size):
            print("Layer: {0} Shape: {1} Activation: {2}".format(j, self.__net[j].shape, self.__net[j].act_fun))
        print("\n")

    def add_layer(self, neurons, activate="sigmoid"):
        item = None 
        if (self.__size == 0):
            item = layer(neurons, self.__ins, self.__size, activate) 
        else:
            item = layer(neurons, self.__net[-1].shape[0], self.__size, activate)

        self.__net.append(item)
        self.__size += 1

    def forward_prop(self, X, Y, accuracy):
        for j in range(self.__size):
            if (j == 0):
                self.__net[j].forward(X)
            else:
                self.__net[j].forward(self.__net[j - 1].O)
        
        estimate = npround(self.__net[-1].O)
        if (self.__cf == "cross_entropy_error"):
            if (argmax(estimate) == argmax(Y)):
                accuracy += 1
        else:
            if (allclose(estimate, Y, equal_nan=True)):
                accuracy += 1

        return accuracy 

    def backward_prop(self):
        #update output layer
        self.__net[-1].delta = self.__dloss
        self.__net[-1].update(0, 0, self.__lrate)

        for j in range(self.__size - 2, -1, -1):
            self.__net[j].delta = multiply(dot(self.__net[j + 1].delta, self.__net[j + 1].old_Wz), self.__net[j].dO)
            self.__net[j].update(self.__net[j + 1].vW, self.__net[j + 1].vB, self.__lrate)

    def train(self, X, Y, epochs=10000, limit=100, verbose=True):
        size, accuracy, batch_size = len(X), 0.0, len(X) * 5 // 100
        batch = batch_gen(size, batch_size)
        batch_acc, batch_ep, batch_range = batch.pop() 
        start, end = batch_range[0], batch_range[1]

        for e in range(epochs):
            for i in range(start, end, 1):
                accuracy = self.forward_prop(X[i], Y[i], accuracy)
                self.__loss, self.__dloss = self.__net[-1].cost(Y[i], self.__cf)
                self.backward_prop()

            if (verbose and e % limit == 0):
                print("Epoch - {0} Batch Range - ({1} {2}) Current Loss - {3} Accuracy - {4}".format(e, start, end, self.__loss, accuracy/(batch_size)))
           
            batch_acc, batch_ep, batch_range = batch.pop()
            batch.push(Connection(epoch = e, accuracy = accuracy + exp(-e), range=(start, end)))
            start, end = batch_range[0], batch_range[1]
            accuracy = 0.0 

    def test(self, X, Y):
        size = len(X)
        accuracy = 0
        for i in range(size):
            accuracy = self.forward_prop(X[i], Y[i], accuracy)
            self.__loss, self.__dloss = self.__net[-1].cost(Y[i], self.__cf)
        
        print("Current Loss - {0} Accuracy - {1}%".format(self.__loss, 100 * accuracy / size ))
