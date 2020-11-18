from numpy.random import rand 
from numpy import ones, dot, sqrt, square, absolute
from activation import * 

class layer():
    def __init__(self, neurons, inputs, id, activation="sigmoid"):
        self.shape = (neurons, inputs)
        self.layer_id = id
        self.B = ones((1, neurons))
        self.act_fun = activation
        self.Wz, self.old_Wz = rand(neurons, inputs), None   
        self.vB, self.vW = 0, 0 
        self.O, self.dO, self.I, self.delta = None, None, None, None
        self.activate = {"sigmoid" : sigmoid, "relu" : relu, "tanh" : ztanh,\
                         "teq" : teq, "softmax" : softmax,"teq2": teq2}
        self.cost_funs = {"mean_square_error": self.mean_square_error, "cross_entropy_error": self.cross_entropy_error}
        self.lda = 0.01 

    def mean_square_error(self, Y):
        return 0.5 * square(self.O - Y).sum() + self.lda * square(self.Wz).sum(), multiply(self.O - Y, self.dO)

    def cross_entropy_error(self, Y):
        return multiply(-Y, log(self.O)).sum() + self.lda * square(self.Wz).sum(), self.O - Y

    def forward(self, X):
        self.I = X.reshape(1, -1)
        self.O = dot(self.I, self.Wz.transpose()) + self.B
        self.O, self.dO = self.activate[self.act_fun](self.O)
        self.old_Wz = self.Wz 

    def update(self, vW, vB, lr=0.001, e=1e-8):
        gradW = dot(self.delta.transpose(), self.I)

        #Using rms optimization
        self.vW = 0.1 * vW + 0.9 * square(gradW).sum()
        self.vB = 0.1 * vB + 0.9 * square(self.delta).sum()

        self.Wz = self.Wz - lr/sqrt(self.vW + e) * (gradW + self.lda * 2 * self.Wz)  
        self.B = self.B - lr/sqrt(self.vB + e) * self.delta

    def cost(self, Y, cf):
        return self.cost_funs[cf](Y)

