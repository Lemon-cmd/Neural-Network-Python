"""
    Lemon-cmd 12/15/2019
    Neural Network Project: Python

    Check out the project in C++ on my github as well.
"""

import math
import random
import numpy as np

"""---------------------------------------------------------------------------------------------------------------------------------"""

def sigmoid(X):
    expression = 1 + math.exp(-X)
    return 1 / (expression)

def sigmoidD(X):
    return (sigmoid(X) * (1 - sigmoid(X)))

def random_weight():
    return random.random()

"""---------------------------------------------------------------------------------------------------------------------------------"""

class Connection():
    def __init__(self):
        """Connection Structure"""

        self.weight = 0
        self.deltaWeight = 0

"""---------------------------------------------------------------------------------------------------------------------------------"""

class Neuron():
    def __init__(self, member_num, output_num):
        """Neuron"""

        self.output = 0 
        self.learning_rate = 0.5 
        self.alpha = 0.5    
        self.memN = member_num  
        self.outputN = output_num
        self.member_gradient = 0
        self.outputWeights = []
        
    def init_weight(self):
        """Initialization of Neuron weights based on output #s"""

        for index in range(self.outputN):
            self.outputWeights.append(Connection())
            self.outputWeights[-1].weight = random_weight()

    def setOutput(self, value):
        """Set Neuron Output"""
        self.output = value
    
    def getOutput(self):
        """Return Neuron Output"""
        return self.output

    def feedF(self, prevLayer):
        """Sum all of the previous layer's outputs including the bias output of the bias neutron (last neutron)"""
        sumF = 0
        for neuron in prevLayer:
            
            #print("Neuron weight: ", neuron.outputWeights[self.memN].weight)
            #summation of current neuron output times all of the weights
            sumF += neuron.getOutput() * neuron.outputWeights[self.memN].weight
        
        #apply sigmoid function onto the sum and set the neuron output
        self.output = sigmoid(sumF)
        
    def sumDOW(self, nextLayer):
        """Calculate the sum of output weights and the gradient of neurons in the next layer"""
        sumDOW = 0

        for n in range (len(nextLayer) - 1):
            sumDOW += self.outputWeights[n].weight * nextLayer[n].member_gradient
        
        return sumDOW

    def hiddenGradient(self, nextLayer):
        """Calculate Gradient of hidden layer neuron"""
        dow = self.sumDOW(nextLayer)
        self.member_gradient = dow * sigmoidD(self.output)

    def outputGradient(self, target):
        """Calculate Gradient of output layer neuron"""
        delta = target - self.output
        self.member_gradient = delta * sigmoidD(self.output)

    def updateWeight(self, prevLayer):
        """Update Weight of neurons in the previous layer"""

        for n in range(len(prevLayer)):
            #current neuron
            current_neuron = prevLayer[n]
            
            #grab old delta weight
            oldDeltaW = current_neuron.outputWeights[self.memN].deltaWeight

            #new delta weight = learning rate * output * gradient + momentum * old delta weight
            newDeltaW = self.learning_rate * current_neuron.getOutput() * self.member_gradient + self.alpha * oldDeltaW

            #set new weight
            current_neuron.outputWeights[self.memN].deltaWeight = newDeltaW
            #add new delta weight to weight
            current_neuron.outputWeights[self.memN].weight += newDeltaW
            
"""---------------------------------------------------------------------------------------------------------------------------------"""

class Network():
    def __init__(self):
        """Network"""

        self.topology = []
        self.layers = []
        self.net_errors = 0
        self.netRAE = 0
        self.netRAE_smooth_factor = 0

    def getTopology(self, topology):
        #grab topology and store in a list
        for num in topology:
            self.topology.append(int(num))

    def initLayers(self):
        """Initalization of Layers"""
        numLayers = len(self.topology)
        for layerN in range(numLayers):
            #create a new layer
            self.layers.append([])
            
            output_num = 0
    
            if layerN != numLayers-1:
                output_num = self.topology[layerN + 1]
            else:
                output_num = 0
 
            for neuronNum in range (self.topology[layerN] + 1):
                self.layers[layerN].append(Neuron(neuronNum, output_num))
                self.layers[layerN][-1].init_weight()

                print("Neuron Created!")

            #set the last neuron in the layer as the bias neuron with 1.0 
            self.layers[-1][-1].setOutput(1.0)

    def getResults(self):
        """Result Method"""
        results = []
        outputLayer = self.layers[-1]

        #loop through last layer and grab all of its neurons
        for n in range(len(outputLayer)-1):
            results.append(outputLayer[n].getOutput())

        print("Result: ", results)
    
    def feedForward(self, inputs):
        """Method for the network feed forward process"""
        #check inputs size; -1 on length of layers due to extra bias neuron
        assert(len(inputs) == len(self.layers[0])-1)
        for i in range(len(inputs)):
            #assign inputs to the first layer aka input neurons
            self.layers[0][i].setOutput(inputs[i])

        #Begin feed forward; skip first (output) layer
        for layerN in range(1, len(self.layers), 1):
            for neuron in range(len(self.layers[layerN]) - 1):
                #grab previous layer
                prevLayer = self.layers[layerN - 1]
                #current neuron
                self.layers[layerN][neuron].feedF(prevLayer)  

    def backPropagation(self, targets):
        """Back Propagation Method"""
        #grab output layer aka last layer
        outputLayer = self.layers[-1]
        self.net_errors = 0
        for n in range(len(outputLayer)-1):
            delta = targets[n] - outputLayer[n].getOutput()

            #set total error of network
            self.net_errors += (delta * delta)
        
        self.net_errors /= len(outputLayer) - 1
        self.net_errors = math.sqrt(abs(self.net_errors))

        self.netRAE = (self.netRAE * self.netRAE_smooth_factor + self.net_errors) / (self.netRAE_smooth_factor + 1.0)
 
        #calculate output layer gradients
        for n in range(len(outputLayer) - 1):
            outputLayer[n].outputGradient(targets[n])

        #calculate hidden layers gradients
        for layerN in range(len(self.layers) - 2, 0, -1):
            hiddenLayer = self.layers[layerN]
            nextLayer = self.layers[layerN + 1]

            #loop through hidden layer
            for neuron in hiddenLayer:
                #calculate hidden gradient using next Layer
                neuron.hiddenGradient(nextLayer)

        #update connection weights from first hidden layer to output layer
        for layerN in range(len(self.layers) - 1, 0 , -1):
            currentLayer = self.layers[layerN]
            prevLayer = self.layers[layerN - 1]

            for n in range(len(currentLayer) - 1):
                currentLayer[n].updateWeight(prevLayer)
            
    def getLayers(self):
        """Method that print out layers"""
        for layerN in range (len(self.layers)):
            print("Layer #: ", layerN)

            for neuron in self.layers[layerN]:
                print("Neuron #: ", str(neuron.memN) + " Output #: ", neuron.outputN)
            
    def getOutputs(self):
        """Method for printing out output of neurons"""
        for layerN in range (len(self.layers)):
            for neuron in self.layers[layerN]:
                print("Layer #: ", str(layerN) + " Neuron #: ", str(neuron.memN) + " Neuron output: ", neuron.output)

    def getRAE(self):
        print("Net Recent Average Error: ", self.netRAE)
        print()

"""---------------------------------------------------------------------------------------------------------------------------------"""

def getTop(trainData):
    topology = trainData.readline()

    index = topology.find(":")
    topology = topology[index + 2:].split(' ')

    return topology

def get(line):
    index = line.find(":")
    line = line[(index + 2):].split(" ")

    inputs = []
    for num in line:
        try:
            inputs.append(float(num))
        except:
            pass

    return inputs

def getData(file):
    inputs = []
    expects = []

    for line in file.readlines():
        if "input:" in line:
            inputs.append(get(line))
        elif "expected:" in line:
            expects.append(get(line))
    
    return inputs, expects

def main():
    #grab file name
    name = str(input("Training file path: "))
    trainData = open(name, "r")

    #construct network
    net = Network()
    #get topology
    net.getTopology(getTop(trainData))
    #initalize the network :: creating layers
    net.initLayers()

    #grab inputs and outputs
    inputs, expects = getData(trainData)

    for count in range(len(inputs)):
        print("Training Count: ", count)

        print("Inputs:", inputs[count])
        net.feedForward(inputs[count])
        net.getResults()

        print("Expected:", expects[count])
        net.backPropagation(expects[count])
        #get average error
        net.getRAE()

main()