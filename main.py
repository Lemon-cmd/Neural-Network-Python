from network import * 
from numpy.random import rand
from numpy import zeros, array 
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 

def pad_sequence(max_length, targets):
    print("MAX LENGTH: ", max_length)
    size = len(targets) 
    out = [] 
    for j in range(size):
        current = zeros((1,max_length))
        current[0][npmax(targets[j])] = 1
        out.append(current)
    
    return out

dataset = load_breast_cancer()
print(dataset.feature_names)
X = dataset.data
Y = dataset.target
max_length = npmax(Y) + 1
Y = pad_sequence(max_length, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,random_state=len(X))


model = seq_net(len(max(X, key=len)), "cross_entropy_error", 0.001)
model.add_layer(10, "relu")
model.add_layer(10, "relu")
model.add_layer(max_length, "softmax")

model.train(X_train, Y_train, epochs=10000, limit=100)
model.test(X_test, Y_test)
