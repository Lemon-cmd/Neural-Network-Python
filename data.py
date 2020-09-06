import csv 
import numpy as np

def shuffle(a, b, seed):
    x, y = a, b
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(x)
    rand_state.seed(seed)
    rand_state.shuffle(y)
    return x, y

X = []
Y = []
id = -1
x_id = {}
id_x = {}

"""
with open('iris.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        X.append(np.matrix(list(map(float, row[:4])), dtype=float))

        if(row[-1] == 'Iris-setosa'):
            Y.append(np.matrix([1, 0, 0], dtype=float))
        elif(row[-1] == 'Iris-versicolor'):
            Y.append(np.matrix([0, 1, 0], dtype=float))
        else:
            Y.append(np.matrix([0, 0, 1], dtype=float))
"""

"""
with open('animal.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        X.append(np.matrix(list(map(float, row[:3])), dtype=float))

        if(row[-1] == 'chicken'):
            Y.append(np.matrix([1, 0, 0], dtype=float))
        elif(row[-1] == 'squirel'):
            Y.append(np.matrix([0, 1, 0], dtype=float))
        else:
            Y.append(np.matrix([0, 0, 1], dtype=float))
"""
"""
with open('fish.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        X.append(np.matrix(list(map(float, row[1:6])), dtype=float))
        Y.append(np.matrix(row[6], dtype=float))
"""
"""
with open("szeged_weather.csv", newline='') as csvfile:
   reader = csv.reader(csvfile)
   next(reader, None)  # skip the headers
   for row in reader:
        x = []
        for j in range(len(row)-1):
            try:
                x.append(float(row[j]))
            except:
                if row[j] not in x_id:
                    x_id[row[j]] = id 
                    x.append(x_id[row[j]])
                    id += 1
                else:
                    x.append(x_id[row[j]])
        
        #x = np.matrix(x, dtype=float)
        #X.append(x)
        X.append(np.matrix(list(map(float, row[2:9])), dtype=float))
        #X.append(np.matrix(list(map(float, x[1:9])), dtype=float))
        
        try:
            Y.append(np.matrix(float(row[-1])))
        except:
            if row[-1] not in x_id:
                x_id[row[-1]] = id 
                Y.append(np.matrix([x_id[row[-1]]], dtype=float))
                id += 1
            else:
                Y.append(np.matrix([x_id[row[-1]]], dtype=float))

"""

with open("abalone.csv", newline='') as csvfile:
   reader = csv.reader(csvfile)
   next(reader, None)  # skip the headers
   for row in reader:
        X.append(np.matrix(list(map(float, row[5:len(row)-1])), dtype=float))
        
        if (row[-1] == "M"):
            Y.append(np.matrix([1, 0, 0], dtype=float))
        elif (row[-1] == "F"):
            Y.append(np.matrix([0, 1, 0], dtype=float))
        else:
            Y.append(np.matrix([0, 0, 1], dtype=float))


import random
X = np.array(X)
Y = np.array(Y)
#X = X/np.linalg.norm(X, axis=0)