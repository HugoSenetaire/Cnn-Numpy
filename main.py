import architecture
import parse_data
import loss_functions
import utils
import numpy as np


if __name__ == '__main__' :
    fc = architecture.FCNetwork(28*28,10) 
    cnn = architecture.SimpleCNN(28*28,10)
    # simpleFlatten = architecture.simpleFlattenv2(28*28,10)
    # simpleFlatten = architecture.simpleFlatten(28*28,10)

    labels,X = parse_data.parser("train.csv")
    # fc.train(X.T,labels,loss_functions.CategoricalCrossEntropy(),epochs = 100)
    X = X.reshape(28,28,1,-1)
    cnn.train(X,labels,loss_functions.CategoricalCrossEntropy(),epochs = 1000, learning_rate = 1e-4)
    # simpleFlatten.train(X,labels,loss_functions.CategoricalCrossEntropy(),epochs = 10,learning_rate = 1e-4)