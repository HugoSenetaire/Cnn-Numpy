import architecture
import parse_data
import loss_functions
import utils
import numpy as np


if __name__ == '__main__' :
    fc = architecture.FCNetwork(28*28,10) 
    cnn = architecture.SimpleCNN(28*28,10)
    labels,X = parse_data.parser("train.csv")
    
    # print("INIT2",np.shape(labels))
    # fc.train(X.T,labels,loss_functions.CategoricalCrossEntropy())
    X = X.reshape(28,28,1,-1)
    cnn.train(X,labels,loss_functions.CategoricalCrossEntropy())
