import architecture
import loss_functions
import utils
import numpy as np


if __name__ == '__main__' :
    fc = architecture.FCNetwork(28*28,10) 
    cnn = architecture.SimpleCNN(28*28,10)
    cnn2 = architecture.SimpleCNN2(28*28,10)
    data,labels= utils.parser("train.csv")
    data,labels,data_test,labels_test,data_eval,labels_eval = utils.separate_train_eval(data,labels)
    # fc.train(data,labels,loss_functions.CategoricalCrossEntropy(), epochs = 2000, learning_rate = 1e-2, data_test = data_test, labels_test =labels_test)
    cnn2.train(data,labels,loss_functions.CategoricalCrossEntropy(),epochs = 1000, learning_rate = 1e-5, data_test = data_test, labels_test =labels_test)
