import architecture
import parse_data
import loss
import utils
import numpy as np
# print("OK")
if __name__ == '__main__' :
    fc = architecture.FCNetwork(28*28,10) 
    labels,X = parse_data.parser("train.csv")
    # labels_oneHot = utils.oneHotEncoding(labels)
    # print(labels_oneHot.shape)
    # print(np.shape(labels))
    fc.train(X,labels,loss.CategoricalCrossEntropy())
