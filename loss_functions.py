import numpy as np


class CategoricalCrossEntropy:
    def compute_loss(self,labels, predictions):
        return -np.sum(labels * np.log(predictions))

    def compute_grad(self,labels, predictions):
        return -(labels-predictions)

def precision(label,pred):
    label = np.argmax(label,axis = 0)
    pred = np.argmax(pred,axis=0)
    return(np.mean(label==pred))