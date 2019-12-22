import numpy as np
class CategoricalCrossEntropy:
    def compute_loss(labels, predictions, epsilon=1e-8):
        predictions /= np.sum(predictions, axis=0, keepdims=True)
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        return -np.sum(labels * np.log(predictions))

    def compute_grad(labels, predictions):
        return -(labels - predictions)

def precision(label,pred):
    label = np.argmax(label,axis = 0)
    pred = np.argmax(pred,axis=0)
    return(np.mean(label==pred))