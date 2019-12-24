import numpy as np 
import layers
import activate
import utils
import loss_functions 

class Model():
    def __init__(self):
        self.inputLayer = None
        self.outputLayer = None

    def train(self, data, labels, loss, epochs = 10,batch_size = 256,learning_rate = 0.001,data_test=None,labels_test=None):
        for epoch in range(epochs) :
            compteur = 0
            nbBatch = int(np.shape(data)[0]/batch_size)
            for i, (X, Y) in enumerate(utils.get_batches(data, labels, batch_size=batch_size)):
                compteur+=np.shape(X)[-1]
                X = X.reshape(28,28,1,-1)
                Y= utils.oneHotEncoding(Y,self.outputSize)
                self.inputLayer.forward(X)
                if abs(i/float(nbBatch)-0.5)<0.01 or abs(i/float(nbBatch)-0.3)<0.01 or abs(i/float(nbBatch)-0.8)<0.01:
                    valueLoss = loss.compute_loss(Y,self.outputLayer.activatedOutput)
                    precision = loss_functions.precision(Y,self.outputLayer.activatedOutput)
                    print(f"TRAIN ({compteur}/{len(labels)}) Batch: {i+1} Epochs: {epoch+1} - Loss : {valueLoss} Precision : {precision}")
                grad = loss.compute_grad(Y,self.outputLayer.activatedOutput)
                self.outputLayer.backward(grad)
                self.inputLayer.optimize(learning_rate)
                self.inputLayer.initGrad()
            if data_test is not None and labels_test is not None :
                self.evaluate(data_test,labels_test,loss)
                
    
    def evaluate(self, data, labels, loss,batch_size = 256):
        for i, (X, Y) in enumerate(utils.get_batches(data, labels, batch_size=400000)):
            Y= utils.oneHotEncoding(Y,self.outputLayer.outputSize)
            X = X.reshape(28,28,1,-1)
            self.inputLayer.forward(X)
            valueLoss = loss.compute_loss(Y,self.outputLayer.activatedOutput)
            precision = loss_functions.precision(Y,self.outputLayer.activatedOutput)
            print("===================================================")
            print(f"TEST - Loss : {valueLoss} Precision : {precision}")
            print("===================================================")
            self.inputLayer.initGrad() 
            
    def predict(self, X):
        self.inputLayer.initGrad() 
        self.inputLayer.forward(X)
        return self.outputLayer.activatedOutput

## Definition of different network

class FCNetwork(Model):
    def __init__(self,inputSize, outputSize):
        super().__init__()
        self.outputSize = outputSize
        self.Flatten = layers.Flatten()
        self.fc1 = layers.FullyConnected(inputSize, 10, activation=activate.SoftMax())
        self.inputLayer = self.Flatten
        self.outputLayer = self.fc1
        self.Flatten.addSon(self.fc1)





class SimpleCNN(Model):
    def __init__(self,inputSize, outputSize):
        super().__init__()
        self.outputSize = outputSize
        self.conv1 = layers.Convolution(5,1)
        self.Pool1 = layers.Pool(5,stride =2)
        self.Flatten = layers.Flatten()
        self.fc1 = layers.FullyConnected(14*14*5, 50, activation=activate.Relu())
        self.fc2 = layers.FullyConnected(50, outputSize,activation=activate.SoftMax())
        self.inputLayer = self.conv1
        self.outputLayer = self.fc2
        self.conv1.addSon(self.Pool1)
        self.Pool1.addSon(self.Flatten)
        self.Flatten.addSon(self.fc1)
        self.fc1.addSon(self.fc2)

class SimpleCNN2(Model):
    def __init__(self,inputSize, outputSize):
        super().__init__()
        self.outputSize = outputSize
        self.conv1 = layers.Convolution(5,1)
        self.Pool1 = layers.Pool(5,stride =2)
        self.conv2 = layers.Convolution(10,5)
        self.Pool2 = layers.Pool(10,stride =2)
        self.Flatten = layers.Flatten()
        self.fc1 = layers.FullyConnected(7*7*10, 50, activation=activate.Relu())
        self.fc2 = layers.FullyConnected(50, outputSize,activation=activate.SoftMax())
        self.inputLayer = self.conv1
        self.outputLayer = self.fc2
        self.conv1.addSon(self.Pool1)
        self.Pool1.addSon(self.conv2)
        self.conv2.addSon(self.Pool2)
        self.Pool2.addSon(self.Flatten)
        self.Flatten.addSon(self.fc1)
        self.fc1.addSon(self.fc2)


