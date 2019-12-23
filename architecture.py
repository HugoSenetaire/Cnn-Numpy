import numpy as np 
import layers
import activate
# import Exception
import utils
import loss_functions 
class Model():
    def __init__(self):
        self.inputLayer = None
        self.outputLayer = None
        

    # def train(X,Y,batch_size = 64, optimizer = "default"):
    #     if self.inputLayer is None or self.outputLayer is None :
    #         raise NotImplementedError

        
            


##

class FCNetwork(Model):
    def __init__(self,inputSize, outputSize):
        super().__init__()
        # self.fc1 = layers.FullyConnected(inputSize, 10, activation=activate.SoftMax())
        # self.outputLayer = self.fc1
        self.fc1 = layers.FullyConnected(inputSize, 50, activation=activate.Relu())
        self.fc2 = layers.FullyConnected(50, outputSize,activation=activate.SoftMax())
        self.inputLayer = self.fc1
        self.outputLayer = self.fc2
        self.fc1.addSon(self.fc2)



    def train(self, data, labels, loss, epochs = 100,batch_size = 50,learning_rate = 0.01):
        for epoch in range(epochs) :
            for i, (X, Y) in enumerate(utils.get_batches(data, labels = labels, batch_size=batch_size)):
                X = X.reshape(28*28,-1)
                Y= utils.oneHotEncoding(Y,self.outputLayer.outputSize)
            
                self.inputLayer.forward(X)
                valueLoss = loss.compute_loss(Y,self.outputLayer.activatedOutput)
                precision = loss_functions.precision(Y,self.outputLayer.activatedOutput)
                print(f"For {i+1} batch in {epoch+1} epochs we have {valueLoss} loss and precision {precision}")
                grad = loss.compute_grad(Y,self.outputLayer.activatedOutput)
                self.outputLayer.backward(grad)
                self.inputLayer.optimize(learning_rate)
                self.inputLayer.initGrad()


class SimpleCNN(Model):
    def __init__(self,inputSize, outputSize):
        super().__init__()
        self.conv1 = layers.Convolution(5,1)
        
        self.Pool1 = layers.Pool(5,stride =2)
        self.Flatten = layers.Flatten()
        self.fc1 = layers.FullyConnected(720, 50, activation=activate.Relu())
        self.fc2 = layers.FullyConnected(50, outputSize,activation=activate.SoftMax())
        self.inputLayer = self.conv1
        self.outputLayer = self.fc2

        self.conv1.addSon(self.Pool1)
        self.Pool1.addSon(self.Flatten)
        self.Flatten.addSon(self.fc1)
        self.fc1.addSon(self.fc2)



    def train(self, data, labels, loss, epochs = 100,batch_size = 100,learning_rate = 0.001):
        
        for epoch in range(epochs) :
            for i, (X, Y) in enumerate(utils.get_batches(data, labels, batch_size=batch_size)):
                Y= utils.oneHotEncoding(Y,self.outputLayer.outputSize)
                self.inputLayer.forward(X)
                valueLoss = loss.compute_loss(Y,self.outputLayer.activatedOutput)
                precision = loss_functions.precision(Y,self.outputLayer.activatedOutput)
                print(f"For {i+1} batch in {epoch+1} epochs we have {valueLoss} loss and precision {precision}")
                grad = loss.compute_grad(Y,self.outputLayer.activatedOutput)
                self.outputLayer.backward(grad)
                self.inputLayer.optimize(learning_rate)
                self.inputLayer.initGrad() 
                print(np.max(self.inputLayer.parameters),np.min(self.inputLayer.parameters))           