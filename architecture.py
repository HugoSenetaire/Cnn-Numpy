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
        

    def train(self, data, labels, loss, epochs = 10,batch_size = 100,learning_rate = 0.001):
        for epoch in range(epochs) :
            compteur = 0
            for i, (X, Y) in enumerate(utils.get_batches(data, labels, batch_size=batch_size)):
                Y= utils.oneHotEncoding(Y,self.outputSize)
                compteur+=np.shape(X)[-1]
                self.inputLayer.forward(X)
                valueLoss = loss.compute_loss(Y,self.outputLayer.activatedOutput)
                precision = loss_functions.precision(Y,self.outputLayer.activatedOutput)
                print(f"For {i+1} batch in {epoch+1} epochs we have {valueLoss} loss and precision {precision}")
                print(f"({compteur}/{len(labels)})")
                grad = loss.compute_grad(Y,self.outputLayer.activatedOutput)
                self.outputLayer.backward(grad)
                self.inputLayer.optimize(learning_rate)
                self.inputLayer.initGrad() 
                print(np.max(self.inputLayer.parameters),np.min(self.inputLayer.parameters))           
    
    def evaluate(self, X, Y, loss):
        self.inputLayer.initGrad() 
        Y= utils.oneHotEncoding(Y,self.outputLayer.outputSize)
        self.inputLayer.forward(X)
        valueLoss = loss.compute_loss(Y,self.outputLayer.activatedOutput)
        precision = loss_functions.precision(Y,self.outputLayer.activatedOutput)
        print(f"For prediction we have {valueLoss} loss and precision {precision}")
            
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
        self.Flatten2 = layers.Flatten()
        self.fc1 = layers.FullyConnected(inputSize, 50, activation=activate.Relu())
        self.fc2 = layers.FullyConnected(50, outputSize,activation=activate.SoftMax())
        self.inputLayer = self.Flatten
        self.outputLayer = self.Flatten2
        self.Flatten.addSon(self.fc1)
        self.fc1.addSon(self.fc2)
        self.fc2.addSon(self.Flatten2)





class SimpleCNN(Model):
    def __init__(self,inputSize, outputSize):
        super().__init__()
        self.outputSize = outputSize
        self.conv1 = layers.Convolution(5,1)
        self.Pool1 = layers.Pool(5,stride =2)
        self.Flatten = layers.Flatten()
        self.fc1 = layers.FullyConnected(26*26*5, 50, activation=activate.Relu())
        self.fc2 = layers.FullyConnected(50, outputSize,activation=activate.SoftMax())
        self.inputLayer = self.conv1
        self.outputLayer = self.fc2
        self.conv1.addSon(self.Flatten)
        # self.Pool1.addSon(self.Flatten)
        self.Flatten.addSon(self.fc1)
        self.fc1.addSon(self.fc2)



class simpleFlatten(Model):
    def __init__(self,inputSize, outputSize):
        super().__init__()
        self.outputSize = outputSize
        self.Flatteninit=layers.Flatten()
        self.fc1=layers.FullyConnected(28*28, 10, activation=activate.Relu())
        self.Flatten=layers.Flatten()
        self.inputLayer = self.Flatteninit
        self.outputLayer = self.Flatten
        self.fc1.addSon(self.Flatten)
        self.Flatteninit.addSon(self.fc1)

class simpleFlattenv2(Model):
    def __init__(self,inputSize, outputSize):
        super().__init__()
        self.outputSize = outputSize
        self.Flatteninit=layers.Flatten()
        self.fc1=layers.FullyConnected(28*28, 10, activation=activate.Relu())
        self.Flatten=layers.Flatten()
        self.inputLayer = self.Flatteninit
        self.outputLayer = self.fc1
        self.Flatteninit.addSon(self.fc1)