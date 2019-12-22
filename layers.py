import numpy as np 
import activate

# Abstract Layer class :
class Layer():
    def __init__(self, activation = activate.Identity):
        self.sons = []
        self.sonsVisited = 0
        self.parents = []
        self.gradient = None # Récursive
        self.output = None
        self.activatedOutput = None
        self.activation = activation
        self.parameters = None
        self.trainable = True
    
    def addSon(self,son):
        self.sons.append(son)
        son.parents.append(self)

    def optimize(self, learningRate,l2_penalty = 1e-4):
        self.parameters -= learningRate * (self.gradient+l2_penalty * self.parameters)
        # self.parameters -=learningRate*self.gradient
        for son in self.sons :
            son.optimize(learningRate)


    def initGrad(self):
        self.gradient = np.zeros(np.shape(self.parameters))
        self.output = None
        self.activatedOutput = None
        self.parentsVisited = 0
        for son in self.sons :
            son.initGrad()

 
    
    # @abstractmethod
    def forward(self,x):
        raise NotImplentedError 

    def backward(self,x,target):
        raise NotImplentedError

# Implemented Layers :

class FullyConnected(Layer):
    def __init__(self, inputSize, outputSize,activation = activate.Identity()):
        super().__init__(activation)
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.parameters = np.random.normal(size=(outputSize,inputSize))
        self.batch_size = None
        self.gradient = np.zeros(np.shape(self.parameters))
        self.output = None
        self.first = True
        self.parentsVisited = 0
        self.sonsVisited = 0
        self.input = None
        

    def forward(self,x):
        if self.output is None :
            self.output = np.dot(self.parameters,x)
            self.input = x
            self.first = False
        else : 
            self.output += np.dot(self.parameters,x)
            self.input += np.dot(self.parameters,x)


        if (self.parentsVisited == len(self.parents)):
            self.batch_size = np.shape(x)[1]
            self.activatedOutput = self.activation.forward(self.output)
            for son in self.sons :
                son.parentsVisited+=1
                son.forward(self.activatedOutput)
                
    def backward(self,previousGrad):

        self.batch_size = np.shape(self.output)[1]    
        previousGrad = self.activation.gradient(self.output,previousGrad)
        self.gradient += np.dot(previousGrad,self.input.T)/self.batch_size
        if(self.sonsVisited == len(self.sons)):
            auxGrad = np.dot(self.parameters.T,previousGrad)
            for parent in self.parents :
                parent.backward(auxGrad)
                parent.sonsVisited+=1
  





class Convolution(Layer):
     def __init__(self, nbfilters, kernel_shape = (3,3), padding=None,stride = 1, activation = activate.Identity()):
        super().__init__(activation)
        

        #Parameters :
        self.nbfilters = nbfilters
        self.kernel_shape = kernel_shape
        self.parameters = np.random.normal(size=(kernel_shape[0],kernel_shape[1],filters))
        self.padding = padding
        self.stride = stride



        # Cache for training :
        self.batch_size = None
        self.gradient = np.zeros(np.shape(self.parameters))
        self.output = None
        self.first = True
        self.parentsVisited = 0
        self.sonsVisited = 0
        self.input = None
        
    def forward(self,x):
        if self.output is None :
            self.output = np.zeros(kernel_shape[0],kernel_shape[1],filters,np.shape(x)[0])
            self.in
        # if self.output is None :
        #     self.output = np.dot(self.parameters,x)
        #     self.input = x
        #     self.first = False
        # else : 
        #     self.output += np.dot(self.parameters,x)
        #     self.input += np.dot(self.parameters,x)


        # if (self.parentsVisited == len(self.parents)):
        #     self.batch_size = np.shape(x)[1]
        #     self.activatedOutput = self.activation.forward(self.output)
        #     for son in self.sons :
        #         son.parentsVisited+=1
        #         son.forward(self.activatedOutput)
                
    def backward(self,previousGrad):

        self.batch_size = np.shape(self.output)[1]    
        previousGrad = self.activation.gradient(self.output,previousGrad)
        self.gradient += np.dot(previousGrad,self.input.T)/self.batch_size
        if(self.sonsVisited == len(self.sons)):
            auxGrad = np.dot(self.parameters.T,previousGrad)
            for parent in self.parents :
                parent.backward(auxGrad)
                parent.sonsVisited+=1