import numpy as np 
import activate
import utils
import copy
import scipy
from scipy import signal


# Abstract Layer class :
class Layer():
    def __init__(self, activation = activate.Identity,name = None):

       

        # Cache
        self.gradient = None 
        self.output = None
        self.activatedOutput = None
        self.parameters = None
        self.name = name
        ## State of forward or backward ()
        self.sonsVisited = 0
        self.parentsVisited = 0
        # Parameters
        self.sons = []
        self.parents = []
        self.activation = activation
        self.trainable = True #TODO
    
    def addSon(self,son):
        """ Add another layer as child of the current layer (if multiple childs, outputs are added) """
        self.sons.append(son)
        son.parents.append(self)

    def optimize(self, learningRate,l2_penalty = 0):
        """ Use current cache gradient to make gradient descent """
        if self.parameters is not None :
            self.parameters -= learningRate * (self.gradient+l2_penalty * self.parameters)
        for son in self.sons :
            son.optimize(learningRate)
    # TODO
    # def save_weights(self,path):
    #     if self.parameters is not None :
            
    
    #     with open(save_path, 'wb') as d:
    #         pickle.dump(dump_cache, d)

    def initGrad(self):
        """ Put every cache to initial state """
        self.gradient = np.zeros(np.shape(self.parameters))
        self.output = None
        self.activatedOutput = None
        self.parentsVisited = 0
        self.sonsVisited = 0
        for son in self.sons :
            son.initGrad()

 
    
    # @abstractmethod
    def forward(self,x):
        raise NotImplementedError 

    def backward(self,x,target):
        raise NotImplementedError

# Implemented Layers :

class FullyConnected(Layer):
    def __init__(self, inputSize, outputSize,activation = activate.Identity()):
        super().__init__(activation)
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.parameters = np.random.uniform(-1,1,size=(outputSize,inputSize))
        self.batch_size = None
        self.gradient = np.zeros(np.shape(self.parameters))
        self.output = None
        self.input = None
        

    def forward(self,x):
        
        if self.output is None :
            self.output = np.dot(self.parameters,x)
            self.input = x
        else : 
            self.output += np.dot(self.parameters,x)
            self.input += x


        if (self.parentsVisited == len(self.parents)):
            self.batch_size = np.shape(x)[1]
            self.activatedOutput = self.activation.forward(self.output)
            for son in self.sons :
                son.parentsVisited+=1
                son.forward(self.activatedOutput)
                
    def backward(self,previousGrad):
        self.batch_size = np.shape(self.output)[1]    
        previousGrad = self.activation.gradient(self.output,previousGrad)
        self.gradient += np.dot(previousGrad,self.input.T)
        if(self.sonsVisited == len(self.sons)):
            self.gradient = self.gradient/self.batch_size
            auxGrad = np.dot(self.parameters.T,previousGrad)
            for parent in self.parents :
                parent.sonsVisited+=1
                parent.backward(auxGrad)
                
  





class Convolution(Layer):

    def __init__(self, nbfilters, previousFilters, kernel_shape = (3,3), padding="same",stride = 1):
        super().__init__()
        

        #Parameters :
        self.nbfilters = nbfilters
        self.kernel_shape = kernel_shape
        self.parameters = np.random.uniform(-1,1,size=(kernel_shape[0],kernel_shape[1],previousFilters, nbfilters))
        self.padding = padding
        self.previousFilters = previousFilters
        self.stride = stride



        # Cache for training :
        self.batch_size = None
        self.gradient = np.zeros(np.shape(self.parameters))
        self.output = None
        self.input = None


    def forward(self,x):
        if self.output is None :
            self.input = x
            self.batch_size = np.shape(x)[1]
            if self.padding == 'same':
                self.pad_h = int(((np.shape(x)[1] - 1)*self.stride + self.kernel_shape[1] - np.shape(x)[1]) / 2)
                self.pad_w = int(((np.shape(x)[0] - 1)*self.stride + self.kernel_shape[0] - np.shape(x)[0]) / 2)
                self.n_H = np.shape(x)[1]
                self.n_W = np.shape(x)[0]
            else:
                self.pad_h = 0  
                self.pad_w = 0
                self.n_H = int((np.shape(x)[1] - self.kernel_shape[1]+1) / self.stride)
                self.n_W = int((np.shape(x)[0] - self.kernel_shape[0]+1) / self.stride)
            self.output = np.zeros((self.n_W,self.n_H,self.nbfilters,np.shape(x)[-1]))
        else : 
            self.input += x
        X_pad = utils.pad(x,self.pad_w,self.pad_h)

        for i in range(np.shape(x)[-1]):
            X = X_pad[:,:,:,i]
            for f in range(self.nbfilters):
                for g in range(self.previousFilters):
                    self.output[:,:,f,i]+= signal.convolve2d(X[:,:,g], self.parameters[:,:,g,f], 'valid')

        if (self.parentsVisited == len(self.parents)):
            for son in self.sons :
                son.parentsVisited+=1
                son.forward(self.output)

    def backward(self,previousGrad):
        X = self.input
        batch_size = np.shape(self.input)[-1]
        dX = np.zeros(np.shape(self.input))
        self.grads = np.zeros(np.shape(self.parameters))

        X_pad = utils.pad(X,self.pad_w,self.pad_h)
        dX_pad = utils.pad(dX,self.pad_w,self.pad_h)


        for h in range(self.n_H):
            for w in range(self.n_W):
                y_start = self.stride * h
                y_end = y_start + self.kernel_shape[1]
                x_start = self.stride * w
                x_end = x_start + self.kernel_shape[0]
                x_slice = X_pad[x_start: x_end, y_start: y_end, :,:]
                dX_pad[x_start:x_end, y_start:y_end, :, :] += np.dot(self.parameters[:, :, :, :],previousGrad[w, h, :, :])
                self.gradient[:,:,:,:] += np.dot(x_slice,previousGrad[w, h, :, :].T)


        self.gradient = self.gradient/batch_size
        for parent in self.parents :
            parent.sonsVisited+=1
            parent.backward(dX)
            




class Pool(Layer):
    def __init__(self, previousFilters, kernel_shape = (2,2), mode = "average",stride = 1):
        super().__init__()
        #Parameters :
        self.nbfilters = previousFilters
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.mode = mode



        # Cache for training :
        self.batch_size = None
        self.gradient = np.zeros(np.shape(self.parameters))
        self.output = None
        self.parentsVisited = 0
        self.sonsVisited = 0
        self.input = None
    
    def forward(self,x):
        if self.output is None :
            self.n_H = int((np.shape(x)[1] - self.kernel_shape[1]) / self.stride)+1
            self.n_W = int((np.shape(x)[0] - self.kernel_shape[0]) / self.stride)+1
            self.output = np.zeros((self.n_W,self.n_H,self.nbfilters,np.shape(x)[-1]))
            self.input = copy.deepcopy(x)
        else : 
            self.input += x


        for h in range(self.n_H):
            for w in range(self.n_W):
                y_start = self.stride * h
                y_end = y_start + self.kernel_shape[1]
                x_start = self.stride * w
                x_end = x_start + self.kernel_shape[0]

                if self.mode == "max" :
                    X_slice = self.input[x_start: x_end, y_start: y_end, :, :]
                    self.output[w,h,:,:] = np.max(np.max(X_slice,axis=0),axis=0)
                elif self.mode == "average" :
                    nbPixels = self.kernel_shape[0]*self.kernel_shape[1]
                    X_slice = self.input[x_start: x_end, y_start: y_end, :, :]
                    self.output[w,h,:,:] = np.sum(np.sum(X_slice,axis=0),axis=0)/nbPixels
                    
        

        if (self.parentsVisited == len(self.parents)):
            self.batch_size = np.shape(x)[1]
            for son in self.sons :
                son.parentsVisited+=1
                son.forward(self.output)
                
    def backward(self,previousGrad):
        batch_size = np.shape(self.input)[-1]
        grad = np.zeros(np.shape(self.input))
        x = self.input[:,:,:,:]
        for h in range(self.n_H): 
            for w in range(self.n_W):
                y_start = self.stride * h
                y_end = y_start + self.kernel_shape[1]
                x_start = self.stride * w
                x_end = x_start + self.kernel_shape[0]
                for f in range(self.nbfilters):
                    if self.mode == "average" :
                        nbPixels = float(self.kernel_shape[0]*self.kernel_shape[1])
                        grad[x_start: x_end, y_start: y_end, f,:] += np.dot(np.ones((self.kernel_shape[0],self.kernel_shape[1],1))/(nbPixels),previousGrad[w,h,f,:].reshape(1,-1))
                    elif self.mode == "max" :
                        x_slice = x[x_start: x_end, y_start: y_end, f,:]
                        grad[x_start: x_end, y_start: y_end, f,:] += np.dot(np.ones(x_slice==np.max(x_slice)).reshape(self.kernel_shape[0],self.kernel_shape[1],1),previousGrad[w,h,f,:].reshape(1,-1))
                    
        for parent in self.parents :
            parent.sonsVisited+=1
            parent.backward(grad)
            


class Flatten(Layer):
    def __init__(self, transpose=True):
        super().__init__()
        self.shape = None
        self.transpose = transpose

    def forward(self, x, save_cache=False):
        self.shape = x.shape
        batch_size = np.shape(x)[-1]
        self.output = np.reshape(x,(-1,batch_size))
        self.activatedOutput = self.output
        if (self.parentsVisited == len(self.parents)):
            self.batch_size = np.shape(x)[1]
            for son in self.sons :
                son.parentsVisited+=1
                son.forward(self.output)

    def backward(self, previousGrad):
        grad = previousGrad.reshape(self.shape)
        for parent in self.parents :
            parent.sonsVisited+=1
            parent.backward(grad)