import numpy as np 


## Abstract class for Activation :
class Activation():
    def forward(self,x):
        raise NotImplementedError

    def gradient(self,x):
        raise NotImplementedError

## Real class for activation :

class Identity(Activation):
    def forward(self,x):
        return x
    def gradient(self,input,previous):
        return previous

class Relu(Activation):
    def forward(self,x):
        return np.maximum(0,x)

    def gradient(self,input,previous):
        previous[input<=0]=0
        return previous

class SoftMax(Activation):

    def forward(self,x):
        x = x/x.std()
        e = np.exp(x)
        e = e / np.sum(e, axis=0, keepdims=True)
        return e / np.sum(e, axis=0, keepdims=True)

    def gradient(self, x,previous):
        return previous * (self.forward(x) * (1 - self.forward(x)))

