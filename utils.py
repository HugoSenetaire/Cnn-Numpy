import numpy as np
import copy

def get_batches(data, labels=None, batch_size=256, shuffle=True):
    N = data.shape[0] 
    num_batches = N//batch_size 
    if shuffle:
        shuffled_indices = np.random.permutation(N)
        data = (np.array(data)[shuffled_indices])
        labels = np.array(labels)[shuffled_indices] if labels is not None else None

    if num_batches == 0:
        if labels is not None:
            yield (data.T, labels) 
        else:
            yield data.T
    for batch_num in range(num_batches):
        if labels is not None:
            yield (data[batch_num*batch_size:(batch_num+1)*batch_size].T,
                  labels[batch_num*batch_size:(batch_num+1)*batch_size]) 
        else:
            yield data[batch_num*batch_size:(batch_num+1)*batch_size].T
    if N%batch_size != 0 and num_batches != 0:
        if labels is not None:
            yield (data[num_batches*batch_size:].T, labels[num_batches*batch_size:]) 
        else:
            yield data[num_batches*batch_size:].T

def oneHotEncoding(labels,num_classes = None):
    if num_classes is None :
        num_classes = np.max(labels-np.min(labels)+1)
    one_hot = np.zeros((num_classes,len(labels)))
    for i,lab in enumerate(labels) :
        one_hot[lab,i]=1
    return one_hot


def pad(x,pad_w,pad_h,pad_type = "constant"):
    if pad_h >0 or pad_w > 0:
        if len(np.shape(x))>3:
            X_pad = copy.deepcopy(np.pad(x, ((pad_w, pad_w), (pad_h, pad_h),(0, 0), (0, 0)), pad_type))
        else :
            X_pad = copy.deepcopy(np.pad(x, ((pad_w, pad_w), (pad_h, pad_h), (0, 0)), pad_type))
        return X_pad
    else :
        return copy.deepcopy(x)

def parser(path,limitation = float("inf")):
    with open(path) as f:
        firstLine = True
        X = []
        label = []
        Train = True
        compteur = 0
        for line in f.readlines():
            if compteur == limitation:
                break
            compteur+=1
            if firstLine :
                if len(line.split(","))>28*28 :
                    Train = True
                firstLine = False
                continue
            aux = line.split(',')
            if Train :
                X.extend([int(i) for i in aux[1:]])
                label.append(int(aux[0]))
            else : 
                X.extend(aux)
        X = np.array(X).reshape(-1,1,28,28).astype(int)
        if Train :
            return X,label
        else : 
            return X,None

def separate_train_eval(X,labels,proportion = [0.70,0.20,0.10]):
    nb_points = len(X)
    Xtrain = X[:int(nb_points*proportion[0])]
    Xtest = X[int(nb_points*proportion[0]):int(nb_points*(proportion[0]+proportion[1]))]
    Xeval = X[int(nb_points*(proportion[0]+proportion[1])):int(nb_points*(proportion[0]+proportion[1]+proportion[2]))]
    
    Ytrain = labels[:int(nb_points*proportion[0])]
    Ytest = labels[int(nb_points*proportion[0]):int(nb_points*(proportion[0]+proportion[1]))]
    Yeval = labels[int(nb_points*(proportion[0]+proportion[1])):int(nb_points*(proportion[0]+proportion[1]+proportion[2]))]
  
    return Xtrain,Ytrain,Xtest,Ytest,Xeval,Yeval
    
    