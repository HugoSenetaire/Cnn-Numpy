
import os
import numpy as np
# from smart_open import open
# class MyCorpus(object):
#     def __init__(self,path,batch_size):
#         self.path = path
#         self.batch_size = batch_size
#     def __iter__(self):
#         compteur = 0

#         for line in open(self.path):
#             if line.startswith("label"):
#                 continue
#             X = []
#             if compteur<self.batch_size :
#                 compteur+=1
#                 X.append()
#             else :
#                 compteur = 0
                
#                 yield

def parser(path):
    with open(path) as f:
        firstLine = True
        X = []
        label = []
        Train = True
        compteur = 0
        for line in f.readlines():
            if compteur ==100:
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
        X = np.array(X).reshape(-1,28,28).astype(int)
        if Train :
            return label,X
        else : 
            return None,X


