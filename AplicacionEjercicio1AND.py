# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:10:54 2020

@author: SEBAS
"""

import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

dataSwiches = np.array(pn.read_csv('trainAND.csv'))
X_train = dataSwiches[:,1:]
Y_train = dataSwiches[:,0]
Z_train = dataSwiches[:,0]
Y_train = Y_train[:,np.newaxis]

f = X_train.shape[1]  
c = Y_train.shape[1]  
w = np.random.randn(f,c) 
b = 1
#plt.plot(X_train.T ,'o') 
#plt.show()
epocas = 5

def fnActivacion(z) :
    if z>0:
        return 1
    else :
        return -1
    
def train(w_,x_,b_,y_, train_=True) :
    for i in range(len(x_)):
        a0 = x_[i, np.newaxis]
        y0 = y_[i, np.newaxis]
            
#        print(a0,a0.shape, w_.shape, y0.shape)
        
        z = np.dot(a0, w_)+b_
        Y_modelo = fnActivacion(z)
        Y_costo[i] = Y_modelo
#        print(Y_modelo)        
        if Y_modelo != y0 and train_:
#            print(y0.shape)
#            print(a0.shape)
            error = np.dot(y0,a0)
#            print('e'+str(error.shape))
            w_ = w_+error.T
#            print(w_.shape)
            b_ = b_+y0
    return Y_modelo, w_, b_    

Y_costo = np.zeros((Y_train.shape[0],Y_train.shape[1]))
Loss = []

for j in range(epocas):
    Y_modelo,w,b =train(w, X_train, b,Y_train)
    print(Y_train, Y_costo)
    Loss.append(np.mean(Y_costo-Y_train)**2)
    
    Resolucion = 50
    _x0 = np.linspace(-2,2,Resolucion)
    _x1 = np.linspace(-2,2,Resolucion)
    _y0 = np.zeros((Resolucion, Resolucion))
    
    
    for i0, x0 in enumerate(_x0):
        for i1, x1 in enumerate(_x1):
            X_predict = np.array([[x0,x1]])
            _y0[i0,i1],_,_ = train(w, X_predict[:, np.newaxis], b, np.array([1]), False)
    
    plt.pcolormesh(_x0, _x1, _y0, cmap = "coolwarm")
    plt.axis('equal')
    plt.plot(X_train.T,'o')
    plt.show()
    
#    plt.plot(range(len(Loss)),Loss)
#    plt.show()


X_predict = np.array([[-1,-1]])
Y_predict,_,_ = train(w, X_predict[:, np.newaxis], b, np.array([1]), False)
print("Entrada del modelo "+str(X_predict))
print("Salida del modelo "+str(Y_predict))


