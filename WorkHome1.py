# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:09:28 2020

@author: SEBAS
"""

import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

Umbral = 0
NumeroEpocas = 0
NumeroNeuronas = 0
b = 1 


def IndexarMatriz(Vector):
    return Vector[:,np.newaxis]

def CargaData(strArchivo):
    return np.array(pn.read_csv(strArchivo))

def Activacion(Z, indModo) :
    if indModo == 1:
        if Z>Umbral:
            return 1
        else :
            return -1
    else:
        return 1/(1 + np.exp(-Z)) 

def CalcularZ(w,x,b) :
    return np.dot(x, w)+b
 

def train(w_,x_,b_,y_, train_=True) :
    for i in range(len(x_)):
        a = x_[i, np.newaxis]
        y = y_[i, np.newaxis]
          


        Y_modelo = Activacion(CalcularZ(w_, a,b_),2)
        print(Y_modelo.shape)
        print(Y_modelo)
        Y_costo[i] = Y_modelo
        print(Y_modelo)        
        if Y_modelo != y and train_:
#            print(y0.shape)
#            print(a0.shape)
            error = np.dot(y,a)
#            print('e'+str(error.shape))
            w_ = w_+error.T
#            print(w_.shape)
            b_ = b_+y
    return Y_modelo, w_, b_ 
#    return x_, w_, b_    



dataSwiches = CargaData('trainAND2.csv')
X_train = dataSwiches[:,1:]
Y_train = IndexarMatriz(dataSwiches[:,0])


Neuronas = Y_train.shape[1] 
Entradas = X_train.shape[1] 


w = np.random.randn(Entradas, Neuronas) 
Y_costo = np.zeros((Y_train.shape[0],Y_train.shape[1]))

X_predict = np.array([[-1,-1,-1,-1]])
X_predict = (X_predict)
#Y_predict,_,_ = train(w, X_predict, b, np.array([1]), False)
Y_predict,_,_ = train(w, X_predict, b, np.array([1]), False)
#print("Entrada del modelo "+str(X_predict))
#print("Salida del modelo "+str(Y_predict))
#
#print(Y_train)
