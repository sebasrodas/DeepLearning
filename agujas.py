# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 21:54:27 2019

@author: SEBAS
"""

import numpy as np
import matplotlib.pyplot as plt


x_train = np.array([[1,1,1]
                    , [-1,-1,1]
                    , [1,-1,-1]
                    , [-1,1,-1]
                    , [-1,-1,-1]])
y_train = np.array([[-1]
                    ,[1]
                    ,[1]
                    ,[1]
                    ,[-1]])

numero_entradas = x_train.shape[1]
numero_neuronas = 2
numero_clases = 1
numero_epocas = 5
numero_aciertos = 0
fncosto= []


w1 = np.random.randint(10,size=(numero_entradas, numero_neuronas))*0.034
w2 = np.random.randint(10,size=(numero_neuronas, numero_clases))*0.034
b1 = np.ones([1,numero_neuronas])
b2 = np.ones([1,numero_clases])
x = np.zeros([1,numero_entradas])

plt.plot(x_train, y_train, 'o' )
plt.show()

def fnActivacion(z):
    z[z>0]=-1
#    y_modelo = np.ones([len(z),1])
#    
#    if z > 0:
#        y_modelo = 1
    return z

def forward(x):
    z1 = np.dot(x,w1)+b1
#    print(z1.size)
    y1 = fnActivacion(z1)
#    print(len(y1))
    z2 = np.dot(y1,w2)+b2
    y_modelo = fnActivacion(z2)
#    print(y_modelo)
    return y_modelo
    

for j in range(numero_epocas):
    numero_aciertos = 0
    for i in range(len(x_train)):
         x[0, :] = x_train[i]
         y_real = y_train[i] 
         y_modelo = forward(x)
         if y_modelo != y_real:
            error = y_real*x.T
            w = w+error
            b = b+y_real*1
         else :
            numero_aciertos = numero_aciertos+1             

    fncosto.append(numero_aciertos/len(x_train))
    print("en la epoca "+str(j+1)+" tuvimos "+str(numero_aciertos)+" aciertos de "+str(len(x_train)))
    if len(x_train) == numero_aciertos :
       print(w)
       print(b)
       break
       
x_prediction = np.array([-1,1,-1])
y_prediction = forward(w,b, x_prediction)
print("la respuesta del modelo es: "+ str(y_prediction))
