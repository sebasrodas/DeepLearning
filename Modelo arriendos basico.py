# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 20:22:57 2019

@author: SEBAS - 'litox
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataCasas = pd.read_csv('housing.csv')

X_train = np.array(dataCasas["RM"])
Y_train = np.array(dataCasas["MEDV"])

#asignacion de titulo de la grafica
plt.title("Modelo cálculo de valor de arriendo por habitaciones")
# creacion de la grafica asignacion de valor en X, Y y la enfoque de los puntos 
plt.plot(X_train, Y_train, 'o',alpha=0.3)
# asignacion del titulo del eje x
plt.xlabel('Numero Habitaciones')
# asignacion del titulo del eje y
plt.ylabel('Valor Arriendo')

#p_train = X_train

X_train = np.array([np.ones(len(X_train)),X_train]).T
#formula m = (X.Traspuesta*X).inversa*X.Traspuesta*Y esto nos da los valores de m(pendiente de la recta) y b(inicio de la recta) que sirven para crear la recta
vectorMB = np.linalg.inv((X_train.T@X_train))@X_train.T@Y_train

B = vectorMB[0]
M = vectorMB[1]

plt.plot([4,8], [M*4+B,M*8+B],c='red')
plt.show()
x_prediction = 5.5
y_prediction = M*x_prediction+B
print('Valor promedio de '+str(x_prediction)+' habitaciones es:  $'+str(y_prediction))
