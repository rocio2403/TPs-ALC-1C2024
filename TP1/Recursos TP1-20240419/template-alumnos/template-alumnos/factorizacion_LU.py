# -*- coding: utf-8 -*-
"""
Trabajo Práctico 1:
Grupo : Susano y Valentin
Integrantes: Ibarra, Abril Magalí, Dominguez, Rocio Julieta
Materia: Algebra Lineal Computacional
Periodo. Primer Cuatrimestre 2024
"""

import numpy as np

def factorizacion_LU(A):
    m=A.shape[0]  #Numero de filas
    n=A.shape[1]   #Numero de columnas
    matriz = A.copy() #Para no modificar la matriz original
    
    if m!=n:
        print('Matriz no cuadrada, no es posible factorizar')
        
    else:
        L = np.eye(n) #inicializamos L
        for k in range(n-1):
            pivote = matriz[k][k]    #Tomamos un pivote
            # print('--------------------')
            # print(pivote)
    
            for f in range(k+1, n):   #Iteramos sobre esa columna
                print(matriz[f][k])
                coef = matriz[f][k] / pivote
                L[f][k] = coef         #Guardamos coeficiente
                for c in range(k, n):
                    matriz[f][c] = matriz[f][c] + ((-coef)*matriz[k][c]) #Actualizamos los demas nros de la matriz
                    
        U = matriz #Guardamos como u, la matriz triangulada
    return L, U

#%%
A = np.array([[2,1,2,3],[4,3,3,4],[-2,2,-4,-12],[4,1,8,-3]])

L, U = factorizacion_LU(A)

print('-------------------------')
print('Matriz L: \n', L)
print('Matriz U: \n',  U)

#%%
    