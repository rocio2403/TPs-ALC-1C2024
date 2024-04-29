# -*- coding: utf-8 -*-
"""
Trabajo Práctico 1:
Grupo : Susano y Valentin
Integrantes: Ibarra, Abril Magalí, Dominguez, Rocio Julieta
Materia: Algebra Lineal Computacional
Periodo. Primer Cuatrimestre 2024
"""
import numpy as np 
from scipy.linalg import solve_triangular

from funciones import *
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

def calcular_grado(matriz,i):
    res =  np.sum(matriz[:, i])
    return res

def page_rank(w, p):
    n = w.shape[0]
    #creamos e
    e = np.ones(shape=n)
    #creamos D
    #para eso calculamos Cj
    D = np.eye(n)
    for i in range(len(D)):
        if calcular_grado(w, i) == 0:
           D[i][i]=0
        else:
          D[i][i] = 1 / calcular_grado(w, i)  #el grado es la suma sobre la columna
        
    #creamos z y luego trasponemos
    z = np.zeros(shape=n)
    for j in range(n):
        if calcular_grado(w, j) == 0:
            z[j] = (1-p)/n
        else:
            z[j] = 1/n
    #ahora trasponemos z
    z_t =  np.transpose(z)
    #Ahora calculamos A
    print(p)
    print(w)
    print(e)
    print(D)
    print(z_t)
    return p, w, e, D, z_t


M = np.array([[1,0,1],[1,0,0],[1,0,1]])
page_rank(M, 0.3)

#habria que llamarlo de otra forma, porque asi no obtenemos el pagerank,
#sino las componentes de A, si es que solo nos dan M y no nos dan A, 
#si nos dan A podemos despejar
def ecuacion_6(w,p):
    e,D,z_t = page_rank(w, p)
    #resolvemos la ecuacion 6
    #para eso podemos armar la matriz con los datos obtenidos
    #Luego hacer factorizacion LU
    #resolver el sistema con L y = b , U X = y
    #así obtenemos x, ya que asumimos que gamma(?es 1
    #por lo que nuestro sistema queda :
    #  (I - PWD)X = e
    #Por lo cual buscamos la matriz I-PWD
    identidad = np.eye(w.shape[0])
    A = identidad - p*(w@D)
    L, U = factorizacion_LU(A)
    
    #resolvemos sistema ya que son matrices triangulares
    y = solve_triangular(L, e)
    x = solve_triangular(U, y)
    
    return x
    

    
#ARCHIVOS DE ENTRADA
archivo_test = './tests/test_dosestrellas.txt'

#CARGA DE ARCHIVO EN GRAFO
W = leer_archivo(archivo_test)

dibujarGrafo(W, print_ejes=False)

# defino la probabilidad de salto de continuar los links de la pagina actual
p = 0.5
# Realizo el test unitario para el calculo del mayor score, que pruebe que el codigo funciona correctamente.
print('*'*50)
print('Test unitario 1')
try:
    assert(np.isclose(obtenerMaximoRankingScore(W, p), 0.1811))
except:
    print('OUCH!! - No paso el test unitario')
else:
    print('BIEN! - Paso correctamente el test unitario')
print('*'*50)
    
    
    
    