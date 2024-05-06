# -*- coding: utf-8 -*-
"""
Trabajo Práctico 1:
Grupo : Susano y Valentin
Integrantes: Ibarra, Abril Magalí, Dominguez, Rocio Julieta
Materia: Algebra Lineal Computacional
Periodo. Primer Cuatrimestre 2024
"""

import numpy as np
import networkx as nx
from scipy.linalg import solve_triangular 
import numpy.linalg as lng
import matplotlib.pyplot as plt 

def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
    	line = f.readline()
    	i = int(line.split()[0]) - 1
    	j = int(line.split()[1]) - 1
    	W[j,i] = 1.0
    f.close()
    
    return W

def dibujarGrafo(W,titulo=None, print_ejes=True,ax=None):
    
    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}
    
    N = W.shape[0]
    G = nx.DiGraph(W.T)
    
    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])
#Modificamos la funcion original para poder agrupar en un mismo grafo
    if ax is None:
        fig, ax = plt.subplots()
    
    nx.draw(G, pos=nx.spring_layout(G), ax=ax, **options)
    
# Agregar título si se recibe como input
    if titulo:
        ax.set_title(titulo)


#Implementamos la factorizacion LU 
#Este codigo asume que no se necesitan permutaciones de filas

def factorizacion_LU(A):
    m, n = A.shape
    
    #verificamos si la matriz es cuadrada
    if m != n:
        print('Matriz no cuadrada, no es posible factorizar')
        return None, None
    
    L = np.eye(n) #iniciamos L
    U = A.copy() #U será la matriz A, triangulada
    
    for k in range(n-1): #iteramos sobre las columnas
        pivote = U[k, k]  #elegimos el pivote siempre como el elemento diagonal 
        for f in range(k+1, n):
            coef = U[f, k] / pivote  #guardamos el coeficiente utilizado para triangular
            L[f, k] = coef #lo colocamos debidamente en L
            U[f, k:] -= coef * U[k, k:] #actualizamos los demás elementos de U
            
    return L, U

# Comenzamos con el calculo de PageRank


def calcular_grado(matriz,i):
    return np.sum(matriz[:, i]) #El grado de una pagina serán la suma de sus links

def componentes_Pagerank(W, p):
    """
    Dada una matriz de conectividad W y una probabilidad p,
    calcula las componentes de la ecuación 6.
    """
    n = W.shape[0] 
    e = np.ones((n, 1))
    D = np.zeros((n, n))
    
    for i in range(n):
        grado = calcular_grado(W, i)
        if grado != 0:
            D[i][i]= 1 / grado

    z = np.where(np.sum(W, axis=0) == 0, 1/n, (1-p)/n)
    z_t = z.T 
    return p, W, e, D, z_t


#calculamos los puntajes (score) resolviendo ecuacion 6
#implementando la descomposicion LU
def calcularPuntajes(W,p):
    p, W, e, D, z_t = componentes_Pagerank(W, p)
    identidad = np.eye(W.shape[0])
    
    if p != 0:
        M = identidad - p * (W @ D)
    else:
        M = W @ D - identidad
        if (lng.det(M) == 0):
            raise ValueError("R-I es singular. Hay infinitas soluciones")
        e = np.zeros((W.shape[0], 1)) 
    
    L, U = factorizacion_LU(M)
    
    y = solve_triangular(L, e, lower=True)
    x = solve_triangular(U, y, lower=False)
    if np.sum(x) == 0:
        return x
    return x / np.sum(x)


def calcularRanking(M, p):
    npages = M.shape[0]
    rnk = np.arange(0, npages) # ind[k] = i, la pagina k tienen el iesimo orden en la lista.
    scr = np.zeros(npages) # scr[k] = alpha, la pagina k tiene un score de alpha 
    # Codigo
    x = calcularPuntajes(M, p)
    scr = x
    pagina_score = sorted(enumerate(x.flatten()), key=lambda x: x[1], reverse=True)
    for t, (pagina, _) in enumerate(pagina_score):
        rnk[pagina] = t + 1
    return rnk, scr 

#En rnk, el indice + 1 (pues los indices comienzan en cero) es la pagina y
# el elemento es la posicion que tiene en el ranking

#La siguiente funcion, obtiene el numero de página que tuvo el mejor puntaje
#Es decir, aquella que en el ranking tiene poscion  1
def obtenerMejorPagina(M,p):
    mejor_pagina = 0
    rnk,scr = calcularRanking(M, p)
    print('rnk:' ,rnk)
    for i in range(len(rnk)):
         if (rnk[i]==1):
             mejor_pagina = i+1
    return mejor_pagina

def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)
    
    return output

#ARCHIVOS DE ENTRADA
archivo_test = './tests/test_dosestrellas.txt'
#CARGA DE ARCHIVO EN GRAFO
W = leer_archivo(archivo_test)
dibujarGrafo(W,titulo = 'Grafo Dos Estrellas', print_ejes=False)

#%%

# defino la probabilidad de salto de continuar los links de la pagina actual
p = 0.5

print(obtenerMaximoRankingScore(W,p))
# Realizo el test unitario para el calculo del mayor score, que pruebe que el codigo funciona correctamente.
print('*'*50)
print('Test unitario 1')
try:
    assert(np.isclose(int(obtenerMaximoRankingScore(W,p)*(10**4))/(10**4), 0.1811))
except:
    print('OUCH!! - No paso el test unitario')
else:
    print('BIEN! - Paso correctamente el test unitario')
print('*'*50)
