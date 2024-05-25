
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
import time
import random
import os
import seaborn as sns
from itertools import permutations
import pandas as pd
import matplotlib.patches as mpatches


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


############################################################################
#               Funciones para el cálculo de Page Rank
############################################################################
"""
La siguiente funcion implementa la descomposicion LU de matrices.
Toma como entrada una matriz A Y retorna dos matrices, L y U
Esta funcion asume que no será necesario el intercambio de filas
"""
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

"""
El grado de una pagina serán la suma de sus links
Las paginas, son representadas por las columnas, por lo que sumamos por columna
"""
def calcular_grado(matriz,i):
    return np.sum(matriz[:, i]) 

"""
    Dada una matriz de conectividad W y una probabilidad p,
    calcula las componentes de la ecuación 6.    
"""

def componentes_Pagerank(W, p):
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

"""
La siguiente funcion calcula los puntajes (score) resolviendo la ecuacion 6
Para ello toma una amtriz de conectividad y una probabilidad p
Luego calcula los componentes necesarios con la funcion anterior y resuelve el sistema
a traves de la factorizacion LU
Finalmente normaliza el vector X solucion y lo retorna
"""

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

"""
La siguiente funcion calcula el ranking, luego de obtener los puntajes con la uncion anterior
recibe una matriz de conectividad w y una probabilidad p 
De acuerdo al puntaje obtenido, separa y retorna dos listas, rnk, la cual contiene el puesto en el que quedo cad pagina del ranking,
representandose estas con el indice en el que estan puestos más uno (la pagina uno, esta en el indice cero), y scr que contiene el score de cada pagina

"""
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

"""
La siguiente funcion, obtiene el numero de página que tuvo el mejor puntaje
Es decir, aquella que en el ranking tiene posicion  1
"""
def obtenerMejorPagina(M,p):
    mejor_pagina = 0
    rnk,scr = calcularRanking(M, p)
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


############################################################################
#               Funciones para el Análisis Cualitativo
############################################################################

def leer_test():
    carpeta = './tests/'
    test_dados = ['instagram_famosos_grafo.txt', 'mathworld_grafo.txt',
             'test_15_segundos.txt', 'test_30_segundos.txt',
             'test_aleatorio.txt', 'test_dosestrellas.txt']
    nombres = [item.replace(".txt", "") for item in test_dados]
    test = []
    for t in test_dados:
        archivo = leer_archivo(carpeta + t)
        test.append(archivo)
   
    return test,nombres
    

#armamos un solo grafico con todos los grafos, para eso creamos subplots

def grafos_test():    
   
    test,nombres  = leer_test()
    fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 

    # Dibujamos  cada grafo en un subplot diferente
    for i in range(3):     
        dibujarGrafo(test[i],titulo = nombres[i],print_ejes= False,ax=axs[0,i])
    for i in range(3,6):     
        dibujarGrafo(test[i],titulo = nombres[i],print_ejes= False,ax=axs[1,5-i])

  
    plt.tight_layout()
     
    plt.show()

def obtenerPosicionEnRanking(w,p,pagina):
    rnk,scr = calcularRanking(w, p)
    posicion = rnk[pagina-1]  #Como la pagina es el indice + 1, resta para poder indexar
    
    return posicion

def obtenerPaginaEnPosicion(w,p,posicion):
    rnk,scr = calcularRanking(w, p)
    pagina=0
    for i in range(len(rnk)):
        if (rnk[i]==posicion): #obtenemos la pagina en la posicion de ranking buscada
            pagina = i + 1
    return pagina

def obtenerIndiceMejorPagina(w,p): 
    mejor_pagina = obtenerMejorPagina(w, p) 
    indice=obtenerPaginaEnPosicion(w, p, 1) 
    return indice-1

#Funcion utilizada para realizar los graficos

def ranking_pagina_vs_p(w, leyenda, color, ax):
    p_valores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
   
    mejores_paginas = []
    for p in p_valores:
        rnk, scr = calcularRanking(w, p)
        mejor_pagina = obtenerMejorPagina(w, p)
        mejores_paginas.append(mejor_pagina)
    ax.plot(p_valores, mejores_paginas, label=leyenda, color=color)
   
    ax.set_xlabel('Valor de p')
    ax.set_ylabel('Número de página con mejor ranking')
    ax.legend(loc='upper right')
    
# Funcion utilizada para realizar los graficos
def ranking_score_vs_p(w, leyenda, color, ax):
    p_valores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    scores = []
    for p in p_valores:
        mejor_score = obtenerMaximoRankingScore(w, p)
        scores.append(mejor_score)
    ax.plot(p_valores, scores, label=leyenda,marker='o', color=color)
   
    ax.set_xlabel('Valor de p')
    ax.set_ylabel('Puntaje de la página con mejor ranking')
    ax.legend(loc='upper right')

#Funciones exploratorias 
def obtenerPuntajeDePagina(matriz,p,pagina):
    rnk,scr = calcularRanking(matriz, p)
    puntaje = scr[pagina]
    return puntaje

def cant_link_asociados(matriz,pagina):
    links_asociados = np.sum(matriz[:, pagina-1])
    return links_asociados

def links_asociados(matriz,pagina):
    links = []
    for i in range(len(matriz)):
        if (matriz[i][pagina-1] == 1):
            links.append(i + 1)
    return links

def grafico_mejor_ranking_vs_p():
    #Creamos Grafico de Mejor Pagina vs P
    p_valores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    test,nombres  = leer_test()
    fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 
    color = ['red','blue','green','orange','purple','brown']
    # Dibujamos  cada grafo en un subplot diferente
    for i in range(3):     
        ranking_pagina_vs_p(test[i],leyenda = nombres[i],color = random.choice(color),ax=axs[0,i])
    for i in range(3,6):     
        ranking_pagina_vs_p(test[i],leyenda = nombres[i],color = random.choice(color),ax=axs[1,5-i])
  
   
 
    plt.show()
    

def grafico_score_vs_p():
    p_valores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    test,nombres  = leer_test()
    fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 
    color = ['red','blue','green','orange','purple','brown']
    # Dibujamos  cada grafo en un subplot diferente
    for i in range(3):     
        ranking_score_vs_p(test[i],leyenda = nombres[i],color = random.choice(color),ax=axs[0,i])
    for i in range(3,6):     
        ranking_score_vs_p(test[i],leyenda = nombres[i],color = random.choice(color),ax=axs[1,5-i])
  
 
    plt.show()


def calcular_densidad(W):
    n = W.shape[0]  # Número de nodos en el grafo
    aristas_presentes = np.sum(W)  # Número de aristas presentes en la matriz de conectividad w
    aristas_posibles = n * (n - 1)  # Número de aristas posibles en un grafo completo

    densidad = aristas_presentes / aristas_posibles
    return densidad

############################################################################
#               Funciones para la generacion de Test Propios
############################################################################

"""
La siguiente función genera y guarda un archivo .txt en el cual guarda
la cantidad de nodos (numero de paginas proporcionadas como parametro)
y enlaza todas las paginas entre sí
"""
def generar_test_todos_conectados(nombre_archivo, num_paginas):
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(str(num_paginas) + '\n')
        #conectamos los enlaces
        num_enlaces = num_paginas * (num_paginas - 1)
        archivo.write(str(num_enlaces) + '\n')
        #escribimos ignorando autolinks
        for i in range(1, num_paginas + 1):
            for j in range(1, num_paginas + 1):
                if i != j:
                    archivo.write(f"{i} {j}\n")



"""
La siguiente función genera y guarda un archivo .txt en el cual guarda
la cantidad de nodos (numero de paginas proporcionadas como parametro)
y  pero no enlaza ninguna
"""
def generar_test_ninguno_conectado(nombre_archivo, num_paginas):
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(f"{num_paginas}\n") #numero de paginas
        archivo.write("0\n")  #No hay enlaces

"""
Dado un numero de paginas y un nombre, genera un archivo .txt, en el cual
se representan los links de una matriz de conectividad de relaciòn ciclica
"""
def generar_test_ciclico(nombre_archivo, num_paginas):
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(str(num_paginas) + '\n')  
        num_enlaces = num_paginas   
        archivo.write(str(num_enlaces) + '\n')

        for i in range(1, num_paginas):
            archivo.write(f"{i} {i+1}\n")
        archivo.write(f"{num_paginas} 1\n")


############################################################################
#               Funciones para el Análisis Cuantitativo
############################################################################


