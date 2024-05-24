
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
#               Funciones para el Análisis Cuantitativo
############################################################################

def generar_matriz(n, m):
    """
    Genera una matriz cuadrada de tamaño nxn con m unos y el resto de entradas ceros.
    """
    # Inicializar una matriz de ceros
    W = np.zeros((n, n))
    
    # Llenar m posiciones aleatorias con unos
    indices = np.random.choice(n*n, m, replace=False)
    W.flat[indices] = 1
    
    return W


def generar_matrizW_no_singular(n, m):
    """
    Genera una matriz cuadrada no singular de tamaño nxn con m unos y el resto ceros.
 
    Advertencia, m no puede ser cualquier número. El valor de m para garantizar que
    la matriz generada sea no singular con respecto a n es desde n (rango de W) hasta
    n*2 -1.
    """
    # Inicializar una matriz cuadrada de ceros
    W = np.zeros((n, n))
    
    # Colocar m unos aleatoriamente en la matriz
    indices = np.random.choice(n*n, m, replace=False)
    W.flat[indices] = 1
    
    # Asegurar que la matriz sea no singular
    while np.linalg.matrix_rank(W) != n:
        indices = np.random.choice(n*n, m, replace=False)
        W = np.zeros((n, n))
        W.flat[indices] = 1
    
    return W


# def calcularTiempoDeProcesamiento(metodo, W):
#     """
#         Obtenemos el tiempo de procesamiento en función del tamaño del grafo de páginas (n) y de 
#         la cantidad de links del mismo (m).
        
#         Para el método 0 se resuelve la ecuación 3.
#         Para el método 1 (navegante aleatorio) se resuelve la ecuación 6 con un p fijo.
        
#         Devuelve un diccionario con los datos de la muestra W
#     """
#     n = W.shape[0]
#     m = int(W.sum())
#     p = 0.5
    
#     if metodo == 1:
#         resultado = %timeit -o -r 1 -n 1 calcularRanking(W, p)
#     else:
#         resultado = %timeit -o -r 1 -n 1 calcularRanking(W, 0)
    
#     tiempoProcesamiento = round(resultado.best * 1000, 2)

#     registro = {'metodo':metodo, 'paginas':n, 'links':m, 'tiempoProcesamiento':tiempoProcesamiento}    
#     return registro
    
############################################################################
#               Funciones para el Análisis Cualitativo
############################################################################

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
La siguiente funcion genera y guarda un archivo .txt que represetna a una matriz
donde una apgina està conectada con la siguiente y la ultima con la primera, generando un ciclo
"""
def generar_test_ciclico(nombre_archivo, num_paginas):
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(str(num_paginas) + '\n')  
        num_enlaces = num_paginas   
        archivo.write(str(num_enlaces) + '\n')

        for i in range(1, num_paginas):
            archivo.write(f"{i} {i+1}\n")
        archivo.write(f"{num_paginas} 1\n")

#%%
##############################################################################
#
# A continuación se adjunta el uso del codigo para los análisis y Test Propios
#
#############################################################################


#============================
# Análisis Cuantitativo
#===========================

# #-------------------------------- MÉTODO NAVEGANTE ALEATORIO
# # Gráfico cantidad de links vs tiempo de procesamiento para cantidad de páginas fija

# cantPaginas = 15
# max_cantLinks = cantPaginas * (cantPaginas-1)

# # Almacenar tiempos para el método 1
# tiempos_pagina_fija = []
# for i in range(0, max_cantLinks+1, 1):
#     W = generar_matriz(cantPaginas, i)
#     registro = calcularTiempoDeProcesamiento(1, W)
#     tiempos_pagina_fija.append(registro['tiempoProcesamiento'])

# # Gráfico cantidad de páginas vs tiempo de procesamiento para cantidad de links fijo
# carpeta = './tests_linksfijo/'

# min_cantPags = 5
# max_cantPags = min_cantPags*6
# cantLinks = min_cantPags * (min_cantPags-1)

# # Almacenar tiempos para el método 2
# tiempos_link_fijo = []
# for i in range(min_cantPags, max_cantPags+1, 1):
#     W = generar_matriz(i, cantLinks)
#     registro = calcularTiempoDeProcesamiento(1, W)
#     tiempos_link_fijo.append(registro['tiempoProcesamiento'])


# df_pagina_fija = pd.DataFrame({
#     'Método': [f'Páginas fijas\nCantidad de páginas: {cantPaginas}\nRango de links: 0-{max_cantLinks}'] * len(tiempos_pagina_fija),
#     'Cantidad de Links': list(range(0, max_cantLinks+1, 1)),
#     'Tiempo de ejecución (ms)': tiempos_pagina_fija
# })

# df_link_fijo = pd.DataFrame({
#     'Método': [f'Links fijos\nCantidad de links: {cantLinks}\nRango de páginas: {min_cantPags}-{max_cantPags}'] * len(tiempos_link_fijo),
#     'Cantidad de Páginas': list(range(min_cantPags, max_cantPags+1)),
#     'Tiempo de ejecución (ms)': tiempos_link_fijo
# })

# # Combinar DataFrames
# df_combined = pd.concat([df_pagina_fija, df_link_fijo])

# # Crear gráficos de caja con Seaborn
# plt.figure(figsize=(8, 6))

# sns.boxplot(x='Método', y='Tiempo de ejecución (ms)', hue='Método', data=df_combined, palette="Set2", legend=False)
# plt.title('Tiempo de ejecución variando cantidad de páginas y links')
# plt.xlabel('NAVEGANTE ALEATORIO', labelpad=30)
# plt.ylabel('Tiempo de ejecución (ms)')

# plt.grid(True)
# plt.show()

# #-------------------------------- MÉTODO ESTÁNDAR
# """
# Para este método tenemos el problema de que la matriz R-I no siempre es no singular,
# por lo que no se garantiza que siempre exista factorización LU. 
# Como Rx=x <-> (R-I)x=0 es homogeneo, siempre hay solución y si det(R-I)=0 las soluciones
# son infinitas.  

# Para poder estudiar los tiempos de ejecución con este método adecuamos la generación de matrices
# de conectividad para que siempre resulte en R-I no singulare y haya una única solución. 

# """

# # Gráfico cantidad de links vs tiempo de procesamiento para cantidad de páginas fija

# cantPaginas = 5
# max_cantLinks = cantPaginas *2 - 1
# min_cantLinks = cantPaginas
# # Almacenar tiempos para el método 1
# tiempos_pagina_fija = []
# for i in range(min_cantLinks, max_cantLinks+1, 1):
#     W = generar_matrizW_no_singular(cantPaginas, i)
#     registro = calcularTiempoDeProcesamiento(1, W)
#     tiempos_pagina_fija.append(registro['tiempoProcesamiento'])

# # Gráfico cantidad de páginas vs tiempo de procesamiento para cantidad de links fijo

# min_cantPags = 5
# max_cantPags = min_cantPags*2 -1
# cantLinks = min_cantPags*2 -1

# # Almacenar tiempos para el método 2
# tiempos_link_fijo = []
# for i in range(min_cantPags, max_cantPags+1, 1):
#     W = generar_matrizW_no_singular(i, cantLinks)
#     registro = calcularTiempoDeProcesamiento(1, W)
#     tiempos_link_fijo.append(registro['tiempoProcesamiento'])


# df_pagina_fija = pd.DataFrame({
#     'Método': [f'Páginas fijas\nCantidad de páginas: {cantPaginas}\nRango de links: 0-{max_cantLinks}'] * len(tiempos_pagina_fija),
#     'Cantidad de Links': list(range(min_cantLinks, max_cantLinks+1, 1)),
#     'Tiempo de ejecución (ms)': tiempos_pagina_fija
# })

# df_link_fijo = pd.DataFrame({
#     'Método': [f'Links fijos\nCantidad de links: {cantLinks}\nRango de páginas: {min_cantPags}-{max_cantPags}'] * len(tiempos_link_fijo),
#     'Cantidad de Páginas': list(range(min_cantPags, max_cantPags+1)),
#     'Tiempo de ejecución (ms)': tiempos_link_fijo
# })

# # Combinar DataFrames
# df_combined = pd.concat([df_pagina_fija, df_link_fijo])

# # Crear gráficos de caja con Seaborn
# plt.figure(figsize=(8, 6))

# sns.boxplot(x='Método', y='Tiempo de ejecución (ms)', hue='Método', data=df_combined, palette="Set2", legend=False)
# plt.title('Tiempo de ejecución variando cantidad de páginas y links')
# plt.xlabel('MÉTODO ESTÁNDAR', labelpad=30)
# plt.ylabel('Tiempo de ejecución (ms)')

# plt.grid(True)
# plt.show()



# #-------------------Tests dados
# carpeta = './tests/'
# tests = ['instagram_famosos_grafo.txt', 'mathworld_grafo.txt', 'test_15_segundos.txt', 'test_30_segundos.txt', 'test_aleatorio.txt', 'test_dosestrellas.txt']
# paginas = []
# links = []
# #NAVEGANTE ALEATORIO
# tiempos_tests1 = []
# #MÉTODO ESTANDAR
# tiempos_tests0 = []


# # Crear DataFrame
# df = pd.DataFrame({
#     'Test': tests,
#     'Cantidad de Páginas': paginas,
#     'Cantidad de Links': links,
#     'Tiempo Método Estándar': tiempos_tests0,
#     'Tiempo Navegante Aleatorio': tiempos_tests1
# })
# # Crear gráfico de burbujas
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plotear burbujas
# scatter1 = sns.scatterplot(data=df, x='Cantidad de Links', y='Cantidad de Páginas', size='Tiempo Método Estándar', sizes=(100, 2000), alpha=0.7, color='red', ax=ax)
# scatter2 = sns.scatterplot(data=df, x='Cantidad de Links', y='Cantidad de Páginas', size='Tiempo Navegante Aleatorio', sizes=(100, 2000), color='pink', alpha=0.5, ax=ax)

# # Crear subgráfico para el zoom
# ax_zoom = plt.axes([0.2, 0.55, .3, .3], facecolor='white')  # Ajustar posición del zoom
# scatter_zoom1 = sns.scatterplot(data=df, x='Cantidad de Links', y='Cantidad de Páginas', size='Tiempo Método Estándar', sizes=(300, 1500),color='red', alpha=0.7, ax=ax_zoom)
# scatter_zoom2 = sns.scatterplot(data=df, x='Cantidad de Links', y='Cantidad de Páginas', size='Tiempo Navegante Aleatorio', sizes=(300, 1500),color='pink', alpha=0.5, ax=   ax_zoom)
# ax_zoom.set_xlim([3, 190])
# ax_zoom.set_ylim([2, 35])

# # Eliminar etiquetas de ejes y leyenda del subgráfico de zoom
# ax_zoom.set_xlabel('')
# ax_zoom.set_ylabel('')
# ax_zoom.legend().set_visible(False)

# # Crear una leyenda personalizada con burbujas de tamaño específico
# legend_elements = [
#     plt.scatter([], [], s=100, color='red', label='Método Estándar'),
#     plt.scatter([], [], s=100, color='pink', label='Navegante Aleatorio')
# ]
# ax.legend(handles=legend_elements, loc='lower right', fontsize='large')
# ax.set_xlim([-500,19000])
# ax.set_ylim([-100,3500])

# # Centrar el título horizontalmente
# title = plt.title('Tiempo de ejecución en función de la cantidad de links y páginas', pad=40)
# title.set_position([.0, 1.04])
# plt.show()

# for t in tests:
#     W = leer_archivo(carpeta+t)
#     datos = calcularTiempoDeProcesamiento(1, W)
#     tiempos_tests1.append(datos['tiempoProcesamiento'])
#     W = leer_archivo(carpeta+t)
#     datos = calcularTiempoDeProcesamiento(0, W)
#     tiempos_tests0.append(datos['tiempoProcesamiento'])
#     paginas.append(datos['paginas'])
#     links.append(datos['links'])
    
#%%

#============================
# Análisis Cualitativo
#===========================

# Obtenemos la matriz para test

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

  
    plt.show()


#Ahora analizamos como varía la pagina mejor rankeada de acuerdo a p
#Para eso utilizamos la funcion calcular ranking del modulo funciones

#----------------

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
    

#Ahora analizamos como varía el puntaje de la mejor pagina de acuerdo a p y graficamos 
   
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

