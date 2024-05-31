
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
import seaborn as sns
import pandas as pd


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
    n = A.shape[0]
    L = np.eye(n)
    
    for k in range(n-1):
        pivote = A[k, k]
        for f in range(k+1, n):
            coef = A[f, k] / pivote
            L[f, k] = coef
            A[f, k:] -= coef * A[k, k:]
            
    return L, A
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
    D = np.diag(1 / np.maximum(np.sum(W, axis=1), 1))  
    
    z = np.where(np.sum(W, axis=0) == 0, 1/n, (1-p)/n)
    
    return p, W, e, D, z

"""
La siguiente funcion calcula los puntajes (score) resolviendo la ecuacion 6
Para ello toma una matriz de conectividad y una probabilidad p
Luego calcula los componentes necesarios con la función anterior y resuelve el sistema
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
La siguiente funcion calcula el ranking, luego de obtener los puntajes con la funcion anterior
recibe una matriz de conectividad w y una probabilidad p 
De acuerdo al puntaje obtenido, separa y retorna dos listas, rnk, la cual contiene el puesto en el que quedo cada pagina del ranking,
representandose estas con el indice en el que estan puestos más uno (la pagina uno, esta en el indice cero), y scr que contiene el score de cada pagina

"""
def calcularRanking(M, p):
    npages = M.shape[0]
    rnk = np.arange(0, npages) # ind[k] = i, la pagina k tienen el iesimo orden en la lista.
    # Codigo
    scr = calcularPuntajes(M, p)
    pagina_score = sorted(enumerate(scr.flatten()), key=lambda scr: scr[1], reverse=True)
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
#               Funciones para el Análisis Cuantitativo
############################################################################
def generar_matriz(n, m, seed=42):
    """
    Genera una matriz cuadrada de tamaño nxn con m unos y el resto de entradas ceros.
    """
    np.random.seed(seed)
    # Inicializar una matriz de ceros
    W = np.zeros((n, n))
    
    # Llenar m posiciones aleatorias con unos
    indices = np.random.choice(n*n, m, replace=False)
    W.flat[indices] = 1
    
    return W

def calcularTiempoDeProcesamiento(metodo, W, repeticiones=1):
    """
    Obtenemos el tiempo de procesamiento en función del tamaño del grafo de páginas (n) y de 
    la cantidad de links del mismo (m).
    
    Para el método 0 se resuelve la ecuación 3.
    Para el método 1 (navegante aleatorio) se resuelve la ecuación 6 con un p fijo.
    
    Devuelve un diccionario con los datos de la muestra W
    """
    n = W.shape[0]
    m = int(W.sum())
    p = 0.5 if metodo == 1 else 0

    tiempos = []
    for _ in range(repeticiones):
        start_time = time.time()
        calcularRanking(W, p)
        end_time = time.time()
        tiempos.append((end_time - start_time))  # En segundos para evitar errores de redondeo

    tiempoProcesamiento = round(np.mean(tiempos) * 1000, 2)  # Convertir a milisegundos

    registro = {
        'metodo': metodo,
        'paginas': n,
        'links': m,
        'tiempoProcesamiento': tiempoProcesamiento
    }

    return registro    

def datos_graficoComparacion_LinksPaginas():
    np.random.seed(42)
    #-------------------------------- MÉTODO NAVEGANTE ALEATORIO
    # Datos para el gráfico cantidad de links vs tiempo de procesamiento con cantidad de páginas fija
    cantPaginas = 15
    max_cantLinks = cantPaginas * (cantPaginas-1)

    tiempos_pagina_fija = []
    for i in range(0, max_cantLinks+1, 1):
        W = generar_matriz(cantPaginas, i)
        registro = calcularTiempoDeProcesamiento(1, W)
        tiempos_pagina_fija.append(registro['tiempoProcesamiento'])

    # Datos para el gráfico cantidad de páginas vs tiempo de procesamiento con cantidad de links fijo
    min_cantPags = 5
    max_cantPags = min_cantPags*6
    cantLinks = min_cantPags * (min_cantPags-1)

    tiempos_link_fijo = []
    for i in range(min_cantPags, max_cantPags+1, 1):
        W = generar_matriz(i, cantLinks)
        registro = calcularTiempoDeProcesamiento(1, W)
        tiempos_link_fijo.append(registro['tiempoProcesamiento'])
    return cantPaginas, max_cantLinks, tiempos_pagina_fija, min_cantPags, max_cantPags,cantLinks, tiempos_link_fijo

def graficoComparacion_LinksPaginas(): 
    cantPaginas = 15
    max_cantLinks = cantPaginas * (cantPaginas-1)
    min_cantPags = 5
    max_cantPags = min_cantPags*6
    cantLinks = min_cantPags * (min_cantPags-1)
    tiempos_pagina_fija = [1.83, 1.31, 1.31, 1.28, 1.33, 1.22, 1.82, 1.83, 2.15, 1.49, 1.59, 2.12, 2.0, 1.62, 1.82, 1.8, 1.79, 1.82, 1.72, 1.81, 1.5, 1.95, 1.89, 1.62, 1.75, 1.82, 1.87, 1.57, 1.26, 1.28, 1.3, 1.37, 1.44, 1.38, 1.32, 1.44, 1.39, 1.48, 1.88, 1.31, 2.21, 2.11, 1.4, 1.58, 1.38, 1.38, 1.39, 2.16, 2.35, 1.72, 1.86, 2.31, 2.43, 2.22, 2.37, 2.34, 2.39, 1.37, 1.38, 1.39, 1.37, 1.34, 1.38, 1.33, 1.35, 1.66, 2.37, 1.72, 1.29, 1.34, 1.38, 1.84, 2.23, 1.85, 1.88, 1.87, 1.81, 1.91, 2.46, 2.42, 1.73, 2.38, 1.78, 1.52, 1.53, 1.48, 1.84, 1.89, 1.81, 1.46, 1.57, 1.49, 1.83, 1.8, 1.82, 1.9, 2.45, 1.79, 1.97, 1.72, 1.86, 2.07, 2.07, 1.93, 1.84, 2.44, 1.66, 1.32, 1.66, 1.89, 1.82, 1.8, 1.94, 1.46, 1.53, 1.84, 1.83, 1.82, 1.87, 1.84, 1.89, 2.13, 2.45, 2.44, 1.51, 1.8, 2.06, 1.97, 1.4, 1.31, 1.83, 1.48, 1.97, 2.17, 2.23, 2.16, 2.08, 1.92, 1.63, 2.39, 1.85, 1.57, 1.58, 2.1, 2.37, 2.27, 1.44, 1.46, 1.33, 1.44, 1.82, 1.77, 2.07, 1.9, 1.85, 1.82, 1.83, 1.69, 1.41, 1.29, 1.36, 1.85, 1.55, 2.0, 1.77, 1.56, 2.37, 1.79, 1.79, 1.9, 1.61, 2.01, 1.86, 1.64, 1.3, 1.46, 1.93, 1.46, 1.61, 1.36, 2.17, 2.36, 2.44, 2.26, 2.43, 2.53, 2.3, 2.3, 2.46, 2.41, 2.54, 2.35, 2.04, 2.04, 2.17, 1.83, 1.46, 1.36, 1.4, 1.38, 1.44, 1.89, 1.32, 1.76, 2.05, 2.0, 2.03, 1.34, 1.31, 1.33, 1.28]
    tiempos_link_fijo = [0.65, 0.51, 0.44, 0.61, 0.95, 1.05, 1.26, 1.38, 1.52, 1.59, 1.82, 1.47, 1.55, 1.63, 1.82, 2.22, 2.17, 3.8, 3.59, 3.01, 4.2, 3.34, 3.49, 3.85, 5.29, 5.04]
    #Guardo la información para un box plot
    df_pagina_fija = pd.DataFrame({
        'Método': [f'Páginas fijas\nCantidad de páginas: {cantPaginas}\nRango de links: 0-{max_cantLinks}'] * len(tiempos_pagina_fija),
        'Cantidad de Links': list(range(0, max_cantLinks+1, 1)),
        'Tiempo de ejecución (ms)': tiempos_pagina_fija
    })

    df_link_fijo = pd.DataFrame({
        'Método': [f'Links fijos\nCantidad de links: {cantLinks}\nRango de páginas: {min_cantPags}-{max_cantPags}'] * len(tiempos_link_fijo),
        'Cantidad de Páginas': list(range(min_cantPags, max_cantPags+1)),
        'Tiempo de ejecución (ms)': tiempos_link_fijo
    })

    df = pd.concat([df_pagina_fija, df_link_fijo])

    #Creo boxplot con Seaborn
    plt.figure(figsize=(8, 6))

    sns.boxplot(x='Método', y='Tiempo de ejecución (ms)', hue='Método', data=df, palette="Set2", legend=False)
    plt.title('Tiempo de ejecución variando cantidad de páginas y links')
    plt.xlabel('NAVEGANTE ALEATORIO', labelpad=30)
    plt.ylabel('Tiempo de ejecución (ms)')
    plt.ylim(0,8)
    plt.grid(True)
    plt.show()

def datos_graficoComparacion_Metodos():
    matrices, tests = leer_test()
    
    #-------------------Tests dados
    paginas = []
    links = []
    #NAVEGANTE ALEATORIO
    tiempos_tests1 = []
    #MÉTODO ESTANDAR
    tiempos_tests0 = []
    
    #Recopilo datos de tiempos para cada método    
    for i in range(len(matrices)):
        datos = calcularTiempoDeProcesamiento(1, matrices[i])
        tiempos_tests1.append(datos['tiempoProcesamiento'])
        datos = calcularTiempoDeProcesamiento(0, matrices[i])
        tiempos_tests0.append(datos['tiempoProcesamiento'])
        paginas.append(datos['paginas'])
        links.append(datos['links'])
    return tests, paginas, links, tiempos_tests0, tiempos_tests1
def graficoComparacion_Metodos():
    """
    Seteamos los datos obtenidos con la función datos_graficoComparación_Metodos() con el fin de obtener el mismo gráfico para poder analizarlo. 
    """
    tests = ['instagram_famosos_grafo', 'mathworld_grafo', 'test_15_segundos', 'test_30_segundos', 'test_aleatorio', 'test_dosestrellas']
    paginas = [30, 10, 2000, 3000, 5, 12]
    links = [178, 33, 12000, 18000, 12, 22]
    tiempos_tests0 = [5.06, 1.5, 27936.3, 78681.49, 0.62, 1.33]
    tiempos_tests1 = [4.24, 0.76, 28008.33, 75104.28, 0.68, 1.29]
    #Creo DataFrame
    df = pd.DataFrame({
        'Test': tests,
        'Cantidad de Páginas': paginas,
        'Cantidad de Links': links,
        'Tiempo Método Estándar': tiempos_tests0,
        'Tiempo Navegante Aleatorio': tiempos_tests1
    })
    #Creo gráfico de burbujas
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter1 = sns.scatterplot(data=df, x='Cantidad de Links', y='Cantidad de Páginas', size='Tiempo Método Estándar', sizes=(100, 2000), alpha=0.7, color='red', ax=ax)
    scatter2 = sns.scatterplot(data=df, x='Cantidad de Links', y='Cantidad de Páginas', size='Tiempo Navegante Aleatorio', sizes=(100, 2000), color='pink', alpha=0.5, ax=ax)
    
    #Creo subgráfico para el zoom
    ax_zoom = plt.axes([0.2, 0.55, .3, .3], facecolor='white')  # Ajustar posición del zoom
    scatter_zoom1 = sns.scatterplot(data=df, x='Cantidad de Links', y='Cantidad de Páginas', size='Tiempo Método Estándar', sizes=(300, 1500),color='red', alpha=0.7, ax=ax_zoom)
    scatter_zoom2 = sns.scatterplot(data=df, x='Cantidad de Links', y='Cantidad de Páginas', size='Tiempo Navegante Aleatorio', sizes=(300, 1500),color='pink', alpha=0.5, ax=   ax_zoom)
    ax_zoom.set_xlim([3, 190])
    ax_zoom.set_ylim([2, 35])
    
    # Eliminar etiquetas de ejes y leyenda del subgráfico de zoom
    ax_zoom.set_xlabel('')
    ax_zoom.set_ylabel('')
    ax_zoom.legend().set_visible(False)
    
    # Crear una leyenda personalizada con burbujas de tamaño específico
    legend_elements = [
        plt.scatter([], [], s=100, color='red', label='Método Estándar'),
        plt.scatter([], [], s=100, color='pink', label='Navegante Aleatorio')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize='large')
    ax.set_xlim([-500,19000])
    ax.set_ylim([-100,3500])
    
    # Centrar el título horizontalmente
    title = plt.title('Tiempo de ejecución en función de la cantidad de links y páginas', pad=40)
    title.set_position([.0, 1.04])
    plt.show()
    

def numeroDeCondicionSegunP():
    matrices, tests = leer_test()
    
    valores_p = np.linspace(0.1, 0.9, 9)
    
    numeros_de_condicion = []
    
    for W in matrices :
        cond = []
        for p in valores_p :            
            p, W, e, D, z_t = componentes_Pagerank(W, p)
            I = np.eye(W.shape[0])
            M = I-p*(W@D)
            cond.append(np.linalg.cond(M, 1))
        numeros_de_condicion.append(cond)

    plt.figure(figsize=(10, 6))

    plt.plot(valores_p, numeros_de_condicion[4], label='5 páginas', alpha=0.6, linestyle='-')
    plt.plot(valores_p, numeros_de_condicion[0], label='30 páginas', alpha=0.6, linestyle='-')
    plt.plot(valores_p, numeros_de_condicion[2], label='2000 páginas', alpha=0.6, linestyle='-')
    plt.plot(valores_p, numeros_de_condicion[3], label='3000 páginas', alpha=0.6, linestyle='-')
    
    plt.xlabel('valores de p')
    plt.ylabel('numero de condición de I-pWD')
    plt.title('Número de condición según p y tamaño de la matriz')
    plt.legend()
    plt.grid(True)
    plt.show()
