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
    if ax is None:
        fig, ax = plt.subplots()
    
    # Dibujar el grafo en el subplot proporcionado
    nx.draw(G, pos=nx.spring_layout(G), ax=ax, **options)
    
    # Agregar título si se proporciona
    if titulo:
        ax.set_title(titulo)
    
def factorizacion_LU(A):
    m, n = A.shape
    
    if m != n:
        print('Matriz no cuadrada, no es posible factorizar')
        return None, None
        
    L = np.eye(n)
    U = A.copy()
    
    for k in range(n-1):
        pivote = U[k, k]
        for f in range(k+1, n):
            coef = U[f, k] / pivote
            L[f, k] = coef
            U[f, k:] -= coef * U[k, k:]
            
    return L, U

def calcular_grado(matriz,i):
    return np.sum(matriz[:, i])

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

#rnk nos da el indice +1 es la pagina y el elmento la posicion donde está
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
    print(output)
    a=np.max(rnk)
    print(a)
    return output

#ARCHIVOS DE ENTRADA
archivo_test = './tests/test_dosestrellas.txt'
#CARGA DE ARCHIVO EN GRAFO
W = leer_archivo(archivo_test)
dibujarGrafo(W, print_ejes=False)

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

#%%
# =============================================================================
#                             ANÁLISIS CUANTITATIVO 
# =============================================================================




#%%
##########################################################################
#
#                  ANÁLISIS CUALITATIVO                                  #
#
##########################################################################

# obtengo graficos para cada test
carpeta = './tests/'
files = ['instagram_famosos_grafo.txt', 'mathworld_grafo.txt',
         'test_15_segundos.txt', 'test_30_segundos.txt', 
         'test_aleatorio.txt', 'test_dosestrellas.txt']

famosos = leer_archivo(carpeta + 'instagram_famosos_grafo.txt')
mathworld = leer_archivo(carpeta + 'mathworld_grafo.txt')
segundos_15 = leer_archivo(carpeta + 'test_15_segundos.txt')
segundos_30 = leer_archivo(carpeta + 'test_30_segundos.txt')
aleatorio = leer_archivo(carpeta + 'test_aleatorio.txt')
dos_estrellas = leer_archivo(carpeta + 'test_dosestrellas.txt')
#%%
# Crear subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Dibujar cada grafo en un subplot diferente
dibujarGrafo(famosos, titulo='Grafo Test Instagram_famosos', print_ejes=False, ax=axs[0, 0])
dibujarGrafo(mathworld, titulo='Grafo Test Mathworld', print_ejes=False, ax=axs[0, 1])
dibujarGrafo(segundos_15, titulo='Grafo Test 15 segundos', print_ejes=False, ax=axs[0, 2])
dibujarGrafo(segundos_30, titulo='Grafo Test 30 segundos', print_ejes=False, ax=axs[1, 0])
dibujarGrafo(aleatorio, titulo='Grafo Test Aleatorio', print_ejes=False, ax=axs[1, 1])
dibujarGrafo(dos_estrellas, titulo='Grafo Test Dos Estrellas', print_ejes=False, ax=axs[1, 2])


# Ajustar el espaciado entre subplots
plt.tight_layout()

# Mostrar la figura
plt.show()

#%%
#Ahora analizamos como varía la pagina mejor rankeada de acuerdo a p

#Para eso utilizamos la funcion calcular ranking
p_valores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#ahora vamos a calcular el mejor numero de pagina en funcion de p 

def ranking_pagina_vs_p(w, leyenda, color, ax):
   
    mejores_paginas = []
    for p in p_valores:
        rnk, scr = calcularRanking(w, p)
        mejor_pagina = max(rnk)
        mejores_paginas.append(mejor_pagina)
    ax.plot(p_valores, mejores_paginas, label=leyenda, color=color)
   
    ax.set_xlabel('Valor de p')
    ax.set_ylabel('Número de página con mejor ranking')
    ax.legend(loc='upper right')

# Lista de pruebas
tests = [famosos, mathworld, segundos_15, segundos_30, aleatorio, dos_estrellas]

fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 

# Graficar cada prueba en un subgráfico específico
ranking_pagina_vs_p(famosos, leyenda='Test Famosos', color='red', ax=axs[0, 0])
ranking_pagina_vs_p(mathworld, leyenda='Test Mathworld', color='blue', ax=axs[0, 1])
ranking_pagina_vs_p(segundos_15, leyenda='Test 15 Segundos', color='green', ax=axs[0, 2])
ranking_pagina_vs_p(segundos_30, leyenda='Test 30 Segundos ', color='orange', ax=axs[1, 0])
ranking_pagina_vs_p(aleatorio, leyenda='Test Aleatorio', color='purple', ax=axs[1, 1])
ranking_pagina_vs_p(dos_estrellas, leyenda='Test Dos Estrellas', color='brown', ax=axs[1, 2])


plt.tight_layout()  # Ajustar el espacio entre los subgráficos
plt.show()


#%%
#Ahora analizamos como varía el puntaje de la mejor pagina de acuerdo a p

p_valores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def ranking_score_vs_p(w, leyenda, color, ax):
   
    scores = []
    for p in p_valores:
        mejor_score = obtenerMaximoRankingScore(w, p)
        scores.append(mejor_score)
    ax.plot(p_valores, scores, label=leyenda,marker='o', color=color)
   
    ax.set_xlabel('Valor de p')
    ax.set_ylabel('Puntaje de la página con mejor ranking')
    ax.legend(loc='upper right')

fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 

 
ranking_score_vs_p(famosos, leyenda='Test Famosos', color='red', ax=axs[0, 0])
ranking_score_vs_p(mathworld, leyenda='Test Mathworld', color='blue', ax=axs[0, 1])
ranking_score_vs_p(segundos_15, leyenda='Test 15 Segundos', color='green', ax=axs[0, 2])
ranking_score_vs_p(segundos_30, leyenda='Test 30 Segundos ', color='orange', ax=axs[1, 0])
ranking_score_vs_p(aleatorio, leyenda='Test Aleatorio', color='purple', ax=axs[1, 1])
ranking_score_vs_p(dos_estrellas, leyenda='Test Dos Estrellas', color='brown', ax=axs[1, 2])

plt.tight_layout()  
plt.show()


#%%

##############################################################################
#
#                   CASO TEST DOS ESTRELLAS
#
##############################################################################   

"""
¿Cual es la minima cantidad de links que se deben agregar para que la pagina 1 
quede primera en el ranking?

¿Como se modificó la amtriz de concetividad?
"""    

###################################################################
#
# LUEGO DE SABER COMO FUNCIONABA RNK
#Los indices son las paginas y los elementos son los rankings
##################################################################



#Para poder responder estas preguntas, veamos cuantos links asociados tiene
#la pagina 1 y cuantos tiene la pagina que queda primera en el ranking
p = 0.5
# veamos cual es la pagina que queda primero
mejor_pagina = obtenerMejorPagina(dos_estrellas, p)
# observamos e grafo de dos_estrellas
dibujarGrafo(dos_estrellas,titulo = 'Grafo Dos Estrellas',print_ejes=False)
print('La primera pagina que queda en el ranking es la pagina ', mejor_pagina)

#vemos en que ranking está 1
def obtenerPosicionEnRanking(w,p,pagina):
    rnk,scr = calcularRanking(w, p)
    posicion = rnk[pagina-1]
    return posicion
    
#ahora vemos la posicion de la pagina 1
posicion_pag_1 = obtenerPosicionEnRanking(dos_estrellas, p, 1)
print('La posicion en ranking de la pagina 1 es: ', posicion_pag_1)

#sabemos que la primera en el ranking es 6 y la tercera es 1, buscamos la segunda

def obtenerPaginaEnPosicion(w,p,posicion):
    rnk,scr = calcularRanking(w, p)
    pagina=0
    for i in range(len(rnk)):
        if (rnk[i]==posicion):
            pagina = i + 1
    return pagina

segunda_ranking =obtenerPaginaEnPosicion(dos_estrellas, p, 2)
print('pagina en posicion 2:', segunda_ranking)

#probamos conectando un nodo a la pagina 7
def modificar_links_Asociados(matriz,pagina,links):
    nueva_matriz = matriz.copy()
    for link in links:
        nueva_matriz[link-1][pagina-1] = 1
    return nueva_matriz

modificada = modificar_links_Asociados(dos_estrellas, 1, [7])

modificada = modificar_links_Asociados(dos_estrellas, 7, [1])
dibujarGrafo(modificada)
print('mod',obtenerMejorPagina(modificada, p))
#ahora vamos a crear un enlace de 6 a 1

#ahora vamos a enlazar al azar

def agregar_links_minimos(matriz,pagina,p):
    M = matriz.copy()
    primer_puesto = obtenerMejorPagina(M, p)
    puesto_pagina = obtenerPosicionEnRanking(M, p, pagina)
    cant_links = 0
    while primer_puesto != puesto_pagina:
        pagina_no_conectada = None
        for i in range(len(M)):
            #no se tienen en cuenta autolinks
            if i != pagina - 1 and M[i, pagina - 1] == 0:
                pagina_no_conectada = i
                M[pagina_no_conectada, pagina_objetivo - 1] = 1
                cant_links +=1
        #Recalcular el ranking de Page
        pagina_actual = obtenerMejorPagina(M, p)
    return M
        
cantidad_minima_enlaces = agregar_links_minimos(dos_estrellas,1, p)
print('mejor',obtenerMejorPagina(cantidad_minima_enlaces, p))
#print("La cantidad mínima de enlaces que deben agregarse es:", cantidad_minima_enlaces)


#%%
#
#
#            TODO LO SIGUIENTE NO SIRVE
####
#
#
############################################################################

#%%
#Ahora veamos cuantos links asociados tiene la pagina
def cant_link_asociados(matriz,pagina):
    links_asociados = np.sum(matriz[:, pagina-1])
    return links_asociados

links_pagina_12 = cant_link_asociados(dos_estrellas, 12)
print('cantidad de links asociados a la pagina 12: ', links_pagina_12)

links_pagina_1 = cant_link_asociados(dos_estrellas, 1)
print('cantidad de links asociados a la pagina 1: ', links_pagina_1)
#vemos que la pagina 12 solo está concetada con un link, pero ese link debe ser de gran importancia, por lo que llocalizamos ese link y lo concetamos con 1

def links_asociados(matriz,pagina):
    links = []
    for i in range(len(matriz)):
        if (matriz[i][pagina-1] == 1):
            links.append(i + 1)
    return links
linsasociadosa12 = links_asociados(dos_estrellas, 12)
print(linsasociadosa12)

#ahora conectamos a uno con 7 y vemos como se modifica

def modificar_links_Asociados(matriz,pagina,links):
    nueva_matriz = matriz.copy()
    for link in links:
        nueva_matriz[link-1][pagina-1] = 1
    return nueva_matriz

def eliminar_link(matriz,pagina,links):
    nueva_matriz = matriz.copy()
    for link in links:
        nueva_matriz[link-1][pagina-1] = 0
    return nueva_matriz



def intercambiar_nodos(matriz,nodo1,nodo2):
    m = matriz.copy()
    links_nodo1 = links_asociados(matriz,nodo1)
    links_nodo2= links_asociados(matriz,nodo2)
    m = modificar_links_Asociados(m,nodo1,links_nodo2)
    m = modificar_links_Asociados(m,nodo2,links_nodo1)
    m = eliminar_link(m,nodo1,links_nodo1)
    for link in links_nodo1:
        m=eliminar_link(m,link,[nodo1])
        m = modificar_links_Asociados(m,link,[nodo2])
        
    for link in links_nodo2:
        m=eliminar_link(m,link,[nodo2])
        
        m = modificar_links_Asociados(m,link,[nodo1])
        
    m = eliminar_link(m,nodo2,links_nodo2)
    return m


ultimo = intercambiar_nodos(dos_estrellas,1,12)
print(links_asociados(ultimo,1))
print(links_asociados(ultimo,12))
print('score',obtenerMaximoRankingScore(ultimo,0.5))

print('best',obtenerMejorPagina(ultimo,0.5))
    
dibujarGrafo(ultimo,print_ejes=False)

#funcion que agrega enlaces:
def agregar_links_minimos(matriz,pagina):
    mejor_pagina = obtenerMejorPagina(matriz,0.5)
    w =matriz.copy()
    while mejor_pagina != pagina:
        # Encontrar una página no conectada a la página que queremos enlazar
        pagina_no_conectada = None
        for i in range(w.shape[0]):
            if i != pagina_objetivo - 1 and w[i, pagina_objetivo - 1] == 0:
                pagina_no_conectada = i
                break #terminamos el ciclo
        # Agregar un enlace desde la página no conectada hacia la página 
        if pagina_no_conectada is not None:
            w[pagina_no_conectada, pagina_objetivo - 1] = 1
        # else:
        #     # Si no se puede encontrar una página no conectada, terminamos
        #     break
        # Recalcular el ranking de Page
        pagina_actual = obtenerMejorPagina(w, 0.5)
    return w
matriz_nueva = agregar_links_minimos(dos_estrellas,1)
print(obtenerMejorPagina(matriz_nueva,0.5))
dibujarGrafo(matriz_nueva,titulo = 'links minimos',print_ejes=False)



#%%
def encontrar_minima_cantidad_enlaces(matriz, pagina_lider, p):
    pagina_objetivo = 1
    enlaces_agregados = 0
    
    while True:
        # Identificar las páginas conectadas con la página líder actual
        li = links_asociados(matriz, pagina_lider)
        
        # Conectar una de estas páginas directamente con la página objetivo
        matriz_modificada = modificar_links_Asociados(matriz, pagina_objetivo, li)
        
        # Recalcular el ranking de Page
        nuevo_ranking = obtenerMejorPagina(matriz_modificada, p)
        
        # Si la página objetivo ahora ocupa el primer lugar en el ranking, terminar
        if nuevo_ranking == pagina_objetivo:
            break
        
        # Si no, aumentar el contador de enlaces agregados y continuar
        enlaces_agregados += 1
    
    return enlaces_agregados

# Llamar a la función para encontrar la cantidad mínima de enlaces necesarios
cantidad_minima_enlaces = encontrar_minima_cantidad_enlaces(dos_estrellas, 12, 0.5)

print("La cantidad mínima de enlaces que se deben agregar para que la página 1 quede primera en el ranking es:", cantidad_minima_enlaces)


        
#%%
def agregar_enlaces_para_mejorar_ranking(W, p, pagina_objetivo):
    W_modificada = W.copy()
    pagina_actual = obtenerMejorPagina(W_modificada, p)
    while pagina_actual != pagina_objetivo:
        # Encontrar una página no conectada a la página objetivo
        pagina_no_conectada = None
        for i in range(W_modificada.shape[0]):
            if i != pagina_objetivo - 1 and W_modificada[i, pagina_objetivo - 1] == 0:
                pagina_no_conectada = i
                break
        # Agregar un enlace desde la página no conectada hacia la página objetivo
        if pagina_no_conectada is not None:
            W_modificada[pagina_no_conectada, pagina_objetivo - 1] = 1
        else:
            # Si no se puede encontrar una página no conectada, terminamos
            break
        # Recalcular el ranking de Page
        pagina_actual = obtenerMejorPagina(W_modificada, p)
    return W_modificada

# Cargar la matriz de conectividad del archivo de prueba
archivo_test = './tests/test_dosestrellas.txt'
W = leer_archivo(archivo_test)

# Definir la probabilidad p
p = 0.5

# Definir la página objetivo (en este caso, la página 1)
pagina_objetivo = 1

# Agregar enlaces para mejorar el ranking de la página objetivo
W_modificada = agregar_enlaces_para_mejorar_ranking(W, p, pagina_objetivo)

# Calcular el nuevo ranking de Page
ranking_modificado, _ = calcularRanking(W_modificada, p)

# Mostrar el nuevo ranking de Page y la matriz de conectividad modificada
print("Nuevo ranking de Page:", ranking_modificado)
print("Matriz de conectividad modificada:")
print(W_modificada)

b =obtenerMejorPagina(W_modificada, 0.5)
print(b)
dibujarGrafo(W_modificada, print_ejes=False)


#%%
#de chat gpt

def agregar_enlaces_para_mejorar_ranking(W, p, pagina_objetivo):
    W_modificada = W.copy()
    pagina_actual = obtenerMejorPagina(W_modificada, p)
    while pagina_actual!= pagina_objetivo:
        # Encontrar una página no conectada a la página objetivo
        pagina_no_conectada = None
        for i in range(W_modificada.shape[0]):
            if i!= pagina_objetivo - 1 and W_modificada[i, pagina_objetivo - 1] == 0:
                pagina_no_conectada = i
                break
        # Agregar un enlace desde la página no conectada hacia la página objetivo
        if pagina_no_conectada is not None:
            W_modificada[pagina_no_conectada, pagina_objetivo - 1] = 1
        else:
            # Si no se puede encontrar una página no conectada, terminamos
            break
        # Recalcular el ranking de Page
        pagina_actual = obtenerMejorPagina(W_modificada, p)
    return W_modificada

# Cargar la matriz de conectividad del archivo de prueba
archivo_test = './tests/test_dosestrellas.txt'
W = leer_archivo(archivo_test)

# Definir la probabilidad p
p = 0.5

# Definir la página objetivo (en este caso, la página 1)
pagina_objetivo = 1

# Agregar enlaces para mejorar el ranking de la página objetivo
W_modificada = agregar_enlaces_para_mejorar_ranking(W, p, pagina_objetivo)

# Calcular el nuevo ranking de Page
ranking_modificado, _ = calcularRanking(W_modificada, p)

# Mostrar el nuevo ranking de Page y la matriz de conectividad modificada
print("Nuevo ranking de Page:", ranking_modificado)
print("Matriz de conectividad modificada:")
dibujarGrafo(W_modificada,0.5)
print(obtenerMejorPagina(W_modificada,0.5))
