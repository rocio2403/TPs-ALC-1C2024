# -*- coding: utf-8 -*-
"""
Trabajo Práctico 1:
Grupo : Susano y Valentin
Integrantes: Ibarra, Abril Magalí, Dominguez, Rocio Julieta
Materia: Algebra Lineal Computacional
Periodo. Primer Cuatrimestre 2024
"""

from funciones import *
import matplotlib.pyplot as plt

#%%
##########################################################################
#
#                  ANÁLISIS CUALITATIVO                                  #
#
##########################################################################

# Obtenemos la amtriz para test

carpeta = './tests/'

famosos = leer_archivo(carpeta + 'instagram_famosos_grafo.txt')
mathworld = leer_archivo(carpeta + 'mathworld_grafo.txt')
segundos_15 = leer_archivo(carpeta + 'test_15_segundos.txt')
segundos_30 = leer_archivo(carpeta + 'test_30_segundos.txt')
aleatorio = leer_archivo(carpeta + 'test_aleatorio.txt')
dos_estrellas = leer_archivo(carpeta + 'test_dosestrellas.txt')

#armamos un solo grafico con todos los grafos, para eso creamos subplots

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Dibujamos  cada grafo en un subplot diferente
dibujarGrafo(famosos, titulo='Grafo Test Instagram_famosos', print_ejes=False, ax=axs[0, 0])
dibujarGrafo(mathworld, titulo='Grafo Test Mathworld', print_ejes=False, ax=axs[0, 1])
dibujarGrafo(segundos_15, titulo='Grafo Test 15 segundos', print_ejes=False, ax=axs[0, 2])
dibujarGrafo(segundos_30, titulo='Grafo Test 30 segundos', print_ejes=False, ax=axs[1, 0])
dibujarGrafo(aleatorio, titulo='Grafo Test Aleatorio', print_ejes=False, ax=axs[1, 1])
dibujarGrafo(dos_estrellas, titulo='Grafo Test Dos Estrellas', print_ejes=False, ax=axs[1, 2])

plt.tight_layout()
 
plt.show()

#%%

"""
Ahora analizamos como varía la pagina mejor rankeada de acuerdo a p
"""
#Para eso utilizamos la funcion calcular ranking del modulo funciones

p_valores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#vemos en que ranking está 1
def obtenerPosicionEnRanking(w,p,pagina):
    rnk,scr = calcularRanking(w, p)
    posicion = rnk[pagina-1]
    return posicion

def obtenerPaginaEnPosicion(w,p,posicion):
    rnk,scr = calcularRanking(w, p)
    pagina=0
    for i in range(len(rnk)):
        if (rnk[i]==posicion):
            pagina = i + 1
    return pagina

def obtenerIndiceMejorPagina(w,p):
    mejor_pagina = obtenerMejorPagina(w, p) #me da 6
    indice=obtenerPaginaEnPosicion(w, p, 1) #me da
    return indice-1


def ranking_pagina_vs_p(w, leyenda, color, ax):
   
    mejores_paginas = []
    for p in p_valores:
        rnk, scr = calcularRanking(w, p)
        mejor_pagina = obtenerMejorPagina(w, p)
        mejores_paginas.append(mejor_pagina)
    ax.plot(p_valores, mejores_paginas, label=leyenda, color=color)
   
    ax.set_xlabel('Valor de p')
    ax.set_ylabel('Número de página con mejor ranking')
    ax.legend(loc='upper right')


#Realizamos el grafico que combina los demas graficos
fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 


ranking_pagina_vs_p(famosos, leyenda='Test Famosos', color='red', ax=axs[0, 0])
ranking_pagina_vs_p(mathworld, leyenda='Test Mathworld', color='blue', ax=axs[0, 1])
ranking_pagina_vs_p(segundos_15, leyenda='Test 15 Segundos', color='green', ax=axs[0, 2])
ranking_pagina_vs_p(segundos_30, leyenda='Test 30 Segundos ', color='orange', ax=axs[1, 0])
ranking_pagina_vs_p(aleatorio, leyenda='Test Aleatorio', color='purple', ax=axs[1, 1])
ranking_pagina_vs_p(dos_estrellas, leyenda='Test Dos Estrellas', color='brown', ax=axs[1, 2])


plt.tight_layout()  
plt.show()

#%%
#analicemos el test 30 segundos, el cual presenta variaciones

print('la mejor pagina en el Test 30 segundos con p = 0.6 es la pagina: ',
      obtenerMejorPagina(segundos_30, 0.6))

#veamos con cuantos nodos está conectada esta pagina
def cant_link_asociados(matriz,pagina):
    links_asociados = np.sum(matriz[:, pagina-1])
    return links_asociados

pagina = obtenerIndiceMejorPagina(segundos_30, 0.6)
print('Cantindad de links asociados a la pagina {pagina} : ', 
      cant_link_asociados(segundos_30, pagina))


print('*'*50)
print('Ahora vemos que sucede cuando aumentamos p')

print('la mejor pagina en el Test 30 segundos con p = 0.7 es la pagina: ',
      obtenerMejorPagina(segundos_30, 0.7))


pagina_07 = obtenerIndiceMejorPagina(segundos_30, 0.7)
print('Cantindad de links asociados a la pagina {pagina_07} : ', 
      cant_link_asociados(segundos_30, pagina_07))


#%%

#Ahora analizamos como varía el puntaje de la mejor pagina de acuerdo a p y graficamos 
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
#Exploramos las diferentes densidades de los grafos
def calcular_densidad(W):
    n = W.shape[0]  # Número de nodos en el grafo
    aristas_presentes = np.sum(W)  # Número de aristas presentes en la matriz de conectividad w
    aristas_posibles = n * (n - 1)  # Número de aristas posibles en un grafo completo

    densidad = aristas_presentes / aristas_posibles
    return densidad

# Calcular la densidad de cada grafo
densidad_famosos = calcular_densidad(famosos)
densidad_mathworld = calcular_densidad(mathworld)
densidad_segundos_15 = calcular_densidad(segundos_15)
densidad_segundos_30 = calcular_densidad(segundos_30)
densidad_aleatorio = calcular_densidad(aleatorio)
densidad_dos_estrellas = calcular_densidad(dos_estrellas)

# Imprimir las densidades
print("Densidad del grafo de famosos:", densidad_famosos)
print("Densidad del grafo de mathworld:", densidad_mathworld)
print("Densidad del grafo de segundos_15:", densidad_segundos_15)
print("Densidad del grafo de segundos_30:", densidad_segundos_30)
print("Densidad del grafo de aleatorio:", densidad_aleatorio)
print("Densidad del grafo de dos_estrellas:", densidad_dos_estrellas)

"""
Un grafo con densidad 0 significa que no hay conexiones entre los nodos,
 mientras que un grafo con densidad 1 significa que todos los nodos están
 conectados entre sí.
"""
