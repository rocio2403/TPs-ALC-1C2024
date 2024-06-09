# -*- coding: utf-8 -*-
"""
Trabajo Práctico 2:
Grupo : Susano y Valentin
Integrantes: Ibarra, Abril Magalí, Dominguez, Rocio Julieta
Materia: Algebra Lineal Computacional
Periodo. Primer Cuatrimestre 2024
Tema:  Contenido nutricional de la Canasta Básica
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans

# =============================================================================
# FUNCIONES
# =============================================================================
def normalizar_tabla_nutricional(tabla_nutricional):
    """
    Dado un dataframe, llena los valores nulos con 0 y convierte los valores en
    miligramos a gramos.
    """
    tabla_nutricional = tabla_nutricional.fillna(0)
    columnas_mg = [col for col in tabla_nutricional.columns if 'mg' in col]
    tabla_nutricional[columnas_mg] /= 1000
    tabla_nutricional.columns = tabla_nutricional.columns.str.replace('mg', 'gr', regex=True)
    return tabla_nutricional

def evaluarCumplimiento_dieta_margenes(data):
    """
    Evalúa si las cantidades de los principales elementos de la dieta, proteínas,
    carbohidratos, grasas, sodio, fibra, frutas y verduras, cumple los márgenes
    de ingesta de la OMS (2750kcal/Dia).
    
    Para la comparación se tiene en cuenta las siguientes relaciones:
        -Proteínas: 4kcal/g
        -Grasas: 9kcal/g
        -Carbohidratos: 4kcal/g
    """
    #Extraemos manualmente las frutas y verduras en la tabla nutricional
    frutas_verduras_lista = ['Acelga', 'Zanahoria', 'Tomate', 'Lechuga', 'Cebolla', 'Zapallo', 'Manzana', 'Naranja', 'Mandarina', 'pera', 'Banana', 'Papa', 'Batata']
    
    #Calculamos las cantidades totales
    proteinas = data['Proteinas (gr)'].sum()
    carbohidratos = data['HC (gr)'].sum()
    grasas = data['Grasas (gr)'].sum()
    sodio = data['Na (gr)'].sum()
    fibra = data['Fibra (gr)'].sum()
    frutas_verduras = data[data['Alimento'].isin(frutas_verduras_lista)]['Cantidad (gr/ml)'].sum()

    #Comparamos las cantidades totales con los márgenes
    proteinas_cumple = 10 <= ((proteinas*4)/2750)*100 <= 15
    carbohidratos_cumple = 55 <= ((carbohidratos*4)/2750)*100 <=75
    grasas_cumple = 15 <= ((grasas*9)/2750)*100 <= 30
    sodio_cumple = sodio <= 2
    fibra_cumple = fibra > 25
    frutas_verduras_cumple = frutas_verduras >= 400
    
    return proteinas_cumple and carbohidratos_cumple and grasas_cumple and sodio_cumple and fibra_cumple and frutas_verduras_cumple

    
    

# =============================================================================
# SCRIPT (poner en celdas)
# =============================================================================

carpeta = 'C:/Users/Rocio/Desktop/TPs-ALC-1C2024/TP2-ALC/'

#Creamos dataframes con los datos en formato csv
tabla_nutricional = pd.read_csv(carpeta +  'tabla_nutricional.csv',delimiter= ';')
consumidores_libres = pd.read_csv(carpeta + 'consumidores_libres.csv' ,delimiter = ';')

#Normalizamos la tabla
tabla_nutricional = normalizar_tabla_nutricional(tabla_nutricional)


#%%

#vamos a hacer un Analisis de primeras comoponentes

"""
consigna 3
Consigna 3.- Realizar un An´alisis en Componentes Principales (ACP), tomando
el valor nutricional de cada alimento por gramo. Graficar cada alimento como un
punto en ejes coordenados del nuevo espacio creado por los autovectores generadores del ACP.
 En este gr´afico, analizar los clusteres que quedan planteados por
la proyecci´on.
"""

#primero preparo la matriz sin el nombre de los alimentos

tabla_nutricional_sin_alimento = tabla_nutricional.drop(columns=['Alimento'])

# luego lo convierto en una matriz, donde cada fila son los valores nutricionales de cada alimento
matriz_nutricional = tabla_nutricional_sin_alimento.values

def matriz_covarianza(m):
    n = m.shape[1]  
    mCov = np.dot(m,m.T)/n #dividmos por n para normalizar
    return mCov


def centrar_matriz(m):
    n = m.shape[1] 
    media=np.mean(m, axis=1)  #Calcula el vector de medias m de las filas de la matriz

    matriz_centrada = m - np.tile(media.reshape((len(media), 1)), (1, n))
    
    return matriz_centrada,media

def calculoPCA(data):
    d,n = data.shape
    data, media = centrar_matriz(data)
    cov = matriz_covarianza(data)
    avals, avecs = np.linalg.eigh(cov)
    # ordenamos los autovalores de mayor a menor
    idx = np.argsort (- avals )
    #Ordena los autovalores en orden descendente y devuelve los índices de orden correspondientes.
    avals = avals[idx]
    #Ordena los autovalores y los autovectores de acuerdo con los índices obtenidos en el paso anterior.a
    avecs = avecs[:, idx]
    return avals,avecs,data,media
    

#ahora hacemos el grafico de pca
#para ello primero protectamos los datos


def proyectar_datos(x,w):
    """
    dao un conjunto de datos x y una amtriz de proyeccion w, devuelve los datos transformados
    como una amtriz
    matriz de proyeccion va a tener como columnas los autovectores de la amtriz de covarianza
    
    """
    return x.dot(w)

def pca(data):
    avals, avecs, data, media = calculoPCA(data)
    
    # Armamos una matriz con los avecs para proyectar
    m_proyeccion = np.array(avecs).T
  
    # Ahora proyectamos los datos
    data_pca = proyectar_datos(data.T, m_proyeccion)
    
    
    # Determinar el número de clústeres
    kmeans = KMeans(n_clusters=3,n_init=10)
    clusters = kmeans.fit_predict(data_pca[:, :2])

    # Graficar con los clústeres
  #  plt.figure(figsize=(10, 7))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis',alpha=0.5)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Analisis de Componentes Principales de la Canasta Basica')
    plt.show()
    
    # # Graficar los alimentos en el espacio de los componentes principales
    # plt.scatter(data_pca[:, 0], data_pca[:, 1])
    # plt.xlabel('Componente Principal 1')
    # plt.ylabel('Componente Principal 2')
    # plt.title('Análisis en Componentes Principales de la Canasta Básica')
    # plt.legend()
    # plt.show()

pca(matriz_nutricional)




