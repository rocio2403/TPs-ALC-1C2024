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
from sklearn.cluster import KMeans
import seaborn as sns

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


def componentes_principalesSVD(X, k):
    """
    Dada una matriz X, computa su factorización SVD y retorna la proyección
    sobre el subespacio generado por los primeros k vectores singulares izquierdos
    de X y la martriz truncada de la SVD Vt_k.
    """
    # Centramos los vectores columna (características) según sus promedios
    n, d = X.shape
    m=np.mean(X, axis=1)

    X = X - np.tile(m.reshape((len(m), 1)), (1, d))

    # Obtenemos los autovalores y autovectores de la matriz de covarianza 
    MCov = np.dot(X.T,X)/X.shape[0] 
    avals, avecs = np.linalg.eigh(MCov)
    # Ordenamos los autovalores y autovectores
    indices_ordenados = np.argsort(-avals)
    avals = avals[indices_ordenados]
    avecs = avecs[:, indices_ordenados]
    # Seleccionamos los primeros k vectores singulares derechos de X
    Vt_k = avecs[:, :k]    
    # Matriz de reflexión en el eje de la primer componente
    R1 = np.array([[1, 0], [0, -1]])
    # Proyectamos los datos en el subespacio generado por los primeros k vectores singulares izquierdos
    X_proyectado = X @ Vt_k @ R1 
    return X_proyectado, Vt_k


def graficarProyeccion(X_proyectado, alimentos, titulo, n_clusters=4):
    """
    Grafica los alimentos como puntos en el subespacio generado por las 2
    componentes principales, coloreando los puntos según sus clusters.
    """
    # Aplicamos KMeans para encontrar clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_proyectado)
    labels = kmeans.labels_
    
    # Crear DataFrame con el número de cluster y alimento 
    df_clusters = pd.DataFrame({
        'Alimento': alimentos,
        'Número de Cluster': labels,
       
    })
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_proyectado[:, 0], X_proyectado[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title(titulo)
    plt.colorbar(scatter)
    plt.grid(True)
    plt.show()
    
    return df_clusters

def graficos_cluster():
            
    # Para el grupo 1
    grupo1 = df_clusters[df_clusters['Número de Cluster'] == 0]['Alimento'].tolist()
    grupo1 = tabla_nutricional[tabla_nutricional['Alimento'].isin(grupo1)]
    grupo1 = grupo1.merge(df_clusters, on='Alimento', how='left')
        
    # Para el grupo 2
    grupo2 = df_clusters[df_clusters['Número de Cluster'] == 1]['Alimento'].tolist()
    grupo2 = tabla_nutricional[tabla_nutricional['Alimento'].isin(grupo2)]
    grupo2 = grupo2.merge(df_clusters, on='Alimento', how='left')
        
    # Para el grupo 3
    grupo3 = df_clusters[df_clusters['Número de Cluster'] == 2]['Alimento'].tolist()
    grupo3 = tabla_nutricional[tabla_nutricional['Alimento'].isin(grupo3)]
    grupo3 = grupo3.merge(df_clusters, on='Alimento', how='left')
        
    # Para el grupo 4
    grupo4 = df_clusters[df_clusters['Número de Cluster'] == 3]['Alimento'].tolist()
    grupo4 = tabla_nutricional[tabla_nutricional['Alimento'].isin(grupo4)]
    grupo4 = grupo4.merge(df_clusters, on='Alimento', how='left')
        
        
    # Lista de macronutrientes
    macronutrientes = ['HC (gr)', 'Proteinas (gr)', 'Azucares Libres (gr)', 'Grasas (gr)', 'Fibra (gr)']
        
    df_combined = pd.concat([grupo1, grupo2, grupo3, grupo4])
        
    promedios_grupos = df_combined.groupby('Número de Cluster')[macronutrientes].mean().reset_index()
        
    sns.set(style="darkgrid")
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
        
    # Aplanar los ejes para iterar fácilmente
    axes = axes.flatten()
        
    for idx, macronutriente in enumerate(macronutrientes):
        sns.barplot(ax=axes[idx], x='Número de Cluster', y=macronutriente, data=promedios_grupos, palette='viridis')
            
        axes[idx].set_title(f'Promedio de {macronutriente} por Cluster')
        axes[idx].set_xlabel('Número de Cluster')
        axes[idx].set_ylabel(macronutriente)
        
    # Eliminar el último subgráfico si hay menos de 6
    if len(macronutrientes) < len(axes):
        fig.delaxes(axes[-1])
        
    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()
        
        
    plt.show()

def palabras_contenidas(nombre1, nombre2):
    """
    Verifica si todas las palabras de una cadena (excepto casos excepcionales) están contenidas en otra cadena
    o viceversa, retorna true si se cumple cualquiera de las dos condiciones,
    de lo contrario, retorna false.
    
    """
    casos_excepcionales =['dulce de leche', 'harina maiz', 'leche entera en polvo', 'paleta cocida', 'tomate envasado', 'vina re']
    if nombre1 not in casos_excepcionales:
        palabras1 = nombre1.split()
        for palabra in palabras1:
            if palabra != 'en' and palabra != 'de' and palabra in nombre2:
                return True
    return 

def filtrarAlimentos(Y, X):
    """
    Recibe 2 dataframes de información de alimentos, compara sus primeras columnas
    y devuelve los datos de X filtrados según sus alimentos que aparecen en Y. 
    """
    # Convertimos los nombres a minúsculas para una comparación insensible a mayúsculas/minúsculas
    X.iloc[:,0] = X.iloc[:,0].str.lower()
    Y.iloc[:,0] = Y.iloc[:,0].str.lower()
    
    # Guardamos los alimentos a filtrar 
    alimentos_Y = Y.iloc[:,0].tolist()

    #Filtramos
    filtro = X.iloc[:, 0].apply(lambda x: any(palabras_contenidas(x, alimento) for alimento in alimentos_Y))

    return X[filtro]

def proyectar(Y, proyector):
    """
    Proyecta la matriz Y en el subespacio generado por Vt_k obtenido de la matriz X.
    """
    # Centramos Y 
    n, d = Y.shape
    m=np.mean(Y, axis=1)
    Y = Y - np.tile(m.reshape((len(m), 1)), (1, d))
    # Matriz de reflexión en el eje de la primer componente
    R1 = np.array([[1, 0], [0, -1]])
    # Proyectamos en el subespacio de las primeras k componentes principales
    Y_proyectado = Y @ proyector @ R1
    return Y_proyectado
    
# =============================================================================
# SCRIPT (poner en celdas)
# =============================================================================

carpeta = ''

# Creamos dataframes con los datos en formato csv
tabla_nutricional = pd.read_csv(carpeta + 'tabla_nutricional.csv', delimiter=';')
consumidores_libres = pd.read_csv(carpeta + 'consumidores_libres.csv', delimiter=';')

# Normalizamos la tabla
tabla_nutricional = normalizar_tabla_nutricional(tabla_nutricional)
alimentos = tabla_nutricional['Alimento'].values 

# Reducimos la dimensionalidad de los datos (X) mediante PCA
matriz_nutricional = tabla_nutricional.iloc[:, 1:].values
X_proyectado, Vt_2= componentes_principalesSVD(matriz_nutricional, 2)

# Graficamos la proyección con los clusters y obtenemos el DataFrame de clusters
df_clusters = graficarProyeccion(X_proyectado, alimentos, 'Análisis en Componentes Principales de la Canasta Básica')
graficos_cluster()

# Proyectamos los alimentos de consumidores_libres (Y) en el subespacio del PCA anterior
Y = filtrarAlimentos(consumidores_libres, tabla_nutricional)
alimentos2 = Y['Alimento'].values
Y = Y.iloc[:,1:].values
proyector =  Vt_2
Y_proyectado = proyectar(Y, proyector)

# Graficamos la proyección con los clusters y obtenemos el DataFrame de clusters
df_clusters2 = graficarProyeccion(Y_proyectado, alimentos2, "Análisis de Componentes Principales de Alimentos de 'Consumidores Libres'")


