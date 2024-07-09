# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:14:48 2024

@author: Rocio
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.colors as mcolors

from mpl_toolkits.mplot3d import Axes3D

#%%

"""
consigna 1
"""
def estandarizar_tabla_nutricional(tabla_nutricional):
    """
    Dado un dataframe, llena los valores nulos con 0 y convierte los valores en
    miligramos a gramos.
    """
    tabla_nutricional = tabla_nutricional.fillna(0)
    columnas_mg = [col for col in tabla_nutricional.columns if 'mg' in col]
    tabla_nutricional[columnas_mg] /= 1000
    tabla_nutricional.columns = tabla_nutricional.columns.str.replace('mg', 'gr', regex=True)
    return tabla_nutricional

#Creamos dataframes con los datos en formato csv
tabla_nutricional = pd.read_csv('tabla_nutricional.csv',delimiter= ';')
consumidores_libres = pd.read_csv('consumidores_libres.csv' ,delimiter = ';')

#Estandarizamos la tabla
tabla_nutricional = estandarizar_tabla_nutricional(tabla_nutricional)

tabla_nutricional.head(3)
#%%
consumidores_libres.head(3)

#%%

"""
Consigna 2
"""

def evaluarCumplimiento_dieta_margenes(data):
    """
    Evalúa si las cantidades de los principales elementos de la dieta, proteínas,
    carbohidratos, grasas, sodio, fibra, frutas y verduras, cumple los márgenes
    de ingesta de la OMS (2750kcal/Dia).

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
    data = {
       'Nutriente': ['proteinas', 'carbohidratos', 'grasas', 'sodio', 'fibra', 'frutas y verduras'],
       'Cumple': [proteinas_cumple, carbohidratos_cumple, grasas_cumple, sodio_cumple, fibra_cumple, frutas_verduras_cumple],
       'Valor': [(((proteinas*4)/2750)*100).round(2), (((carbohidratos*4)/2750)*100).round(2),
                 (((grasas*9)/2750)*100).round(2), sodio.round(2), fibra, frutas_verduras],
       'Unidad':['Calorias','Calorias','Calorias','gr','gr/dia','gr']
   }
                     
    df = pd.DataFrame(data)
    print(f'Grasas cumple : {grasas_cumple}, Valor = { (((grasas*9)/2750)*100).round(2) } calorias')
    print(f'HC cumple : {carbohidratos_cumple}, Valor = {(((carbohidratos*4)/2750)*100).round(2)} calorias')
    print(f'Proteinas cumple : {proteinas_cumple}, Valor = { (((proteinas*4)/2750)*100).round(2) } calorias')
    print(f'Sodio cumple : {sodio_cumple}, Valor = {sodio.round(2)}gr')
    print(f'Fibra cumple : {fibra_cumple}, Valor ={fibra} gr/dia')
    print(f'Frutas y verduras cumplen : {frutas_verduras_cumple}, Valor = {frutas_verduras}')
    
    cumple = proteinas_cumple and carbohidratos_cumple and grasas_cumple and sodio_cumple and fibra_cumple and frutas_verduras_cumple

    return 'Cumple los márgenes' if cumple else 'No cumple los márgenes', df
#Verificamos si la CBA cumple los márgenes de la OMS
print(evaluarCumplimiento_dieta_margenes(tabla_nutricional))
df = evaluarCumplimiento_dieta_margenes(tabla_nutricional)[1]
#REVISAR IMPRESION DE RESULTADOS

#%%
"""
Consigna 3
"""

# ########################
# #MODIFICAR A GRAFICO 3D
# #########################
# def componentes_principalesSVD(X, k):
#     """
#     Dada una matriz X, computa su factorización SVD y retorna la proyección
#     sobre el subespacio generado por los primeros k vectores singulares izquierdos
#     de X y la martriz truncada de la SVD Vt_k.
#     """
#     # Centramos los vectores columna (características) según sus promedios
#     n, d = X.shape
#     m=np.mean(X, axis=1)

#     X = X - np.tile(m.reshape((len(m), 1)), (1, d))

#     # Obtenemos los autovalores y autovectores de la matriz de covarianza
#     MCov = np.dot(X.T,X)/X.shape[0]
#     avals, avecs = np.linalg.eigh(MCov)
#     # Ordenamos los autovalores y autovectores
#     indices_ordenados = np.argsort(-avals)
#     avals = avals[indices_ordenados]
#     avecs = avecs[:, indices_ordenados]
#     # Seleccionamos los primeros k vectores singulares derechos de X
#     Vt_k = avecs[:, :k]
#     # Matriz de reflexión en el eje de la primer componente
#     R1 = np.array([[1, 0], [0, -1]])
#     # Proyectamos los datos en el subespacio generado por los primeros k vectores singulares izquierdos
#     X_proyectado = X @ Vt_k @ R1
#     return X_proyectado, Vt_k

# # Reducimos la dimensionalidad de los datos (X) mediante PCA
# matriz_nutricional = tabla_nutricional.iloc[:, 1:].values #convertimos la tabla nutricional a una matriz guardando los datos numericos

# X_proyectado, Vt_2= componentes_principalesSVD(matriz_nutricional, 2) #obtenemos la matriz nutricional proyectada


# def graficarProyeccion(X_proyectado, alimentos, titulo, n_clusters=4):
#     """
#     Grafica los alimentos como puntos en el subespacio generado por las 2
#     componentes principales, coloreando los puntos según sus clusters.
#     """
#     # Aplicamos KMeans para encontrar clusters
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_proyectado)
#     labels = kmeans.labels_

#     # Crear DataFrame con el número de cluster y alimento
#     df_clusters = pd.DataFrame({
#         'Alimento': alimentos,
#         'Número de Cluster': labels,

#     })

#     plt.figure(figsize=(12, 8))
#     scatter = plt.scatter(X_proyectado[:, 0], X_proyectado[:, 1], c=labels, cmap='viridis', alpha=0.5)
#     plt.xlabel('Componente Principal 1')
#     plt.ylabel('Componente Principal 2')
#     plt.title(titulo)
#     plt.colorbar(scatter)
#     plt.grid(True)
#     plt.show()

#     return df_clusters

# # Graficamos la proyección con los clusters y obtenemos el DataFrame de clusters
# alimentos = tabla_nutricional['Alimento'].values

# df_clusters = graficarProyeccion(X_proyectado, alimentos, 'Análisis en Componentes Principales de la Canasta Básica') #guardamos un dataframe coon la informacion de los clusters

# df_clusters.head(3)

#%%
#PRUEBA GRAFICOS 3D

from mpl_toolkits.mplot3d import Axes3D

def componentes_principalesSVD(X, k):
    """
    Dada una matriz X, computa su factorización SVD y retorna la proyección
    sobre el subespacio generado por los primeros k vectores singulares izquierdos
    de X y la martriz truncada de la SVD Vt_k.
    """
    # Centramos los vectores columna (características) según sus promedios
    n, d = X.shape
    m = np.mean(X, axis=0)
    X = X - m

    # Obtenemos los autovalores y autovectores de la matriz de covarianza
    MCov = np.dot(X.T, X) / X.shape[0]
    avals, avecs = np.linalg.eigh(MCov)
    # Ordenamos los autovalores y autovectores
    indices_ordenados = np.argsort(-avals)
    avals = avals[indices_ordenados]
    avecs = avecs[:, indices_ordenados]
    # Seleccionamos los primeros k vectores singulares derechos de X
    Vt_k = avecs[:, :k]
    # Proyectamos los datos en el subespacio generado por los primeros k vectores singulares izquierdos
    X_proyectado = X @ Vt_k
    
    # Calculamos la proporción de la varianza explicada
    var_explicada = avals[:k] / np.sum(avals)
    var_explicada_acumulada = np.cumsum(var_explicada)
    return X_proyectado, Vt_k, var_explicada, var_explicada_acumulada

# Reducimos la dimensionalidad de los datos (X) mediante PCA
matriz_nutricional = tabla_nutricional.iloc[:, 1:].values

X_proyectado, Vt_3, var_explicada, var_explicada_acumulada = componentes_principalesSVD(matriz_nutricional, 3) # Proyección en 3 componentes

# Imprimimos la proporción de la varianza explicada
print(f'Proporción de la varianza explicada por cada una de las 3 primeras componentes principales: {var_explicada * 100}')
print(f'Proporción de la varianza explicada acumulada: {var_explicada_acumulada * 100}')

def graficarProyeccion3D(X_proyectado, alimentos, titulo, n_clusters=4):
    """
    Grafica los alimentos como puntos en el subespacio generado por las 3
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

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_proyectado[:, 0], X_proyectado[:, 1], X_proyectado[:, 2], c=labels, cmap='viridis', alpha=0.5)
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_zlabel('Componente Principal 3')
    ax.set_title(titulo)
    fig.colorbar(scatter)
    plt.show()

    return df_clusters

# Graficamos la proyección con los clusters y obtenemos el DataFrame de clusters
alimentos = tabla_nutricional['Alimento'].values

df_clusters = graficarProyeccion3D(X_proyectado, alimentos, 'Análisis en Componentes Principales de la Canasta Básica')



#%%
def graficos_cluster():
    """
    Genera gráficos de barras para visualizar el promedio de macronutrientes
    (HC, proteínas, azúcares libres, grasas, y fibra) por cada grupo de
    alimentos agrupados en cuatro clusters.

    Parámetros:Ninguno.

    Retorna: Muestra gráficos y realiza impresiones en pantalla.
    """

    # Para el grupo 1
    grupo1 = df_clusters[df_clusters['Número de Cluster'] == 0]['Alimento'].tolist()
    grupo1 = tabla_nutricional[tabla_nutricional['Alimento'].isin(grupo1)]
    grupo1 = grupo1.merge(df_clusters, on='Alimento', how='left')
    cant1 = len(grupo1)

    # Para el grupo 2
    grupo2 = df_clusters[df_clusters['Número de Cluster'] == 1]['Alimento'].tolist()
    grupo2 = tabla_nutricional[tabla_nutricional['Alimento'].isin(grupo2)]
    grupo2 = grupo2.merge(df_clusters, on='Alimento', how='left')
    cant2 = len(grupo2)

    # Para el grupo 3
    grupo3 = df_clusters[df_clusters['Número de Cluster'] == 2]['Alimento'].tolist()
    grupo3 = tabla_nutricional[tabla_nutricional['Alimento'].isin(grupo3)]
    grupo3 = grupo3.merge(df_clusters, on='Alimento', how='left')
    cant3 = len(grupo3)


    # Para el grupo 4
    grupo4 = df_clusters[df_clusters['Número de Cluster'] == 3]['Alimento'].tolist()
    grupo4 = tabla_nutricional[tabla_nutricional['Alimento'].isin(grupo4)]
    grupo4 = grupo4.merge(df_clusters, on='Alimento', how='left')
    cant4 = len(grupo4)


    # Lista de macronutrientes
    macronutrientes = ['HC (gr)', 'Proteinas (gr)', 'Azucares Libres (gr)', 'Grasas (gr)', 'Fibra (gr)']

    df_combined = pd.concat([grupo1, grupo2, grupo3, grupo4])

    promedios_grupos = df_combined.groupby('Número de Cluster')[macronutrientes].mean().reset_index()
    promedios_grupos['Número de Cluster'] += 1

    sns.set(style="darkgrid")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)

    # Aplanar los ejes para iterar fácilmente
    axes = axes.flatten()


    for idx, macronutriente in enumerate(macronutrientes):
        sns.barplot(ax=axes[idx], x='Número de Cluster', hue= 'Número de Cluster', y=macronutriente, data=promedios_grupos, palette='viridis', legend=False)
        axes[idx].set_title(f'Promedio de {macronutriente} por Cluster')
        axes[idx].set_xlabel('Número de Cluster')
        axes[idx].set_ylabel(macronutriente)

    # Eliminar el último subgráfico si hay menos de 6
    if len(macronutrientes) < len(axes):
        fig.delaxes(axes[-1])

    # Ajustar el layout para evitar solapamientos
    plt.tight_layout()


    plt.show()
    print(f'cantidad de alimentos cluster 1: {cant1}')
    print(f'cantidad de alimentos cluster 2: {cant2}')
    print(f'cantidad de alimentos cluster 3: {cant3}')
    print(f'cantidad de alimentos cluster 4: {cant4}')
    return grupo1, grupo2,grupo3,grupo4
    
cluster_1,cluster_2,cluster_3,cluster_4 = graficos_cluster()
#%%
"""
consigna 4
"""
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
    R1 = np.array([[1, 0,0], [0, -1,0],[0,0,1]])
    # Proyectamos en el subespacio de las primeras k componentes principales
    Y_proyectado = Y @ proyector @ R1
    return Y_proyectado

# Proyectamos los alimentos de consumidores_libres (Y) en el subespacio del PCA anterior
nutricional_filtrada = filtrarAlimentos(consumidores_libres,tabla_nutricional)

nutricional_filtrada.shape


Y = nutricional_filtrada
alimentos2 = Y['Alimento'].values
Y = Y.iloc[:,1:].values
proyector =  Vt_3
Y_proyectado = proyectar(Y, proyector)
# Graficamos la proyección con los clusters y obtenemos el DataFrame de clusters
df_clusters_consumidores = graficarProyeccion3D(Y_proyectado, alimentos2, "Análisis de Componentes Principales de Alimentos de 'Consumidores Libres'") #guardamos el df para más adelante

#%%
######################
#
#   QUEDA IGUAL
######################
"""
consigna 5

"""
# Eliminamos la fila correspondiente a "berenjenas", ya que no tenemos coincidencia en la tabla nutricional
consumidores_libres = consumidores_libres[consumidores_libres['PRODUCTOS'] != 'berenjenas']

#listamos las fechas
meses = ['31/12/2023', '31/1/2024', '29/2/2024', '31/3/2024', '30/4/2024']

# Inicializamos diccionarios para almacenar los precios por gramos de cada nutriente
precios_hc = {}
precios_proteinas = {}
precios_grasas = {}
precios_carne ={}

#listamos casos excepcionales que quedan fuera al filtrar
casos_excepcionales ={
        'aceite girasol': 'aceite cocinero girasol',
    'fideos secos': 'fideos guiseros',
    'harina trigo': 'harina de trigo',
    'leche fluida entera': 'leche c/vit.fort',
    'pan frances': 'pan fresco',
       }

# Iteramos sobre cada alimento en la tabla nutricional
for _, row in nutricional_filtrada.iterrows():

    alimento = row['Alimento'] #Toma el alimento de tabla nutricional

    cant_ali = row['Cantidad (gr/ml)'] #Toma la cantidad de ese alimento para ver su composicion en nutrientes

    gr_hc = row['HC (gr)'] #toma su composición en gramos de carbohidratos

    gr_proteinas = row['Proteinas (gr)'] #toma su composición en gramos de proteinas

    gr_grasas = row['Grasas (gr)'] #toma su composición en gramos de grasas

    # Buscamos alimento en consumidores libres
    if alimento in casos_excepcionales.keys():
       alimento =  casos_excepcionales[alimento]

    match = consumidores_libres[consumidores_libres['PRODUCTOS'].str.contains(alimento)] # utilizamos una sentencia match para comparar patrones y extraer componentes

    if not match.empty: # evitamos operar con el producto 'yerba', ya que contiene 0 proteinas,grasas y hc

        cant_con_precio = match.iloc[0]['Cantidad'] #guardamos la cantidad  para la cual tenemos precio

        precios = match.iloc[0][meses].values #listamos los precios

        # Calculamos precios por gramo de nutriente
        if gr_hc > 0:
            #aplico regla de tres

            x = (gr_hc*cant_con_precio)/cant_ali

            #ahora veo para los gramos dados
            y = (precios*x)/cant_con_precio

            #ahora saco para un gramo
            precio_hc = y/x

            precios_hc[alimento] = precio_hc #guardamos el precio obtenido
        if gr_proteinas > 0:
            #aplico regla de tres

            x = (gr_proteinas*cant_con_precio)/cant_ali
            #ahora veo para los gramos dados

            y = (precios*x)/cant_con_precio
            #ahora saco para un gramo

            precio_proteinas = y/x

            precios_proteinas[alimento] = precio_proteinas
        if gr_grasas > 0:
            #aplico regla de tres

            x = (gr_grasas*cant_con_precio)/cant_ali
            #ahora veo para los gramos dados

            y = (precios*x)/cant_con_precio

            #ahora saco para un gramo

            precio_grasas = y/x

            precios_grasas[alimento] = precio_grasas

#convertimos en dataframes los diccioanrios obtenidos y trasponemos para una mejor organización y visuaizacion.
#las columnas seran los meses y los indices de las filas, los alimentos
df_hc = pd.DataFrame(precios_hc, index=meses).T
df_proteinas = pd.DataFrame(precios_proteinas, index=meses).T
df_grasas = pd.DataFrame(precios_grasas, index=meses).T

print(df_hc.head(3))

def minimosCuadrados_grafico(df, titulo):
    """
    Genera un gráfico de precios originales y su interpolación por mínimos cuadrados
    para diferentes alimentos a lo largo de cinco meses.

    Parámetros:
    df -- DataFrame donde las filas son alimentos y las columnas son los precios mensuales.
    titulo -- Título del gráfico.
    """
    plt.figure(figsize=(12, 8))
    mes_indices = np.array([0, 1, 2, 3, 4])  #tomamos 5 fechas de datos
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i, alimento in enumerate(df.index):
        precios = df.loc[alimento].values.astype(np.float64)  # Convertimos a float64 por problemas al compilar

        # Realizamos las ecuaciones normales
        A = np.vstack([mes_indices, np.ones(len(mes_indices))]).T
        ATA = A.T @ A
        ATy = A.T @ precios

        # Resolvemos el sistema de ecuaciones
        coef = np.linalg.solve(ATA, ATy)
        m, b = coef


        color = colors[i % len(colors)]

        # Graficamos precios originales
        plt.plot(mes_indices, precios, 'o', label=f'{alimento} (original)', color=color)

        # Graficamos la interpolacion por minimos cuadrados
        plt.plot(mes_indices, m * mes_indices + b, '-', label=f'{alimento} (Interpolacion)', color=color)

        plt.xlabel('Meses')
        plt.ylabel('Precio por gramo de nutriente')
        plt.title(titulo)
    plt.grid(True)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(mes_indices, ['Dic23', 'Ene24', 'Feb24', 'Mar24', 'Abr24'])
    plt.show()


minimosCuadrados_grafico(df_hc, 'Aplicación de Cuadrados mínimos para Precios por HC')
minimosCuadrados_grafico(df_proteinas, 'Aplicación de Cuadrados mínimos para Precios por Proteínas')
minimosCuadrados_grafico(df_grasas, 'Aplicación de Cuadrados mínimos para Precios por Grasas')

def minimos_cuadrados(x, y):
    """
    Calcula los coeficientes de la línea de regresión lineal y = mx + b
    utilizando el método de mínimos cuadrados.

    Parámetros:
    x -- vector de valores independientes
    y -- vector de valores dependientes

    Retorna:
    m -- pendiente de la línea de regresión
    b -- intersección con el eje y
    """
    A = np.vstack([x, np.ones(len(x))]).T
    ATA = A.T @ A
    ATy = A.T @ y

    # Resolver el sistema de ecuaciones
    coef = np.linalg.solve(ATA, ATy)
    m, b = coef

    return m, b

def grafico_promedio_nutrientes(grasas, hc, proteinas):
    """
    Genera un gráfico de dispersión con ajuste por mínimos cuadrados
    para los precios promedio por gramo de grasas, carbohidratos (HC)
    y proteínas a lo largo del tiempo.

    Parámetros:
    grasas -- DataFrame con los precios de grasas por productos
    hc -- DataFrame con los precios de carbohidratos por productos
    proteinas -- DataFrame con los precios de proteínas por productos
    """

    precios_promedio_grasas = []
    precios_promedio_hc = []
    precios_promedio_proteinas = []

    for fecha in meses:
        # Calcular el precio promedio por gramo de grasas
        precio_promedio_grasas = grasas[fecha].mean()
        precios_promedio_grasas.append(precio_promedio_grasas)

                # Calcular el precio promedio por gramo de HC
        precio_promedio_hc = hc[fecha].mean()
        precios_promedio_hc.append(precio_promedio_hc)

                # Calcular el precio promedio por gramo de proteínas
        precio_promedio_proteinas = proteinas[fecha].mean()
        precios_promedio_proteinas.append(precio_promedio_proteinas)

    # Crear el gráfico de dispersión y ajuste por mínimos cuadrados
    plt.figure(figsize=(12, 8))

    x = np.arange(len(meses))
    m, b = minimos_cuadrados(x, precios_promedio_grasas)
    plt.scatter(x, precios_promedio_grasas, label='Grasas', color='blue')
    plt.plot(x, m * x + b, '-', label='Grasas (Interpolación)', color='blue')

    m, b = minimos_cuadrados(x, precios_promedio_hc)
    plt.scatter(x, precios_promedio_hc, label='HC', color='green')
    plt.plot(x, m * x + b, '-', label='HC (Interpolación)', color='green')

    m, b = minimos_cuadrados(x, precios_promedio_proteinas)
    plt.scatter(x, precios_promedio_proteinas, label='Proteínas', color='red')
    plt.plot(x, m * x + b, '-', label='Proteínas (Interpolación)', color='red')

    plt.xlabel('Fecha')
    plt.ylabel('Precio Promedio por Gramo de MacroNutriente')
    plt.title('Precio Promedio por Gramo de Macronutriente en funcion del tiempo')
    plt.xticks(x, meses, rotation=45)
    plt.legend()
    plt.grid(True)


    plt.show()
    
grafico_promedio_nutrientes(df_grasas, df_hc, df_proteinas)

#%%
"""
consigna 6
"""
def calcular_aumentos(df):
    mes_indices = np.array([0, 1, 2, 3, 4])  # Asume que tenes 5 meses de datos
    aumentos = {}

    for alimento in df.index:
        precios = df.loc[alimento].values.astype(np.float64)

        A = np.vstack([mes_indices, np.ones(len(mes_indices))]).T
        ATA = A.T @ A
        ATy = A.T @ precios

        coef = np.linalg.solve(ATA, ATy)
        m, b = coef

        aumentos[alimento] = m

    return aumentos

aumentos_hc = calcular_aumentos(df_hc)
aumentos_proteinas = calcular_aumentos(df_proteinas)
aumentos_grasas = calcular_aumentos(df_grasas)

#convertimos los diccionarios a dataframes

aumentos_hc = pd.DataFrame(list(aumentos_hc.items()), columns=['Alimento', 'Aumento HC'])
aumentos_proteinas = pd.DataFrame(list(aumentos_proteinas.items()), columns=['Alimento', 'Aumento Proteínas'])
aumentos_grasas = pd.DataFrame(list(aumentos_grasas.items()), columns=['Alimento', 'Aumento Grasas'])

#unimos los datframes
aumentos_nutrientes = pd.merge(aumentos_hc, aumentos_proteinas, on='Alimento', how='outer')
aumentos_nutrientes = pd.merge(aumentos_nutrientes, aumentos_grasas, on='Alimento', how='outer')

#completamos las columnas con cero, que no fueron consideradas aal separar por macronutriente
aumentos_nutrientes = aumentos_nutrientes.fillna(0)
print(aumentos_nutrientes.head())

#Adaptamos la función calcular aumentos para poder usarla con consumidores libres
#calculamos aumento por alimento

#%%
"""
consigna 7
"""
def calcular_aumentos_productos(df):
    """
    Calcula la tasa de aumento de precios de diferentes alimentos
    utilizando el método de mínimos cuadrados a lo largo de cinco meses.

    Parámetros:
    df -- DataFrame donde las filas son alimentos y las columnas son
          los precios mensuales.

    Retorna:
    aumentos -- Diccionario con la tasa de aumento de precios para
                cada alimento.
    """
    mes_indices = np.array([0, 1, 2, 3, 4])  # Asume que tienes 5 meses de datos
    aumentos = {}

    for producto in df['PRODUCTOS']:
        precios = df[df['PRODUCTOS'] == producto].iloc[0, 2:].values.astype(np.float64)

        #ecuaciones normales
        A = np.vstack([mes_indices, np.ones(len(mes_indices))]).T
        ATA = A.T @ A
        ATy = A.T @ precios

        coef = np.linalg.solve(ATA, ATy)
        m, b = coef

        aumentos[producto] = m

    return aumentos


# Calcular los aumentos (obtenemos la pendiente como tasa de crecimiento para la interpolacion)
aumentos_productos = calcular_aumentos_productos(consumidores_libres)
#convertimos el diccionario a un dataframe
aumentos_productos = pd.DataFrame(list(aumentos_productos.items()), columns=['Productos', 'Aumento'])
print(aumentos_productos.head())

# ordenamos para una mejor visualizacion
aumentos_productos = aumentos_productos.sort_values(by='Aumento', ascending=False)
#listamos las carnes y las distinguimos con otro color
carnes = ['asado','bola de lomo','carne picada comun','paleta']


plt.figure(figsize=(12, 8))
colores = ['purple' if producto not in carnes else 'green' for producto in aumentos_productos['Productos']]

plt.barh(aumentos_productos['Productos'], aumentos_productos['Aumento'], color=colores)
plt.xlabel('Aumento (AR$)')
plt.ylabel('Producto')
plt.title('Aumento de Precios de Productos Calculado con Mínimos Cuadrados ')
plt.grid(True)
plt.show()

def calcular_aumento_porcentual(df, fecha_inicio, fecha_fin):
    """
    Calcula el aumento porcentual de precios de productos dadas dos fechas

    Parámetros:
    df -- DataFrame que contiene los precios de los productos en diferentes fechas.
    fecha_inicio -- La fecha de inicio como una cadena en el formato 'dd/mm/yyyy'.
    fecha_fin -- La fecha de fin como una cadena en el formato 'dd/mm/yyyy'.

    Retorna:
    df_resultado -- DataFrame con los productos y su correspondiente aumento porcentual de precio
                    entre las dos fechas dadas
    """
    # Calcula el aumento porcentual para cada fila entre las columnas especificadas
    df['Aumento %'] = ((df[fecha_fin] - df[fecha_inicio]) / df[fecha_inicio]) * 100

    # Crea un nuevo DataFrame solo con las columnas 'PRODUCTOS' y 'Aumento %'
    df_resultado = df[['PRODUCTOS', 'Aumento %']].copy()

    return df_resultado

#creamos un dataframe con el nombre de los alimentos y su aumento porcentual
aumento_porcentual_productos = calcular_aumento_porcentual(consumidores_libres.copy(), '31/12/2023', '30/4/2024')

#ordenamos de manera descencente
aumento_porcentual_productos = aumento_porcentual_productos.sort_values(by='Aumento %', ascending=False)
carnes = ['asado', 'bola de lomo', 'carne picada comun', 'paleta']

plt.figure(figsize=(12, 8))
colores = ['green' if producto not in carnes else 'lightblue'
           for producto in aumento_porcentual_productos['PRODUCTOS']]
plt.barh(aumento_porcentual_productos['PRODUCTOS'], aumento_porcentual_productos['Aumento %'], color=colores)
plt.xlabel('Aumento %')
plt.ylabel('Producto')
plt.title('Aumento porcentual de Productos en el periodo Diciembre 2023- Abril 2024')
plt.grid(True)
plt.show()

carnes = ['asado', 'bola de lomo', 'carne picada','paleta ']

#seleccionamos las carnes para ver sus valores
carnes_df = nutricional_filtrada[nutricional_filtrada['Alimento'].isin(carnes)]

columnas = nutricional_filtrada.columns
carnes_nutrientes = carnes_df[columnas]

#tomamos las columnass que queremos ver
columnas_interes = ['Alimento', 'Cantidad (gr/ml)','HC (gr)', 'Proteinas (gr)',
       'Grasas (gr)']

carnes_nutrientes_seleccionado = carnes_nutrientes[columnas_interes]
print(carnes_nutrientes_seleccionado)

 #extraigo el porcentaje
carnes_porcentaje = aumento_porcentual_productos[aumento_porcentual_productos['PRODUCTOS'].isin(carnes)]
resultado = carnes_porcentaje['Aumento %'].mean()

print(f'El porcentaje de aumento de la carne es: {resultado.round(2)}%')

# Ordenar el DataFrame por la columna 'Proteinas' en orden descendente
df_ordenado = nutricional_filtrada.sort_values(by='Proteinas (gr)', ascending=False)

# Seleccionar las columnas 'HC (gr)', 'Proteinas (gr)' y 'Grasas (gr)'
df_seleccionado = df_ordenado[['Alimento','HC (gr)', 'Proteinas (gr)', 'Grasas (gr)']]

# Mostrar los primeros cinco elementos
df_seleccionado.head()

df_ordenado = nutricional_filtrada.sort_values(by='Grasas (gr)', ascending=False)

df_seleccionado = df_ordenado[['Alimento','HC (gr)', 'Proteinas (gr)', 'Grasas (gr)']]

df_seleccionado.head()

# Listamos las carnes
carnes = ['asado', 'bola de lomo', 'carne picada comun', 'paleta']

# manejamos excepciones
similares = {'carne picada comun': 'carne picada '}

#guardamos el porcentaje de aumento para las carnes
carnes_porcentaje = aumento_porcentual_productos[aumento_porcentual_productos['PRODUCTOS'].isin(carnes)]

# creamos una copia para reducir el aumento, sin modificar la tabla original
cba_reducida = tabla_nutricional.copy()

# Iteramos sobre cada fila del dataframe de aumento porcentual
for index, row in carnes_porcentaje.iterrows():
    alimento = row['PRODUCTOS']
    aumento_porcentaje = row['Aumento %']

    if alimento in similares:
        alimento = similares[alimento]

    # Buscar el alimento en cba_reducida y aplicar la reducción
    mask = cba_reducida['Alimento'].str.strip() == alimento.strip()

    if mask.any():
        # Calcular el factor de reducción
        factor_reduccion = (100 - aumento_porcentaje) / 100.0

        # Reducir los valores en las columnas relevantes
        cba_reducida.loc[mask, 'Cantidad (gr/ml)'] *= factor_reduccion
        cba_reducida.loc[mask, 'HC (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'Proteinas (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'Grasas (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'Na (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'Ca (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'Fe (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'Azucares Libres (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'AGS (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'AGNI (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'AG p (gr)'] *= factor_reduccion
        cba_reducida.loc[mask, 'Fibra (gr)'] *= factor_reduccion

print('Vemos como quedan los margenes al reducir el consumo')
print('*'*50)
print(evaluarCumplimiento_dieta_margenes(cba_reducida))
print('*'*50)
print('+'*50)


print('*'*50)
print('Comparamos con los margenes de la tabla original')
print(evaluarCumplimiento_dieta_margenes(tabla_nutricional))
print('*'*50)
##################################
#   HASTA ACA QUEDA IGUAL
###################################
#%%
"""
consigna7
"""
carnes = ['asado', 'bola de lomo', 'carne picada', 'paleta ']
reduccion = 0.185
#Copiamos el dataframe para no modificar el original, trabajamos con aquellos alimentos para los cuales tenemos precio
reducida = nutricional_filtrada.copy()

# Aplicar la reducción del 18.5% a las filas correspondientes
reducida.loc[reducida['Alimento'].isin(carnes), ['Cantidad (gr/ml)', 'HC (gr)', 'Proteinas (gr)', 'Grasas (gr)', 'Na (gr)',
                                                  'Ca (gr)', 'Fe (gr)', 'Azucares Libres (gr)', 'AGS (gr)', 'AGNI (gr)']] *= (1 - reduccion)

# Calculamos la diferencia entre las filas originales y las reducidas para ver los nutrientes a compensar
fila_original = nutricional_filtrada[nutricional_filtrada['Alimento'].isin(carnes)].iloc[:, 1:]
fila_reducida = reducida[reducida['Alimento'].isin(carnes)].iloc[:, 1:]
diferencia_filas = fila_original - fila_reducida


print('Nutrientes a compensar al reducir las carnes')
print(diferencia_filas)

#obtenemos el numero de cluster correspondiente a las carnes
nro_cluster = df_clusters_consumidores.loc[df_clusters_consumidores['Alimento'] == 'asado','Número de Cluster'].values[0]

# Filtrar los alimentos que pertenecen al mismo clúster
alimentos_mismo_cluster = df_clusters_consumidores[df_clusters_consumidores['Número de Cluster'] == nro_cluster]

print(alimentos_mismo_cluster)

#guardamos el aumento porcentual de la carne, calculado anteriormente.
carne_aumento = resultado.round(2)

#creamos una columna que refleje el aumento en pesos de los productos
consumidores_libres['Aumento $'] = consumidores_libres['30/4/2024'] - consumidores_libres['31/12/2023']


# Fusionamos los DataFrames aumento_porcentual_productos y consumidores_libres
aumento_porcentual_productos = pd.merge(aumento_porcentual_productos, consumidores_libres[['PRODUCTOS', 'Aumento $']], on='PRODUCTOS')

aumento_porcentual_productos.head(3)


carnes_precio = aumento_porcentual_productos[aumento_porcentual_productos['PRODUCTOS'].isin(carnes)]
limite_precio = carnes_precio['Aumento $'].mean()

print('Precio limite : ',limite_precio)

#en promedio la carne aumentó 2400 pesos,
#asi que sacamos cuales alimentos estan por debajo de la mitad de ese aumento
#filtramos por limite_precio/2
alimentos_mismo_cluster.loc[:, 'Alimento'] = alimentos_mismo_cluster['Alimento'].replace({
    'huevo': 'huevos color docena',
    'zanahoria': 'zanahorias'
})

alimentos_mismo_cluster=filtrarAlimentos(alimentos_mismo_cluster, aumento_porcentual_productos)

#ahora nos quedamos con aquellos alimentos bajo el limite
posibles_sustitutos = alimentos_mismo_cluster[alimentos_mismo_cluster['Aumento $'] <= limite_precio/2]
print(posibles_sustitutos)


#%%
"""
consigna 8
"""


carnes_index = ['asado','bola de lomo','carne picada','paleta']

# Identificamos  y guardamos los índices de las carnes
indices_carnes = [i for i, alimento in enumerate(alimentos) if alimento.strip().lower() in carnes_index]

# Calculamos el centroide de las carnes en el espacio proyectado
centroide_carnes = np.mean(X_proyectado[indices_carnes], axis=0)

# Calculamos la distancia euclidiana de cada punto al centroide de las carnes
distancias_al_centroide = np.linalg.norm(X_proyectado - centroide_carnes, axis=1)

# Ordenamos los índices por distancia descendente y seleccionamos los más lejanos
num_puntos_lejanos = 3 # Número de puntos a seleccionar
indices_mas_lejanos = np.argsort(distancias_al_centroide)[-num_puntos_lejanos:]

# Obtenemos los alimentos más lejanos y sus distancias
alimentos_mas_lejanos = alimentos[indices_mas_lejanos]
distancias_mas_lejanos = distancias_al_centroide[indices_mas_lejanos]

print("Alimentos más lejanos de las carnes:")
for alimento, distancia in zip(alimentos_mas_lejanos, distancias_mas_lejanos):
    print(f"{alimento}: {distancia:.2f}")


plt.figure(figsize=(12, 8))
plt.scatter(X_proyectado[:, 0], X_proyectado[:, 1], alpha=0.5, label='Todos los alimentos')
plt.scatter(X_proyectado[indices_carnes, 0], X_proyectado[indices_carnes, 1], color='red', label='Carnes')
plt.scatter(X_proyectado[indices_mas_lejanos, 0], X_proyectado[indices_mas_lejanos, 1], color='blue', label='Más lejanos de las carnes')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Proyección de Alimentos en el Espacio Generado por ACP')
plt.legend()
plt.grid(True)
plt.show()
