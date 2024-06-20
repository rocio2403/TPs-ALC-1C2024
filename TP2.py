#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:48:34 2024

@author: Estudiante
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns


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


#Creamos dataframes con los datos en formato csv
tabla_nutricional = pd.read_csv('tabla_nutricional.csv',delimiter= ';')
consumidores_libres = pd.read_csv('consumidores_libres.csv' ,delimiter = ';')

#Normalizamos la tabla
tabla_nutricional = normalizar_tabla_nutricional(tabla_nutricional)

print('Tabla Nutricional', tabla_nutricional.head())

print('Consumidores Libres', consumidores_libres.head())



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
    print(f'Grasas cumple : {grasas_cumple}')
    print(f'HC cumple : {carbohidratos_cumple}')
    print(f'Proteinas cumple : {proteinas_cumple}')
    print(f'Sodio cumple : {sodio_cumple}')
    print(f'Fibra cumple : {fibra_cumple}')
    print(f'Frutas y verduras cumplen : {frutas_verduras_cumple}')

    cumple = proteinas_cumple and carbohidratos_cumple and grasas_cumple and sodio_cumple and fibra_cumple and frutas_verduras_cumple

    return 'Cumple los márgenes' if cumple else 'No cumple los márgenes'

#Verificamos si la CBA cumple los márgenes de la OMS
print(evaluarCumplimiento_dieta_margenes(tabla_nutricional))

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

# Reducimos la dimensionalidad de los datos (X) mediante PCA
matriz_nutricional = tabla_nutricional.iloc[:, 1:].values
X_proyectado, Vt_2= componentes_principalesSVD(matriz_nutricional, 2)


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
# Graficamos la proyección con los clusters y obtenemos el DataFrame de clusters
alimentos = tabla_nutricional['Alimento'].values

df_clusters = graficarProyeccion(X_proyectado, alimentos, 'Análisis en Componentes Principales de la Canasta Básica')

def graficos_cluster():

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
    print(f'cantidad de alimentos cluster 1: {cant1}')
    print(f'cantidad de alimentos cluster 2: {cant2}')
    print(f'cantidad de alimentos cluster 3: {cant3}')
    print(f'cantidad de alimentos cluster 4: {cant4}')

graficos_cluster()
#%%
#################################
#
#AUMENTO DE PRECIOS
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2

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


# Proyectamos los alimentos de consumidores_libres (Y) en el subespacio del PCA anterior
Y = filtrarAlimentos(consumidores_libres, tabla_nutricional)
alimentos2 = Y['Alimento'].values
Y = Y.iloc[:,1:].values
proyector =  Vt_2
Y_proyectado = proyectar(Y, proyector)
# Graficamos la proyección con los clusters y obtenemos el DataFrame de clusters
df_clusters2 = graficarProyeccion(Y_proyectado, alimentos2, "Análisis de Componentes Principales de Alimentos de 'Consumidores Libres'")

#%%%%
######
######
##    MINIMOS CUADRADOS
######
###############
"""

M´ınimos Cuadrados
Consigna 5.- Aplicar M´ınimos Cuadrados sobre los precios de cada nutriente (HC,
Prote´ınas y grasas) para evaluar el aumento en estos 4 meses. Graficar cada punto
de cada alimento y la recta obtenida por M´ınimos Cuadrados. Hacer una tabla
de los aumentos.

"""
# Convertimos los datos a np.array
matriz_consumidores = consumidores_libres.iloc[:, 1:].values
print(matriz_consumidores.shape)
#no todos los alimentos se corresponden con todos, vamos a filtrar
df= tabla_nutricional[['Alimento', 'Cantidad (gr/ml)', 'HC (gr)', 'Proteinas (gr)', 'Grasas (gr)','Fibra (gr)','Azucares Libres (gr)']]
#ahora vamos a ir filtrando sobre los alimentos que tenemos informacion
#consumidores_libres.rename(columns={"PRODUCTOS": "Alimento"}, inplace=True)
#productos =consumidores_libres.iloc[:, 0] = consumidores_libres.iloc[:, 0].str.lower().str.capitalize()

#tengo que aplicar sobre los precios de cada nutriente pero no tengo de todos, por eso filtro productos
palabras= ['Aceite', 'Acelga','Arroz','Azucar','Carne picada','Fideos secos',
                  'Harina trigo','Huevo','Leche fluida entera','Papa','Tomate','Asado','Cebolla',
                  'Manzana','Naranja']
palabras = [p.lower() for p in palabras]


# Convertir todo a minúsculas para hacer la comparación insensible a mayúsculas y minúsculas
df['Alimento'] = df['Alimento'].astype(str).str.lower()
palabras = [p.lower() for p in palabras]
consumidores_libres['PRODUCTOS'] = consumidores_libres['PRODUCTOS'].astype(str)

# Filtrar filas de df donde alguna columna está incluida o es igual a alguna palabra en palabras
filtrado_df = df[df.apply(lambda row: row.astype(str).str.lower().isin(palabras)).any(axis=1)]




data = []
for index, row in filtrado_df.iterrows():
    aux = consumidores_libres[consumidores_libres['PRODUCTOS'].str.contains(row['Alimento'], case=False, na=False)]

    if len(aux) > 0:
        data.append([row['Alimento'],
                     aux['PRODUCTOS'].iloc[0],
                     aux['Cantidad'].iloc[0],
                     aux['31/12/2023'].iloc[0],
                     aux['31/1/2024'].iloc[0],
                     aux['29/2/2024'].iloc[0],
                     aux['31/3/2024'].iloc[0],
                     aux['30/4/2024'].iloc[0]])

# Crear un nuevo DataFrame con los datos recopilados
df_resultado = pd.DataFrame(data, columns=['Alimento', 'Producto', 'Cantidad', '31/12/2023', '31/1/2024', '29/2/2024', '31/3/2024', '30/4/2024'])
df_final = pd.merge(df_resultado, filtrado_df, on='Alimento', how='inner')
# Mostrar el DataFrame resultante
#AL FIN DIOS
#ASI SE FESTEJA EN EL OBESLISCOOOOOOOOOOOO

#MEJORAR CODIGO LUEGO

#AHORA AGREGO LO QUE ME FALTA
#aceite, leche


#ahora aplico minimos cuadrados para predecir el aumento
#quiero aplicarlo sobre el precio de cada nutriente
#como tengo distintos precios para cada alimento y quiero para los nutrientes, saco la media
#elimino las columnas alimento y prodcuto

#%%

#Los alimentos de tabla nutricional estàn parcialemente incluidas
# para hacer la conversion a pesos, tengo en la columna cantidad de consumidores libres el peso en gramos 
#EN CONSUMIDORES LIBRES TENGO LA RELACION PESO CANTIDAD
#EN TABLA NUTRICIONAL TENGO LA RELACION CANTIDAD ALIMENTO --GRAMOS DE MACRONUTRIENTE
#Entonces tendria que ir recorriendo productos, fijarme cual está en consumidores libres y luego hacer la regla de tres
#         250gr fideos --------- 5gr de proteinas
#        1000gr fideos ---------X = 20gr
#LUEGO
#
#        1000gr fideos --------- $300
#          20gr (proteinas)-------X = $6
#

#BASICAMENTE RECOLECTO LOS DATOS COMO DIJO MUY AMABLEMENTE PABLO :/

#LISTO AMBOS

alimentos = tabla_nutricional['Alimento'].values
productos=consumidores_libres.iloc[:,0]

#armo una funcion donde tomo ambas columnas y encuentro sus valores en la otra y operando matematicamente
meses = ['Dic23','Enero24','Febrero24','Marzo24','Abril24'] # EJE X
#obtengo los datos como dije, armo una matriz y 
# # Graficamos
# plt.scatter(datosNP[:,0], datosNP[:,1])

#sigo la logica de palabras contenidas
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

columnas = ['Productos','HC','Proteinas','Grasas','Precios']
datos = pd.DataFrame()
hc_columns = ['Producto','Dic23','Enero24','Febrero24','Marzo24','Abril24'] #completo con los precios
datosHC = pd.DataFrame(columns=hc_columns)

proteinas_columns = ['Producto','Dic23','Enero24','Febrero24','Marzo24','Abril24'] #completo con los precios
datosProt = pd.DataFrame(columns=proteinas_columns)

grasas_columns = ['Producto','Dic23','Enero24','Febrero24','Marzo24','Abril24'] #completo con los precios
datosGRASAS = pd.DataFrame(columns=grasas_columns)

def minimos_cuadrados():
    for alimento in alimentos:
        for row in consumidores_libres.iterrows():
            if palabras_contenidas(alimento, row['PRODUCTOS']): #ENTENDER BIEN COMO FUNCIONA PALABRAS CONTENIDAS 
            #SINO HAGO OTRA FUNCION QUE SE LLAME PRODUCTOS Y PONGO LOS CASOS EXCEPCIONALES
                #ACA MANDO A HACER CONVERSION
                return True
                
                

def conversion(alimento,row):
    cant = consumidores_libres.loc[row]['Cantidad']   
    precios = []
    #para hacer la conversion tengo que encontrar la fila (row) en tabla nutricional
    #recorro tabla_nutricional
    ###########################################
    #ARREGLAR
    gramos = tabla_nutricional.loc[alimento]['HC (gr)']
    return cant,gramos

conversion('Asado',1)
    













