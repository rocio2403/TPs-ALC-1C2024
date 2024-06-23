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
import matplotlib.colors as mcolors


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

del df_clusters2
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


#aca me guardo la informacion de los alimentos que tengo en consumidores libres
nutricional_filtrada = filtrarAlimentos(consumidores_libres,tabla_nutricional)
#elimino berenjena
# Eliminar la fila correspondiente a "berenjenas"
consumidores_libres = consumidores_libres[consumidores_libres['PRODUCTOS'] != 'berenjenas']
# data =[]



meses = ['31/12/2023', '31/1/2024', '29/2/2024', '31/3/2024', '30/4/2024']

# Inicializar diccionarios para almacenar los precios por gramos de cada nutriente
precios_hc = {}
precios_proteinas = {}
precios_grasas = {}
precios_carne ={}
ver = []
casos_excepcionales ={
        'aceite girasol': 'aceite cocinero girasol',
    'fideos secos': 'fideos guiseros',
    'harina trigo': 'harina de trigo',
    'leche fluida entera': 'leche c/vit.fort',
    'pan frances': 'pan fresco',
       }

# Iterar sobre cada alimento en la tabla nutricional
for _, row in nutricional_filtrada.iterrows():
    alimento = row['Alimento'] #agarra el alimento de tabla nutricional
    cant_ali = row['Cantidad (gr/ml)'] #agarra la cantidad de ese alimento para ver proteina
    gr_hc = row['HC (gr)']
    gr_proteinas = row['Proteinas (gr)']
    gr_grasas = row['Grasas (gr)']
    # Buscar alimento en consumidores libres
    if alimento in casos_excepcionales.keys():
       alimento =  casos_excepcionales[alimento]
      
    match = consumidores_libres[consumidores_libres['PRODUCTOS'].str.contains(alimento)]
    if not match.empty: # para las coincidencias
        cant_con_precio = match.iloc[0]['Cantidad'] #agarra la cantidad  para la cual tenemos precio
        precios = match.iloc[0][meses].values
        
        # Calcular precios por gramo de nutriente
        if gr_hc > 0:
            #aplico regla de tres
            x = (gr_hc*cant_con_precio)/cant_ali
            #ahora veo para los gramos dados
            y = (precios*x)/cant_con_precio
            #ahora saco para un gramo
            precio_hc = y/x
            precios_hc[alimento] = precio_hc
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
 

df_hc = pd.DataFrame(precios_hc, index=meses).T
df_proteinas = pd.DataFrame(precios_proteinas, index=meses).T
df_grasas = pd.DataFrame(precios_grasas, index=meses).T

# Función para realizar la regresión lineal y graficar
def regresion_y_grafico(df, titulo):
    plt.figure(figsize=(12, 8))
    mes_indices = np.array([0, 1, 2, 3, 4])  # Asume que tienes 5 meses de datos
    colors = list(mcolors.TABLEAU_COLORS.values())  # Lista de colores predefinidos

    for i, alimento in enumerate(df.index):
        precios = df.loc[alimento].values.astype(np.float64)  # Convertir a float64
        
        # Realizar la regresión lineal
        A = np.vstack([mes_indices, np.ones(len(mes_indices))]).T
        ATA = A.T @ A
        ATy = A.T @ precios
        
        # Resolver el sistema de ecuaciones
        coef = np.linalg.solve(ATA, ATy)
        m, b = coef
        
        # Seleccionar el color
        color = colors[i % len(colors)]
        
        # Graficar precios originales
        plt.plot(mes_indices, precios, 'o', label=f'{alimento} (original)', color=color)
        
        # Graficar la línea de regresión
        plt.plot(mes_indices, m * mes_indices + b, '-', label=f'{alimento} (regresión)', color=color)
        
        # Configurar etiquetas y título
        plt.xlabel('Meses')
        plt.ylabel('Precio por gramo de nutriente')
        plt.title(titulo)
    plt.grid(True)
       
    # Configurar la leyenda fuera del gráfico
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(mes_indices, ['Dic23', 'Ene24', 'Feb24', 'Mar24', 'Abr24'])
    plt.show()

regresion_y_grafico(df_hc, 'Regresión Lineal de Precios por HC')
regresion_y_grafico(df_proteinas, 'Regresión Lineal de Precios por Proteínas')
regresion_y_grafico(df_grasas, 'Regresión Lineal de Precios por Grasas')

def minimos_cuadrados(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    ATA = A.T @ A
    ATy = A.T @ y
    
    # Resolver el sistema de ecuaciones
    coef = np.linalg.solve(ATA, ATy)
    m, b = coef
    
    return m, b

def grafico_promedio_nutrientes(grasas, hc, proteinas): 
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
    plt.plot(x, m * x + b, '-', label='Grasas (Regresión)', color='blue')
    
    m, b = minimos_cuadrados(x, precios_promedio_hc)
    plt.scatter(x, precios_promedio_hc, label='HC', color='green')
    plt.plot(x, m * x + b, '-', label='HC (Regresión)', color='green')
    
    m, b = minimos_cuadrados(x, precios_promedio_proteinas)
    plt.scatter(x, precios_promedio_proteinas, label='Proteínas', color='red')
    plt.plot(x, m * x + b, '-', label='Proteínas (Regresión)', color='red')
    
    plt.xlabel('Fecha')
    plt.ylabel('Precio Promedio por Gramo de MacroNutriente')
    plt.title('Precio Promedio por Gramo de Macronutriente en funcion del tiempo')
    plt.xticks(x, meses, rotation=45)
    plt.legend()
    plt.grid(True)
    
   
    plt.show()
#veo si funca
grafico_promedio_nutrientes(df_grasas, df_hc, df_proteinas)  

#%%
def calcular_aumentos(df):
    mes_indices = np.array([0, 1, 2, 3, 4])  # Asume que tienes 5 meses de datos
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
#creo dataframes

aumentos_hc = pd.DataFrame(list(aumentos_hc.items()), columns=['Alimento', 'Aumento HC'])
aumentos_proteinas = pd.DataFrame(list(aumentos_proteinas.items()), columns=['Alimento', 'Aumento Proteínas'])
aumentos_grasas = pd.DataFrame(list(aumentos_grasas.items()), columns=['Alimento', 'Aumento Grasas'])

aumentos_nutrientes = pd.merge(aumentos_hc, aumentos_proteinas, on='Alimento', how='outer')
aumentos_nutrientes = pd.merge(aumentos_nutrientes, aumentos_grasas, on='Alimento', how='outer')

# Llenar NaN con 0
aumentos_nutrientes = aumentos_nutrientes.fillna(0)


"""
Consigna 6.- Comparar el aumento de la carne en comparaci´on con los otros
rubros. Si la gente consume ese porcentaje menos de carne, como queda la ingesta
individual con respecto a la tabla de metas de la OMS?"

"""
#utilizo aumentos para cada alimento de consumidores libres y luego hago grafiquito de barras
#modifico calcular aumento para poder usarla con consumidores libres

def calcular_aumentos_productos(df):
    mes_indices = np.array([0, 1, 2, 3, 4])  # Asume que tienes 5 meses de datos
    aumentos = {}

    for producto in df['PRODUCTOS']:
        precios = df[df['PRODUCTOS'] == producto].iloc[0, 2:].values.astype(np.float64)

        A = np.vstack([mes_indices, np.ones(len(mes_indices))]).T
        ATA = A.T @ A
        ATy = A.T @ precios

        coef = np.linalg.solve(ATA, ATy)
        m, b = coef

        aumentos[producto] = m

    return aumentos

aumentos_productos = calcular_aumentos_productos(consumidores_libres)
aumentos_productos = pd.DataFrame(list(aumentos_productos.items()), columns=['Productos', 'Aumento'])
print(aumentos_productos.head())

aumentos_productos = aumentos_productos.sort_values(by='Aumento', ascending=False)
carnes = ['asado','bola de lomo','carne picada comun','paleta']


plt.figure(figsize=(12, 8))
colores = ['purple' if producto not in carnes else 'green' for producto in aumentos_productos['Productos']]

plt.barh(aumentos_productos['Productos'], aumentos_productos['Aumento'], color=colores)
plt.xlabel('Aumento')
plt.ylabel('Producto')
plt.title('Aumento de Productos con Cuadrados Minimos vs Alimento')
plt.grid(True)
plt.show()



#%%

def calcular_aumento_porcentual(df, fecha_inicio, fecha_fin):
    # Calcula el aumento porcentual para cada fila entre las columnas especificadas
    df['Aumento %'] = ((df[fecha_fin] - df[fecha_inicio]) / df[fecha_inicio]) * 100
    
    # Crea un nuevo DataFrame solo con las columnas 'PRODUCTOS' y 'Aumento %'
    df_resultado = df[['PRODUCTOS', 'Aumento %']].copy()
    
    return df_resultado

aumento_porcentual_productos = calcular_aumento_porcentual(consumidores_libres.copy(), '31/12/2023', '30/4/2024')


aumento_porcentual_productos = aumento_porcentual_productos.sort_values(by='Aumento %', ascending=False)
carnes = ['asado', 'bola de lomo', 'carne picada comun', 'paleta']

plt.figure(figsize=(12, 8))
colores = ['green' if producto not in carnes else 'lightblue' 
           for producto in aumento_porcentual_productos['PRODUCTOS']]
plt.barh(aumento_porcentual_productos['PRODUCTOS'], aumento_porcentual_productos['Aumento %'], color=colores)
plt.xlabel('Aumento %')
plt.ylabel('Producto')
plt.title('Aumento porcentual de Productos vs Alimento')
plt.grid(True)
plt.show()

#%%

#%%

"""
Si la gente consume ese porcentaje menos de carne, como queda la ingesta
individual con respecto a la tabla de metas de la OMS?"
"""
#extraigo el porcentaje de aumentos de la carne
carnes = ['asado', 'bola de lomo', 'carne picada','paleta ']

# Filtrar el DataFrame para los alimentos en la lista 'carnes'
carnes_df = nutricional_filtrada[nutricional_filtrada['Alimento'].isin(carnes)]


carnes = ['asado', 'bola de lomo', 'carne picada comun','paleta ']

#extraigo el porcentaje 
carnes_porcentaje = aumento_porcentual_productos[aumento_porcentual_productos['PRODUCTOS'].isin(carnes)]
resultado = carnes_porcentaje['Aumento %'].mean()
print(f'El porcentaje de aumento de la carne es: {resultado.round(2)}%')

#%%

# Lista de nombres de carne con porcentaje
carnes = ['asado', 'bola de lomo', 'carne picada comun', 'paleta']

# Suponiendo que tienes cargados los dataframes aumento_porcentual_productos y tabla_nutricional

# Diccionario para manejar nombres similares
similares = {'carne picada comun': 'carne picada '}

# Extraer el porcentaje de aumento para los productos de carne
carnes_porcentaje = aumento_porcentual_productos[aumento_porcentual_productos['PRODUCTOS'].isin(carnes)]
resultado = carnes_porcentaje['Aumento %'].mean()
print(f'El porcentaje de aumento de la carne es: {resultado.round(2)}%')

# Crear una copia de la tabla nutricional reducida
cba_reducida = tabla_nutricional.copy()

# Iterar sobre cada fila del dataframe de aumento porcentual
for index, row in carnes_porcentaje.iterrows():
    alimento = row['PRODUCTOS']
    aumento_porcentaje = row['Aumento %']
    
    # Verificar si el alimento está en el diccionario de nombres similares
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


#%%%%%%%%%%%%%%%%%%%%%%%%%%
#        CONSIGNA 7 
 
"""
En este caso vamos a utilizar el dominio generado por ACP para lo siguiente.
Consigna 7.-
Proponer dos alimentos de la tabla 3 (consumidores libres)
 que lleguen a reemplazar la
disminuci´on del 18.5 % de la carne sin perder valor nutricional. 
Corroborar con la tabla de metas de la OMS la dieta diaria obtenida.
 Adem´as, se pone como restricci´on en la elecci´on, que el precio de los alimentos 
de reemplazo debe ser menor al 50 % del aumento de la carne.
"""

#como tengo que usar la tabla tres(lad e consumidores libres, utilizo nutricional_filtrada)
#la proyecto en acp
#trabajo con df_clusters

"""
hacer el 18.5 porceinto de los valores nutricionales que aporta la carne y ver
que alimentos pueden reemplazar esa cantidad

tomar el porcetnaje del porcetaje
agarras el precio de la carne, ves cuanto es el 41,47 y del resultado tomo el 50%
filtro los alimentos cuyo precio este debajo de eso y hago un segundo filtro con los que tengan los valores nutriiconales necesarios
el acp reduce los valores nutriiconales que tomamos
"""

"""
Consigna 7: Reemplazo dietario de la carne
Objetivo: Proponer dos alimentos de la tabla 3 que reemplacen la disminución del 18.5% del consumo de carne sin perder valor nutricional, y que el precio de estos alimentos sea menor al 50% del aumento de la carne.

Pasos:

Calcular la disminución del consumo de carne:
   18.5% menos de carne (por ejemplo, si originalmente consumías 1000g de carne, ahora consumirías 815g).
Determinar la cantidad de nutrientes que se dejan de consumir:
    
Calcular los nutrientes que aporta la cantidad de carne que se dejará de consumir.
Buscar alimentos de reemplazo en la tabla 3:
Seleccionar alimentos que aporten nutrientes similares a los que se pierden por la reducción de carne.
Asegurarse que el costo de los alimentos seleccionados sea menor al 50% del aumento del precio de la carne.
4.Verificar la dieta diaria obtenida con la tabla de metas de la OMS:
Asegurarse que la nueva combinación de alimentos cumple con las metas nutricionales diarias.
"""

# Lista de nombres de carne con porcentaje
carnes = ['asado', 'bola de lomo', 'carne picada', 'paleta ']

# primero disminuyo un 18,5 porciento  la carne



df_nutricional_carne_redu = tabla_nutricional.copy()

# Iterar sobre cada alimento en la lista
for carne in carnes:
    mask = df_nutricional_carne_redu['Alimento'].str.contains(carne, case=False, regex=False)
    df_nutricional_carne_redu.loc[mask, df_nutricional_carne_redu.columns[1:]] *= 0.815

#ahora veamos cuanto es la diferencia de nutrientes al hacer esta disminucion
mask_original = tabla_nutricional['Alimento'].str.contains('|'.join(carnes), case=False, regex=True)
mask_redu = df_nutricional_carne_redu['Alimento'].str.contains('|'.join(carnes), case=False, regex=True)

# Sumar los valores de cada columna para los alimentos de la lista 'carnes' en ambos DataFrames :)
suma_original = nutricional_filtrada.loc[mask_original, nutricional_filtrada.columns[1:]].sum()
suma_redu = df_nutricional_carne_redu.loc[mask_redu, df_nutricional_carne_redu.columns[1:]].sum()

# Calcular la diferencia
#VEMOS CUANTO MENOS SE CONSUME
nutrientes_a_compensar= suma_original - suma_redu
nutrientes_a_compensar = pd.DataFrame([nutrientes_a_compensar.values], columns=nutrientes_a_compensar.index) #lo volteo

#ahora debo encontrar 2 alimentos que esten en consumidores libres (y en tabla nutricional por supuesto)
# que compensen esa reducciòn
#pero, su valor (en pesos) tiene que ser menor al 50% del aumento de la carne, la carne aumentò un 41.48%
aumento_carne = 41.48
"""
En este caso vamos a utilizar el dominio generado por ACP para lo siguiente.
Consigna 7.-
Proponer dos alimentos de la tabla 3 (consumidores libres)
 que lleguen a reemplazar la
disminuci´on del 18.5 % de la carne sin perder valor nutricional. 
Corroborar con la tabla de metas de la OMS la dieta diaria obtenida.
 Adem´as, se pone como restricci´on en la elecci´on, que el precio de los alimentos 
de reemplazo debe ser menor al 50 % del aumento de la carne.
"""
#REESTRICCION
#FILTRO PRIMERO POR LOS QUE CUMPLAN QUE SUS VALORES ESTAN POR DEBAJO DE LA MITAD PARA CUMPLIR LA COCMPENSACION
# Encontrar alimentos sustitutos que compensen la diferencia de nutrientes
#utilizo nutricional_filtrada, que tiene los alimentos para los que tenemos precios
# Reiniciar los índices para ambos DataFrames
nutricional_filtrada.reset_index(drop=True, inplace=True)
nutrientes_a_compensar.reset_index(drop=True, inplace=True)

#ya tenog los valores de nutrientes a compensar
columnas_comparar = nutrientes_a_compensar.columns

# Aplicamos el filtro
# Aplicamos el filtro
for columna in nutrientes_a_compensar.columns:
    posibles_sustitutos = nutricional_filtrada[nutricional_filtrada[columna] < nutrientes_a_compensar[columna].iloc[0]]
