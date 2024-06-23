# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:19:29 2024

@author: Rocio
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

#%%
alimentos = tabla_nutricional['Alimento'].values
productos=consumidores_libres.iloc[:,0]

#armo una funcion donde tomo ambas columnas y encuentro sus valores en la otra y operando matematicamente
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

#%%

# Lista de productos que deseas unificar bajo la categoría 'carnes'
carnes = ['bola de lomo', 'asado', 'paleta', 'carne picada comun']

# Filtrar las filas correspondientes a los productos de la lista 'carnes'
carnes_df = consumidores_libres[consumidores_libres['PRODUCTOS'].isin(carnes)]

# Crear una nueva fila sumarizando los valores para los productos de la lista 'carnes'
sum_carnes_precio = pd.DataFrame({
    'PRODUCTOS': ['carnes'],
    'Cantidad': [carnes_df['Cantidad'].sum()],
    '31/12/2023': [carnes_df['31/12/2023'].sum()],
    '31/1/2024': [carnes_df['31/1/2024'].sum()],
    '29/2/2024': [carnes_df['29/2/2024'].sum()],
    '31/3/2024': [carnes_df['31/3/2024'].sum()],
    '30/4/2024': [carnes_df['30/4/2024'].sum()]
})

# Eliminar las filas originales de los productos 'carnes' del DataFrame
consumidores_libres = consumidores_libres[~consumidores_libres['PRODUCTOS'].isin(carnes)]

# Concatenar la nueva fila 'carnes' al DataFrame
consumidores_libres = pd.concat([consumidores_libres, sum_carnes_precio], ignore_index=True)

#%%
#unificamos las carnes

#aca me guardo la informacion de los alimentos que tengo en consumidores libres
nutricional_filtrada = filtrarAlimentos(consumidores_libres,tabla_nutricional)
#elimino berenjena
# Eliminar la fila correspondiente a "berenjenas"
consumidores_libres = consumidores_libres[consumidores_libres['PRODUCTOS'] != 'berenjenas']
# data =[]

carnes = ['asado', 'bola de lomo', 'carne picada', 'paleta ']

# Filtrar las filas correspondientes a los alimentos de la lista 'carnes'
carnes_df = nutricional_filtrada[nutricional_filtrada['Alimento'].isin(carnes)]

# Crear una nueva fila sumarizando los valores para los alimentos de la lista 'carnes'
sum_carnes = pd.DataFrame({
    'Alimento': ['carnes'],
    'Cantidad (gr/ml)': [carnes_df['Cantidad (gr/ml)'].sum()],
    'HC (gr)': [carnes_df['HC (gr)'].sum()],
    'Proteinas (gr)': [carnes_df['Proteinas (gr)'].sum()],
    'Grasas (gr)': [carnes_df['Grasas (gr)'].sum()],
    'Na (gr)': [carnes_df['Na (gr)'].sum()],
    'Ca (gr)': [carnes_df['Ca (gr)'].sum()],
    'Fe (gr)': [carnes_df['Fe (gr)'].sum()],
    'Azucares Libres (gr)': [carnes_df['Azucares Libres (gr)'].sum()],
    'AGS (gr)': [carnes_df['AGS (gr)'].sum()],
    'AGNI (gr)': [carnes_df['AGNI (gr)'].sum()],
    'AG p (gr)': [carnes_df['AG p (gr)'].sum()],
    'Fibra (gr)': [carnes_df['Fibra (gr)'].sum()]
})

# Eliminar las filas originales de los alimentos 'carnes' del DataFrame
nutricional_filtrada = nutricional_filtrada[~nutricional_filtrada['Alimento'].isin(carnes)]

# Concatenar la nueva fila 'carnes' al DataFrame
nutricional_filtrada = pd.concat([nutricional_filtrada, sum_carnes], ignore_index=True)


#reducimos la fila carnes un 18,5%

reducida = nutricional_filtrada.copy()

# Reducir la fila "carnes" en un 18.5%
reducida.loc[reducida['Alimento'] == 'carnes', ['Cantidad (gr/ml)', 'HC (gr)', 'Proteinas (gr)', 'Grasas (gr)', 'Na (gr)', 
                                                'Ca (gr)', 'Fe (gr)', 'Azucares Libres (gr)', 'AGS (gr)', 'AGNI (gr)']] *= 0.815  # 1 - 0.185



