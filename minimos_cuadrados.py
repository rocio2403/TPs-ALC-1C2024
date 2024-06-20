# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:52:59 2024

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


alimentos = tabla_nutricional['Alimento'].values

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
    'pan': 'pan fresco',
       }
carnes = ['asado','bola de lomo','carne picada','paleta']

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
            if alimento in carnes:
                precios_carne[alimento] = precio_hc
            else:
               precios_hc[alimento] = precio_hc
        if gr_proteinas > 0:
            #aplico regla de tres
            x = (gr_proteinas*cant_con_precio)/cant_ali
            #ahora veo para los gramos dados
            y = (precios*x)/cant_con_precio
            #ahora saco para un gramo
            precio_proteinas = y/x
            if alimento in carnes:
                precios_carne[alimento] = precio_proteinas
            else:
            
                precios_proteinas[alimento] = precio_proteinas
        if gr_grasas > 0:
            #aplico regla de tres
            x = (gr_grasas*cant_con_precio)/cant_ali
            #ahora veo para los gramos dados
            y = (precios*x)/cant_con_precio
            #ahora saco para un gramo
            precio_grasas = y/x
            if alimento in carnes:
                precios_carne[alimento] = precio_grasas
            else:
            
                precios_grasas[alimento] = precio_grasas
 

df_hc = pd.DataFrame(precios_hc, index=meses).T
df_proteinas = pd.DataFrame(precios_proteinas, index=meses).T
df_grasas = pd.DataFrame(precios_grasas, index=meses).T
df_carne = pd.DataFrame(precios_carne, index=meses).T


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
regresion_y_grafico(df_carne, 'Regresión Lineal de Precios por Carne')


def minimos_cuadrados(x,y):
        
  
    A = np.vstack([x, np.ones(len(x))]).T
    
    ATA = A.T @ A
    ATy = A.T @ y
    
    # Resolver el sistema de ecuaciones
    coef = np.linalg.solve(ATA, ATy)
    m, b = coef
    

    plt.plot(x, y, 'o', label='Datos')
    plt.plot(x, m * x + b, '-', label='mínimos cuadrados')
    
    # Configurar etiquetas y título
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mínimos Cuadrados')
    
    # Configurar la leyenda fuera del gráfico
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Mostrar el gráfico
    plt.show()
#veo si funca
    
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 3, 5, 4, 6])
# minimos_cuadrados(x, y)    








