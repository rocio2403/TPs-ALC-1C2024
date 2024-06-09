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


########################################################
#
#No sirve, pero es sudor y lagrimas
#
#######################################################

""""
#%%


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def realizar_acp(data):
    """
    Realiza el Análisis en Componentes Principales (ACP) del valor nutricional de cada alimento por gramo.
    """
    pca = PCA(n_components=2)
    componentes_principales = pca.fit_transform(data.iloc[:, 1:])  # Asumiendo que la primera columna es 'Alimento'

    plt.figure(figsize=(10, 7))
    plt.scatter(componentes_principales[:, 0], componentes_principales[:, 1])

    # for i, alimento in enumerate(data['Alimento']):
    #     plt.text(componentes_principales[i, 0], componentes_principales[i, 1], alimento, fontsize=9)

    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Análisis en Componentes Principales de los Alimentos')
    plt.grid()
    plt.show()

# Realizar el ACP
realizar_acp(tabla_nutricional)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def realizar_svd(data):
    """
    Realiza la descomposición en valores singulares (SVD) del valor nutricional de cada alimento por gramo.
    """
    # Escalar los datos para que tengan media 0 y varianza 1
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.iloc[:, 1:])

    # Aplicar SVD
    U, S, Vt = np.linalg.svd(data_scaled, full_matrices=False)
    
    # Proyectar los datos en el nuevo espacio formado por los vectores singulares
    data_proyectada = np.dot(data_scaled, Vt.T)
    
   # plt.figure(figsize=(10, 7))
    plt.scatter(data_proyectada[:, 0], data_proyectada[:, 1], color = 'green')
    
    # for i, alimento in enumerate(data['Alimento']):
    #     plt.text(data_proyectada[i, 0], data_proyectada[i, 1], alimento, fontsize=9)
    
    plt.xlabel('Componente Principal 1 ')
    plt.ylabel('Componente Principal 2 ')
    plt.title('Análisis en Componentes Principales con SVD de los Alimentos')
    plt.grid()
    plt.show()

    return data_proyectada, U, S, Vt

realizar_svd(tabla_nutricional)
"""
Escalado de los datos: Antes de aplicar SVD, los datos se escalan para que tengan media 0 y varianza 1, lo cual es una práctica común en análisis multivariado para asegurar que todos los atributos tengan el mismo peso.

Aplicación de SVD: Se utiliza np.linalg.svd para descomponer la matriz de datos escalados en tres matrices: U, S y Vt.

Proyección en el nuevo espacio: Los datos originales se proyectan en el nuevo espacio formado por los vectores singulares. Luego, se grafica la proyección de los datos en los dos primeros componentes principales.

Visualización: Se genera un gráfico donde cada punto representa un alimento proyectado en el nuevo espacio formado por los componentes principales de SVD.

Almacenamiento de resultados: Finalmente, los datos proyectados se guardan en un DataFrame para facilitar el análisis posterior.
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Suponiendo que tienes tu DataFrame llamado df con los datos de la tabla nutricional
df = tabla_nutricional
# Seleccionamos solo las columnas que contienen los valores nutricionales
nutritional_values = df.iloc[:, 2:].values

# Normalizamos los datos para que tengan media 0 y desviación estándar 1
scaler = StandardScaler()
nutritional_values_scaled = scaler.fit_transform(nutritional_values)

# Aplicamos la Descomposición de Valores Singulares (SVD)
U, s, VT = np.linalg.svd(nutritional_values_scaled)

# Tomamos las dos primeras columnas de VT como los componentes principales
components = VT[:2, :]

# Graficamos los resultados
plt.figure(figsize=(10, 8))
plt.scatter(components[0], components[1], alpha=0.5)
plt.title('Análisis en Componentes Principales con SVD')
plt.xlabel('Primer Componente Principal')
plt.ylabel('Segundo Componente Principal')

# Anotamos el nombre de los alimentos
# for i, txt in enumerate(df['Alimento']):
#     plt.annotate(txt, (components[0][i], components[1][i]))

plt.grid(True)
plt.show()

#%%
####################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Suponiendo que tienes tu DataFrame llamado df con los datos de la tabla nutricional

# Seleccionamos solo las columnas que contienen los valores nutricionales
nutritional_values = df.iloc[:, 2:].values

# Normalizamos los datos para que tengan media 0 y desviación estándar 1
scaler = StandardScaler()
nutritional_values_scaled = scaler.fit_transform(nutritional_values)

# Aplicamos la Descomposición de Valores Singulares (SVD)
U, s, VT = np.linalg.svd(nutritional_values_scaled)

# Tomamos las dos primeras columnas de VT como los componentes principales
components = VT[:2, :]

# Graficamos los resultados
plt.figure(figsize=(10, 8))
plt.scatter(components[0], components[1], alpha=0.5)
plt.title('Análisis en Componentes Principales con SVD')
plt.xlabel('Primer Componente Principal')
plt.ylabel('Segundo Componente Principal')

# Anotamos el nombre de los alimentos
for i, txt in enumerate(df['Alimento']):
    plt.annotate(txt, (components[0][i], components[1][i]))

plt.grid(True)
plt.show()
#%%
###################################
#    MI PCA
##################################

#MI matriz va a ser tabla nutricional

#transformo el df en una matriz

tabla_nutricional_sin_alimento = tabla_nutricional.drop(columns=['Alimento'])

# Convertir el DataFrame a una matriz NumPy
matriz_nutricional = tabla_nutricional_sin_alimento.values

#ahora calculo la media

# Primero centramos los valores de las imágenes, restando la media. Luego calculamos la matriz de covarianzas:


# Finalmente encontramos los autovalores y autovectores de esta matriz, los cuales ordenamos de mayor a menor valor

def calculoACP(data):
    d, n = data.shape

    media=np.mean(data, axis=1)  #Calcula el vector de medias m de las filas de la matriz

    m_centrada = data - np.tile(media.reshape((len(media), 1)), (1, n))
   
    
   #Resta las medias de las filas de la matriz data. 
   #Esto centra los datos para que tengan media cero en cada dimensión.
  
    Mcov = np.dot(m_centrada,m_centrada.T) / n # Covariance Matrix
# Calcula la matriz de covarianza Mcov de los datos centrados.
#  Esto se hace multiplicando X por su traspuesta y dividiendo por el número de muestras n.
   
    avals, avecs = np.linalg.eigh(Mcov)
    # Calcula los autovalores D y los autovectores V de la matriz de covarianza
    # Mcov utilizando la función eigh de NumPy, 
    # que está optimizada para matrices simétricas y devuelve los autovalores en orden ascendente.

    # ordenamos los autovalores de mayor a menor
    idx = np.argsort (- avals )
    #Ordena los autovalores en orden descendente y devuelve los índices de orden correspondientes.
    avals = avals[idx]
    #Ordena los autovalores y los autovectores de acuerdo con los índices obtenidos en el paso anterior.a
    avecs = avecs[:, idx]

#Devuelve los autovalores ordenados D, los autovectores ordenados V, la matriz de datos centrada X y el vector de medias m
    return avals, avecs, m_centrada, media

D, V, X_cent, m = calculoACP(matriz_nutricional)

#deberia hacer el pca con la matriz centrada
#%%
def proyectar_datos(x,w):
    """
    dao un conjunto de datos x y una amtriz de proyeccion w, devuelve los datos transformados
    como una amtriz
    matriz de proyeccion va a tener como columnas los autovectores de la amtriz de covarianza
    
    """
    return x.dot(w)
from matplotlib import ListedColormap

def calcular_pca(x,n):
    """
    dado un conjunto de datos x y un numero n, devuelve los n autovalores mas grandes
    y sus autovectores asociados
    """
    xtx = matriz de covarianza(X)
    avecs, autovectores
    #autovalores es un arreglo con los n autovalores mas grandes
    #autovectores es una amtriz cuyas columnas son los autovectores asociados a los autovalores
    return avec,autovectores
def pca(x,y,k=5):
    n = 2 #numero de componentes principales
    
    avals,avecs = calcular_pca(x,n)
    x_pca = proyectar_datos(x,avecs)
    
    cmap = ListedColormap(['blue','red','green'])
    plt.scatter(x_pca[:,0],x_pca[:,1],c=y,cmap=cmap,label = 'datos proyectadis')
    plt.show
"""
