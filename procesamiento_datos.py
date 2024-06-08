#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocesamiento de los datos y algunas notas de índole nutricional
"""
import pandas as pd

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
    tabla_nutricional.columns = tabla_nutricional.columns.str.replace('mg', 'g', regex=True)
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
    sodio = data['Na (g)'].sum()
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

carpeta =''

#Creamos dataframes con los datos en formato csv
tabla_nutricional = pd.read_csv(carpeta +  'tabla_nutricional.csv',delimiter= ';')
consumidores_libres = pd.read_csv(carpeta + 'consumidores_libres.csv' ,delimiter = ';')

#Creamos un dataframe de los márgenes de las metas de nutrientes por la OMS (2003)
margenes_oms = pd.DataFrame({
    'Factor alimentario': [
        'Grasas totales', 'Ácidos grasos saturados', 'Ácidos grasos poliinsaturados (AGPI)',
        'Ácidos grasos poliinsaturados n-6', 'Ácidos grasos poliinsaturados n-3', 'Ácidos grasos trans',
        'Ácidos grasos monoinsaturados', 'Carbohidratos totales', 'Azúcares libres', 'Proteínas',
        'Colesterol', 'Sodio', 'Frutas y verduras', 'Fibra alimentaria total', 'Polisacáridos no amiláceos (PNA)'
    ],
    'Meta (% de la energía total)': [
        '15-30 %', '<10 %', '6-10 %', '5-8 %', '1-2 %', '<1 %', 'Por diferencia', 
        '55-75 %', '<10 %', '10-15 %', '<300 mg/día', '2000 mg/día', '>= 400 g/día', '>25 g/día', 
        '20 g'
    ]
})

#Normalizamos la tabla
tabla_nutricional = normalizar_tabla_nutricional(tabla_nutricional)

#Verificamos si la CBA cumple los márgenes de la OMS
print(evaluarCumplimiento_dieta_margenes(tabla_nutricional)) 

