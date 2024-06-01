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
from inline_sql import sql, sql_val #mejor no


"""
Preprocesamieto de los datos
Consigna 1.- Leer los datos del archivo tabla nutricional.csv. Completar las
celdas faltantes con ceros. Poner todos los datos en la misma unidad.

Consigna 2.- Desarrollar una funci´on que reciba como par´ametro un DataFrame 
y que eval´ue si las cantidades de los principales elementos de la dieta, 
prote´ınas, carbohidratos, grasas, sodio, fibra, frutas y verduras, cumple los m´argenes
de ingesta de la OMS. Esta funci´on debe devolver VERDADERO si los cumple.
Verificar la Canasta B´asica con esta funci´on.
"""
carpeta = 'C:/Users/Rocio/Desktop/TPs-ALC-1C2024/TP2-ALC/'
tabla_nutricional = pd.read_csv(carpeta +  'tabla_nutricional.csv',delimiter= ';')

# ahora pongo todos los datos en la misma unidad

#=============
#CONSIGNA 1
# Consigna 1.- Leer los datos del archivo tabla nutricional.csv. Completar las
# celdas faltantes con ceros. Poner todos los datos en la misma unidad.

#ponemos todo en minusculas
# nan ponemos 0
# pasamos de g a mg multiplicamos por 1000

tabla_nutricional.iloc[:, 0] = tabla_nutricional.iloc[:, 0].str.lower()

tabla_nutricional.iloc[:,:] = tabla_nutricional.iloc[:,:].fillna(0)

columnas = tabla_nutricional.columns

for columna in columnas:
    if '(gr)' in columna:
        tabla_nutricional[columna]=tabla_nutricional[columna]*1000
        
tabla_nutricional.columns = tabla_nutricional.columns.str.replace('(gr)', '(mg)')

consumidores_libres = pd.read_csv(carpeta + 'consumidores_libres.csv' ,delimiter = ';')

margenes_ingesta_nutrientes = pd.DataFrame({
    'Factor alimentario': [
        'Grasas totales', 'Ácidos grasos saturados', 'Ácidos grasos poliinsaturados (AGPI)',
        'Ácidos grasos poliinsaturados n-6', 'Ácidos grasos poliinsaturados n-3', 'Ácidos grasos trans',
        'Ácidos grasos monoinsaturados', 'Carbohidratos totales', 'Azúcares libres', 'Proteínas',
        'Colesterol', 'Sodio', 'Frutas y verduras', 'Fibra alimentaria total', 'Polisacáridos no amiláceos (PNA)'
    ],
    'Meta (% de la energía total)': [
        '15-30 %', '<10 %', '6-10 %', '5-8 %', '1-2 %', '<1 %', 'Por diferencia', 
        '55-75 %',
        '<10 %', '10-15 %', '<300 mg/día', '200 mg/día', '>= 400 g/día', '>25 g/día', 
        '20 g'
    ]
})
# def dieta_margenes_oms(data):
    
        
