#Cargado de datos
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from matplotlib import style

#Cargar datos Fecha, Hora, Latitud, Longitud, Profundidad
df = pd.read_csv('Catalogo1960_2023.xlsx', encoding='utf8', delimiter=r';')

df.head()

df.info()

#Convertir las columnas a tipo numerico
df['LATITUD'] = df['LATITUD'].str.replace(',', '.').astype(float)
df['LONGITUD'] = df['LONGITUD'].str.replace(',', '.').astype(float)
df['MAGNITUD'] = df['MAGNITUD'].str.replace(',', '.').astype(float)

# Verificar que los datos ahora esten en formato numerico
print(df.dtypes)

#Dividir los datos en datos de entrenamiento y pruebas
# Seleccionar columnas relevantes
X = df[['LATITUD', 'LONGITUD', 'PROFUNDIDAD']]
y = df['MAGNITUD']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrenar el modelo de regresión lineal
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predecir los datos de prueba
#Encuentrar los valores predichos y evaluarlos utilizando métricas de regresión lineal
# Estructura de datos para almacenar los resultados
scores= {"Model name": ["Linear regression", "SVM", "Random Forest"], "mse": [], "R^2": []}

# Predecir en el conjunto de prueba
y_pred = regressor.predict(X_test)

# Calcular R^2 y MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Agregar los resultados a la estructura de datos de puntuaciones
scores['mse'].append(mse)
scores['R^2'].append(r2)

# Imprimir los resultados
print("R^2: {:.2f}, MSE: {:.2f}".format(r2, mse))
