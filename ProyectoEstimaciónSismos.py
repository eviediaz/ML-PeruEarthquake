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

# Entrenar el modelo de regresion lineal
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predecir los datos de prueba
#Encuentrar los valores predichos y evaluarlos utilizando métricas de regresion lineal
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

# Predecir para nuevos datos
# Datos nuevos con solo las caracteristicas relevantes
new_data = [[33.89, -118.40, 16.17], [37.77, -122.42, 8.05]]

# Convertir los nuevos datos a un DataFrame con los mismos nombres de columnas que los datos de entrenamiento
new_data_df = pd.DataFrame(new_data, columns=['LATITUD', 'LONGITUD', 'PROFUNDIDAD'])

# Hacer predicciones con los nuevos datos
new_pred = regressor.predict(new_data_df)
print("Nuevas predicciones:", new_pred)

# Traza la linea de regresion para 'LATITUD'
sns.regplot(x=X_test['LATITUD'], y=y_test, color='blue', scatter_kws={'s': 10})
# Traza la linea de regresion para 'LONGITUD'
sns.regplot(x=X_test['LONGITUD'], y=y_test, color='red', scatter_kws={'s': 10})
# Traza la linea de regresion para 'PROFUNDIDAD'
sns.regplot(x=X_test['PROFUNDIDAD'], y=y_test, color='yellow', scatter_kws={'s': 10})

# Ajustar la leyenda para las variables predictoras
plt.legend(labels=['LATITUD', 'LONGITUD', 'PROFUNDIDAD'])

# Etiquetas de los ejes y titulo del grafico
plt.xlabel('Variables Predictoras')
plt.ylabel('Magnitud')
plt.title('Modelo de Regresion Lineal Multiple')

# Mostrar el grafico
plt.show()