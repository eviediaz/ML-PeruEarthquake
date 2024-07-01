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

#SVM
#Cargando el modelo y ajustandolo con datos de entrenamiento
# Seleccionar un subconjunto de los datos de entrenamiento
subset_size = 500
X_train_subset = X_train[:subset_size]
y_train_subset = y_train[:subset_size]

# Crear un modelo SVM
svm = SVR(kernel='rbf', C=1e3, gamma=0.1)

# Entrenar el modelo SVM en el subconjunto de datos
svm.fit(X_train_subset, y_train_subset)

# Evaluar el modelo en el conjunto de prueba
score = svm.score(X_test, y_test)
print("Resultado de la prueba:", score)

# Predecir en el conjunto de prueba con el modelo SVM
y_pred_svm = svm.predict(X_test)

# Calcular R² y MSE
r2_svm = r2_score(y_test, y_pred_svm)
mse_svm = mean_squared_error(y_test, y_pred_svm)

# Añadir las métricas al diccionario 'scores'
scores['mse'].append(mse_svm)
scores['R^2'].append(r2_svm)

# Imprimir las métricas
print("SVM R^2: {:.2f}, MSE: {:.2f}".format(r2_svm, mse_svm))

#Random Forest
#Cargando el modelo y ajustándolo con datos de entrenamiento.
from sklearn.ensemble import RandomForestRegressor

# Crear una instancia del modelo RandomForestRegressor
rf_model = RandomForestRegressor()

# Entrenar el modelo
rf_model.fit(X_train, y_train)

#Predecir los datos de prueba y evaluarlos.
#Encuentre los valores predichos y evalúelos usando métricas como MSE, r2
from sklearn.metrics import mean_squared_error, r2_score

# Predecir sobre datos de prueba
y_pred = rf_model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

scores['mse'].append(mse)
scores['R^2'].append(r2)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

#Gráfico de dispersión
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='blue')
plt.title('Scatter Plot de Valores Reales vs. Predicciones')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.show()

#Importancia de la característica
#Este gráfico muestra la importancia de cada característica en el modelo. Puede crear un gráfico de importancia de características utilizando el atributo feature_importances_ del modelo de bosque aleatorio.
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances, align='center')
plt.xlabel('Importancia de Características')
plt.ylabel('Características')
plt.title('Importancia de Características - Random Forest')
plt.show()

#Residuos del modelo
#Un gráfico residual muestra la diferencia entre los valores reales y los valores predichos. Puede crear un gráfico residual utilizando la función residplot() de la biblioteca seaborn.
residuals = y_test - y_pred

plt.scatter(y_pred, residuals, color='green')
plt.title('Residual Plot')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.axhline(y=0, color='black', linestyle='--')
plt.show()