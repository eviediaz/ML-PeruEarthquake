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

df['LATITUD'] = df['LATITUD'].str.replace(',', '.').astype(float)
df['LONGITUD'] = df['LONGITUD'].str.replace(',', '.').astype(float)
df['MAGNITUD'] = df['MAGNITUD'].str.replace(',', '.').astype(float)

# Verificar que los datos ahora estén en formato numérico
print(df.dtypes)