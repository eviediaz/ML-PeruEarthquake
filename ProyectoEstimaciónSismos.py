df['LATITUD'] = df['LATITUD'].str.replace(',', '.').astype(float)
df['LONGITUD'] = df['LONGITUD'].str.replace(',', '.').astype(float)
df['MAGNITUD'] = df['MAGNITUD'].str.replace(',', '.').astype(float)

# Verificar que los datos ahora estén en formato numérico
print(df.dtypes)