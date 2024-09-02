import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos desde un archivo CSV
df = pd.read_csv('global_air/global_air.csv')

# Mostrar las primeras filas
print(df.head())

# Mostrar la forma del conjunto de datos
print(df.shape)

# Mostrar las columnas
print(df.columns)

# Mostrar los tipos de datos
print(df.dtypes)

# Mostrar el número de valores nulos para cada columna
print(df.isnull().sum())

# Contar el número de ocurrencias por país
grupos = df.groupby('Country').size()
print(grupos)

# Convertir las columnas 'Country' y 'City' a variables dummy (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['Country', 'City'])

# Seleccionar las características y la variable objetivo
X = df_encoded.drop(['PM2.5', 'Date'], axis=1)  # Usar todas las columnas excepto la columna objetivo y 'Date'
y = df_encoded['PM2.5'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos de entrenamiento
modelo.fit(X_train, y_train)

# Realizar predicciones con el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse}')

# Visualizar algunas predicciones
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs Predicciones')
plt.show()