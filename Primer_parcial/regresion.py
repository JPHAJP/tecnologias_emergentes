import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#normalizar datos con sklearn
from sklearn.preprocessing import StandardScaler

from scipy import stats

# Importar los datos
file_path = 'Primer_parcial/data.csv'
data = pd.read_csv(file_path)

print(data.head())

# Borrar las columnas que no se van a utilizar
data_cleaned = data.drop(columns=['country', 'street', 'date', 'statezip', 'city', 'waterfront'])

# Verificar si hay datos nulos
print('Datos nulos o faltantes:')
print(data_cleaned.isnull().sum())

# Eliminar datos nulos o faltantes
data_cleaned = data_cleaned.dropna()

# Verificar si hay datos duplicados
print('Datos duplicados:')
print(data_cleaned.duplicated().sum())

# Eliminar datos duplicados
data_cleaned = data_cleaned.drop_duplicates()

# Verificar si hay valores atípicos
z_scores = np.abs(stats.zscore(data_cleaned))
print('Valores atípicos:')
print(z_scores)

# Eliminar los valores atípicos
data_cleaned = data_cleaned[(z_scores < 3).all(axis=1)]

# Verificar cuantos datos se eliminaron
print(f'Número de datos eliminados: {len(data) - len(data_cleaned)} de {len(data)}')

# Obtener la matriz de correlación respecto al precio
correlation_matrix = data_cleaned.corr()

# Graficar la matriz de correlación
plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Matriz de correlación')
plt.show()

price_correlation = correlation_matrix["price"].sort_values(ascending=False)
print('Correlación con el precio:')
print(price_correlation)

#Obtener las columnas con mayor correlación
print('Columnas con mayor correlación:')
print(price_correlation[price_correlation > 0.20])

# Eliminar las columnas con baja correlación (menor a .20)
data_cleaned = data_cleaned.drop(columns=['yr_renovated','yr_built','condition','sqft_lot','floors'])

# Seleccionar la variable predictora (mayor correlación sqft_living)  y la variable a predecir (precio)
#X = data_cleaned['sqft_living'].values.reshape(-1,1) #no funciono
X = data_cleaned.drop(columns=['price'])
y = data_cleaned['price']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot de los datos
# plt.scatter(X_train, y_train, color='blue')
# plt.xlabel('Pies cuadrados')
# plt.ylabel('Precio')
# plt.title('Precio vs pies cuadrados')
# plt.show()


# Crear el modelo de regresión tipo Lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

print('Error cuadrático medio:', mse)
print('Squared root of MSE:', np.sqrt(mse))

# Graficar los resultados con linea de tendencia
plt.scatter(y_test, y_pred)
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Precio real')
plt.ylabel('Precio predicho')
plt.title('Precio real vs precio predicho')
plt.show()