import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

# Importar los datos
file_path = 'Primer_parcial/data.csv'
data = pd.read_csv(file_path)

print(data.head())

# Borrar las columnas que no se van a utilizar
data_cleaned = data.drop(columns=['country', 'date', 'waterfront'])

# Codificar variables categóricas
label_encoder = LabelEncoder()  # Instancia en minúsculas
data_cleaned['city'] = label_encoder.fit_transform(data_cleaned['city'])
data_cleaned['statezip'] = label_encoder.fit_transform(data_cleaned['statezip'])
data_cleaned['street'] = label_encoder.fit_transform(data_cleaned['street'])

# Verificar si hay datos nulos
print('Datos nulos o faltantes:')
print(data_cleaned.isnull().sum())

# Eliminar datos nulos
data_cleaned = data_cleaned.dropna()

# Verificar si hay datos duplicados
print('Datos duplicados:')
print(data_cleaned.duplicated().sum())

# Eliminar datos duplicados
data_cleaned = data_cleaned.drop_duplicates()

# Verificar si hay valores atípicos
z_scores = np.abs(stats.zscore(data_cleaned))
data_cleaned = data_cleaned[(z_scores < 3).all(axis=1)]  # Eliminar outliers

# Obtener la matriz de correlación
correlation_matrix = data_cleaned.corr()

# Graficar la matriz de correlación
plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Matriz de correlación')
plt.show()

# Correlación con el precio
price_correlation = correlation_matrix["price"].sort_values(ascending=False)
print('Correlación con el precio:')
print(price_correlation)

# Eliminar las columnas con baja correlación
data_cleaned = data_cleaned.drop(columns=['yr_renovated','yr_built','statezip'])

# Variables predictoras (X) y objetivo (y)
X = data_cleaned.drop(columns=['price'])
y = data_cleaned['price']

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Error cuadrático medio: ', mse)
print('R2: ', r2)

# Graficar resultados reales (azul) vs predichos (naranja)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Precio real')
plt.ylabel('Precio predicho')
plt.title('Precio real vs precio predicho')
plt.show()