import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.ensemble import RandomForestRegressor
from scipy import stats

import tensorflow as tf

# Importar los datos
file_path = 'regresion_neuronal/data.csv'
data = pd.read_csv(file_path)

print(data.head())

# Borrar las columnas que no se van a utilizar
data_cleaned = data.drop(columns=['date', 'street', 'statezip', 'country'])

# Verificar si hay datos nulos
print('Datos nulos o faltantes:')
print(data_cleaned.isnull().sum())

# Eliminar datos nulos
data_cleaned = data_cleaned.dropna()

# Creating a new feature 'house_age' and 'renovated' (whether the house was renovated or not)
data_cleaned['house_age'] = 2024 - data_cleaned['yr_built']
data_cleaned['renovated'] = data_cleaned['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

# Drop original year columns since we've created new features
data_cleaned = data_cleaned.drop(columns=['yr_built', 'yr_renovated'])

# Select the columns to scale (most numeric columns)
# scaler = StandardScaler()
# columns_to_scale = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'house_age']
# data_cleaned[columns_to_scale] = scaler.fit_transform(data_cleaned[columns_to_scale])
# print(data_cleaned.head())

# # Codificar variables categóricas
label_encoder = LabelEncoder()
data_cleaned['city'] = label_encoder.fit_transform(data_cleaned['city'])

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
# mostrar la matriz de correlación
print('Matriz de correlación:')
print(correlation_matrix)

# Correlación con el precio
price_correlation = correlation_matrix["price"].sort_values(ascending=False)
print('Correlación con el precio:')
print(price_correlation)

# Print heads
print(data_cleaned.head())

# Variables predictoras (X) y objetivo (y)
X = data_cleaned.drop(columns=['price'])
y = data_cleaned['price']

# One-hot encoding
onehot_encoder = OneHotEncoder()
X = onehot_encoder.fit_transform(X).toarray()

# # Normalización de los datos
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)

# Normalizar
X_train = normalize(X_train, axis=0)
X_val = normalize(X_val, axis=0)
X_test = normalize(X_test, axis=0)

# Revisar cuantas columnas tiene el dataset para la entrada
print(X_train.shape[1])

selected_model=input("Select the model to use: ")
selected_model = 'regresion_neuronal/' + selected_model.lower() + '/model.h5'

# Cargar modelo
model = tf.keras.models.load_model(selected_model)

# Realizar predicciones con el modelo y mostrarlas en grafico 
y_pred = model.predict(X_test)

fig = plt.figure(figsize=(10, 10))
plt.plot(range(len(y_pred)), y_pred, color='blue', label='Precio predicho')
plt.plot(range(len(y_test)), y_test, color='red', alpha=0.5, label='Precio real')
plt.title('Precio Predicho vs Precio Real')
plt.legend()
plt.savefig('regresion_neuronal/predict.png')
plt.show()
