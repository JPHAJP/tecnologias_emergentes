import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

import tensorflow as tf

# Importar los datos
file_path = 'Primer_parcial/data.csv'
data = pd.read_csv(file_path)

print(data.head())

# Borrar las columnas que no se van a utilizar||
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
scaler = StandardScaler()
columns_to_scale = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'house_age']
data_cleaned[columns_to_scale] = scaler.fit_transform(data_cleaned[columns_to_scale])
print(data_cleaned.head())

# Codificar variables categóricas
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

# Correlación con el precio
price_correlation = correlation_matrix["price"].sort_values(ascending=False)
print('Correlación con el precio:')
print(price_correlation)

# Variables predictoras (X) y objetivo (y)
X = data_cleaned.drop(columns=['price'])
y = data_cleaned['price']

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones
y_pred_l = model.predict(X_test)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred_l)
r2 = r2_score(y_test, y_pred_l)
accuracy = model.score(X_test, y_test)
print('Métricas del modelo de regresión lineal:')
print('Error cuadrático medio: ', mse)
print('R2: ', r2)
print('Precisión: ', accuracy)

# Crear y entrenar el modelo de random forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)
accuracy = rf_model.score(X_test, y_test)
print('Métricas del modelo de regresión de RandomForest:')
print('Error cuadrático medio: ', mse)
print('R2: ', r2)
print('Precisión: ', accuracy)

# Crear una arquitecura de red neuronal
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32, input_shape=[len(X.columns)]),
    tf.keras.layers.Dense(units=64),
    tf.keras.layers.Dense(units=32),
    tf.keras.layers.Dense(units=16),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(.05), loss='mse')
model_history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

y_pred_n = model.predict(X_test)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred_n)
r2 = r2_score(y_test, y_pred_n)
print('Métricas del modelo neuronal:')
print('Error cuadrático medio: ', mse)
print('R2: ', r2)
#print('Precisión: ', accuracy)

# Unir graficas en una sola ventana
fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # 1 fila, 3 columnas

# Primera gráfica: Matriz de correlación
sns.heatmap(correlation_matrix, annot=True, ax=axs[0,0])
axs[0,0].set_title('Matriz de correlación')

# Segunda gráfica: Precio real vs precio predicho en Regresión Lineal
axs[0,1].scatter(y_test, y_pred_l)
axs[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[0,1].set_xlabel('Precio real')
axs[0,1].set_ylabel('Precio predicho')
axs[0,1].set_title('Regresión lineal: Precio real vs precio predicho')

# Tercer gráfica: Precio real vs precio predicho en Random Forest
axs[0,2].scatter(y_test, y_pred_l)
axs[0,2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[0,2].set_xlabel('Precio real')
axs[0,2].set_ylabel('Precio predicho')
axs[0,2].set_title('Random Forest: Precio real vs precio predicho')

# Cuarta gráfica: Precio real vs precio predicho en Red Neuronal
axs[1,0].scatter(y_test, y_pred_l)
axs[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axs[1,0].set_xlabel('Precio real')
axs[1,0].set_ylabel('Precio predicho')
axs[1,0].set_title('Red Neuronal: Precio real vs precio predicho')

# Quinta gráfica: Evolución del error en el entrenamiento de la red neuronal
axs[1,1].plot(model_history.history['loss'], label='Train Loss')
axs[1,1].plot(model_history.history['val_loss'], label='Validation Loss')
axs[1,1].set_title('Evolución del error en la red neuronal')
axs[1,1].set_xlabel('Épocas')
axs[1,1].set_ylabel('MSE')
axs[1,1].legend()

# Mostrar todas las gráficas en una ventana
plt.tight_layout()
plt.show()