import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.ensemble import RandomForestRegressor
from scipy import stats

import tensorflow as tf

file_save_path = input(f'Especifica el nombre de la version para entrenamiento: ')
file_save_path = 'regresion_neuronal/' + file_save_path.lower().replace(' ', '_')
#Revisar si la carpeta existe
if os.path.exists(file_save_path):
    print(f'La carpeta {file_save_path} ya existe, por favor elige otro nombre')
    exit()
print(f'Tus archivos se guardaran en la carpeta {file_save_path}')

# Crear la carpeta para guardar los archivos
os.makedirs(file_save_path, exist_ok=True)

# Importar los datos
file_path = 'regresion_neuronal/data.csv'
data = pd.read_csv(file_path)

#print(data.head())

# Borrar las columnas que no se van a utilizar
data_cleaned = data.drop(columns=['date', 'street', 'statezip', 'country'])

# Verificar si hay datos nulos
#print('Datos nulos o faltantes:')
#print(data_cleaned.isnull().sum())

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
#print('Matriz de correlación:')
#print(correlation_matrix)

# Correlación con el precio
price_correlation = correlation_matrix["price"].sort_values(ascending=False)
#print('Correlación con el precio:')
#print(price_correlation)

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

# Crear y entrenar el modelo de regresión lineal
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

model.summary()
tf.keras.utils.plot_model(model, to_file=f'{file_save_path}/model.png', show_shapes=True)

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train, y_train, 
                    epochs=500, 
                    batch_size=128, 
                    validation_data=(X_val, y_val), 
                    callbacks=[early_stopping])

# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test)
print('Mean Absolute Error:', mae)

# Graficar la pérdida
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#Crear titulo con el nombre del archivo y el MAE
plt.title(f'Model loss: {file_save_path} - MAE: {mae}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
#Guardar la imagen en la carpeta
plt.savefig(f'{file_save_path}/loss.png')
plt.show()

# Realizar predicciones con el modelo y mostrarlas en grafico 
y_pred = model.predict(X_test)

fig = plt.figure(figsize=(10, 10))
plt.plot(range(len(y_pred)), y_pred, color='blue', label='Precio predicho')
plt.plot(range(len(y_test)), y_test, color='red', alpha=0.5, label='Precio real')
plt.title('Precio Predicho vs Precio Real')
plt.legend()
plt.savefig(f'{file_save_path}/predict.png')
plt.show()


# Guardar el modelo
model.save(f'{file_save_path}/model.h5')