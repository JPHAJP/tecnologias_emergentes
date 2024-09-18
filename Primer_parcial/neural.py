import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf

from scipy import stats

# Importar los datos
file_path = 'Primer_parcial/data.csv'
data = pd.read_csv(file_path)

print(data.head())

# Borrar las columnas que no se van a utilizar
data_cleaned = data.drop(columns=['country', 'street', 'date', 'statezip','waterfront', 'city'])

# Verificar si hay datos nulos
print('Datos nulos o faltantes:')
print(data_cleaned.isnull().sum())

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
#plt.show()

price_correlation = correlation_matrix["price"].sort_values(ascending=False)
print('Correlación con el precio:')
print(price_correlation)

# Eliminar las columnas con baja correlación (menor a .20)
data_cleaned = data_cleaned.drop(columns=['yr_renovated','yr_built','condition','sqft_lot','floors'])

# Volver a imprimir head de los datos
print(data_cleaned.head())

# Seleccionar las variables predictoras
X = data_cleaned.drop(columns=['price'])
y = data_cleaned['price']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Crear una arquitecura de red neuronal
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32, input_shape=[len(X.columns)]),
    tf.keras.layers.Dense(units=64),
    tf.keras.layers.Dense(units=32),
    tf.keras.layers.Dense(units=16),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(.65), loss='mse')

model_history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

plt.figure(figsize=(10,6))
plt.plot(model_history.history['loss'], label='Train Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#Save figure
plt.savefig('loss.png')