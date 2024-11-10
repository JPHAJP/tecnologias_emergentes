import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
datos = pd.read_csv('temperaturas/temperatures.csv')

# Split the data into training and test sets
X = datos['Celsius'].values.reshape(-1,1)
y = datos['Fahrenheit'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
# model.add(tf.keras.layers.Dense(units=3))
# model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer=tf.keras.optimizers.Adam(0.6), loss='mse')

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
plt.savefig('temperaturas/loss.png')

