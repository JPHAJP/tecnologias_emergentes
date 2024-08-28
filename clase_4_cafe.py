import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Datos
tazas_cafe = np.random.randint(1,6,100)
lineas_codigo = 17*tazas_cafe + 50 + np.random.randint(-5,20,100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(tazas_cafe.reshape(-1,1),
                                                    lineas_codigo,
                                                    test_size=0.2,
                                                    random_state=42)

# Dividir datos en entrenamiento y test
modelo = LinearRegression()
modelo.fit(X_train, y_train)
#modelo.fit(tazas_cafe.reshape(-1,1), lineas_codigo)

# Predicción
y_pred = modelo.predict(X_test)

# Error
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse}')

# x = int(input('¿Cuántas tazas de café has tomado hoy? '))
# tazas_cafe_nuevas = np.array([x])
# lineas_codigo_pred = modelo.predict(tazas_cafe_nuevas.reshape(-1,1))
# print(f'Vas a hacer {lineas_codigo_pred} lineas de código')

# Plotear datos en una grafica y en otra linea de regresión
# plt.plot(tazas_cafe, modelo.predict(tazas_cafe.reshape(-1,1)), color='red')
plt.scatter(X_test, y_test, label='Datos de test')
plt.plot(X_test, y_pred, color='red', label='Línea de regresión')
plt.xlabel('Tazas de café')
plt.ylabel('Líneas de código')
plt.show()



