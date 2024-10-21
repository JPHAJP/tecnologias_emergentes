import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# arbo
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el archivo CSV
data = pd.read_csv('Primer_parcial/data.csv')

# Mostrar información del dataset
print(data.head())
print(data.info())

# Mostrar estadísticas del dataset
print(data.describe())

# Eliminar columnas no deseadas
data = data.drop(['date', 'country', 'street'], axis=1)

# Mostrar la distribución de los datos de la columna price
# sns.kdeplot(data['price'], fill=True)
# plt.show()
sns.displot(data['price'], color='blue')
plt.axline((data['price'].mean(), 0), (data['price'].mean(), 0.01), color='red', linestyle='--', linewidth=2)
plt.title("Distribución de precios")
#plt.savefig('distribucion_precios.png')
plt.show()

# Analizemos la columna price
print(data['price'].describe())
prices_zeros = data[data['price'] == 0]
print(prices_zeros)

# Eliminar los precios con valor 0
data = data[data['price'] != 0]

# Variables categóricas
data_categoricas = data.select_dtypes(include=['object'])
print(data_categoricas.head())

# Verificar si todos los datos de statezip empiezan con WA
print(data['statezip'].str.startswith('WA').all())

# Quitar WA de statezip y convertir a entero
data['statezip'] = data['statezip'].str.replace('WA ', '').astype(int)
print(data['statezip'].head())

# Graficar la distribución de las variables categóricas
for col in data_categoricas.columns:
    sns.countplot(x=col, data=data)
    plt.title(f"Distribución de {col}")
    plt.show()

# Convertir las variables categóricas a numéricas
label_encoder = LabelEncoder()
data['city'] = label_encoder.fit_transform(data['city'])
data['statezip'] = label_encoder.fit_transform(data['statezip'])

# Eliminar outliers

# Variables numéricas
data_numericas = data.select_dtypes(include=['int64', 'float64'])
correlacion = data_numericas.corr()
sns.heatmap(correlacion, annot=True, cmap='coolwarm')
plt.title("Matriz de correlación")
#plt.savefig('correlacion.png')
plt.show()

# Selección de las mejores variables numéricas para el modelo
variables_numericas = correlacion['price'].sort_values(ascending=False).index[1:6]
print(variables_numericas)

# Separar las variables independientes y dependiente
# X = data[variables_numericas]
X = data.drop('price', axis=1)
y = data['price']

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
#print("Error cuadrático medio:", mean_squared_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))

# Graficar las predicciones vs los valores reales
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Precio real vs Precio predicho")
#plt.savefig('predicciones.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_pred)), y_pred, color='blue', label='Precio predicho')
plt.plot(range(len(y_test)), y_test, color='red', alpha=0.5, label='Precio real')
plt.title('Precio Predicho vs Precio Real')
plt.legend()
#plt.savefig('predicciones2.png')
plt.show()