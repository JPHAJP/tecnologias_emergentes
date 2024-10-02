import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

# Cargar el modelo
model = tf.keras.models.load_model('mnist/neural.h5')

# Cargar el dataset para la predicción de ejemplo
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar los datos
x_train = x_train / 255.0
x_test = x_test / 255.0

# Añadir una dimensión de canal a las imágenes (28, 28, 1) para que coincida con lo que el modelo espera
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Predecir 4 imágenes del dataset
predictions = model.predict(x_test)
plt.figure(figsize=(10, 10))
for i in range(4):
    n = np.random.randint(0, x_test.shape[0])
    plt.subplot(2, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[n].reshape(28, 28), cmap='gray')  # Volver a dar forma para mostrar la imagen
    plt.title(f'Número: {y_test[n]}, Predicción: {predictions[n].argmax()}')
plt.show()

# Cargar la imagen real usando PIL y convertir a escala de grises
image = Image.open('mnist/imagen_28x28.png').convert('L')  # Convertir a escala de grises

# Invertir la imagen (blanco a negro y viceversa)
image = ImageOps.invert(image)

# Redimensionar a 28x28 en caso de que no tenga el tamaño adecuado
image = image.resize((28, 28))

# Convertir la imagen a un array numpy
image = np.array(image)

# Normalizar la imagen
image = image / 255.0

# Asegurarse de que tenga 1 canal y añadir dimensión de lote
image = np.expand_dims(image, axis=-1)  # Añadir el canal
image = np.expand_dims(image, axis=0)  # Añadir la dimensión de lote

# Asegurarse de que la imagen sea del tipo float32
image = tf.convert_to_tensor(image, dtype=tf.float32)

# Verificar la forma de la imagen antes de predecir
print(f"Forma de la imagen antes de predecir: {image.shape}")

# Hacer la predicción
prediction = model.predict(image)
predicted_digit = prediction[0].argmax()

# Mostrar la imagen invertida y la predicción
plt.imshow(image[0, :, :, 0], cmap='gray')  # Mostrar la imagen en escala de grises
plt.axis('off')
plt.title(f'Predicción: {predicted_digit}')
plt.show()
