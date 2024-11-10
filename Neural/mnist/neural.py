#Analizar el dataset MNIST con una red neuronal (RNA)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Cargar el dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizar los datos
x_train = x_train / 255.0
x_test = x_test / 255.0

# print(x_train[0])
# print(y_train[0])

# Mostrar una imagen en figura 1
# plt.figure(1)
# plt.imshow(x_train[0], cmap='gray')
# plt.title(f'Número: {str(y_train[0])}')
# plt.show()

# Crear el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo con un conjunto de validación
history = model.fit(x_train, y_train, epochs=30, validation_split=0.2)


# Evaluar el modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

#Crear gráfica de accuracy y otra de loss en la misma figura
plt.figure(figsize=(15, 5))
#Accuracy vs Epochs Train and Test
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
#Loss vs Epochs Train and Test
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('mnist/neural.png')
plt.show()

# Guardar el modelo
model.save('mnist/neural.h5')