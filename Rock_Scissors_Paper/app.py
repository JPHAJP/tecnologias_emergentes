import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Definir los directorios de entrenamiento, validación y prueba
Train_dir = 'Rock_Scissors_Paper/images/train'
Validation_dir = 'Rock_Scissors_Paper/images/train'
Test_dir = 'Rock_Scissors_Paper/images/test'

# Cargar los datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(Train_dir, 
                                             shuffle=True, 
                                             batch_size=32, 
                                             image_size=(150, 150))

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(Validation_dir,
                                             shuffle=True,
                                             batch_size=32,
                                             image_size=(150, 150))

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(Test_dir,
                                             shuffle=True,
                                             batch_size=32,
                                             image_size=(150, 150))

# Imprimir las clases
class_names = train_dataset.class_names
print(class_names)

# Mostrar algunas imágenes del conjunto de entrenamiento
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
plt.show()

# Definir el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
]) 

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Imprimir el resumen del modelo
model.summary()

# Definir EarlyStopping para evitar sobreentrenamiento
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Entrenar el modelo
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10, callbacks=[early_stop])

# Graficar la precisión del modelo durante el entrenamiento
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('\nTest accuracy:', test_acc)

# Guardar el modelo
model.save('Rock_Scissors_Paper/model_1.h5')