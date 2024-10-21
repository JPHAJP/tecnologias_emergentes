import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
# Cargar el modelo de piedra, papel o tijera
model = tf.keras.models.load_model('Rock_Scissors_Paper/model_1.h5')

# Cargar el dataset de entrenamiento
Train_dir = 'Rock_Scissors_Paper/images/test'
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    Train_dir, 
    shuffle=True, 
    batch_size=32, 
    image_size=(150, 150)
)

# Imprimir las clases
class_names = train_dataset.class_names
print(f"Clases: {class_names}")

# Seleccionar aleatoriamente 9 imágenes del conjunto de entrenamiento
def select_random_images(dataset, num_images):
    images, labels = [], []
    for img_batch, label_batch in dataset.take(1):  # Tomar un batch de imágenes
        for i in range(num_images):
            idx = random.randint(0, len(img_batch) - 1)  # Seleccionar un índice aleatorio
            images.append(img_batch[idx].numpy())  # Agregar la imagen
            labels.append(label_batch[idx].numpy())  # Agregar la etiqueta correspondiente
    return images, labels

# Obtener 9 imágenes y etiquetas aleatorias del conjunto de entrenamiento
random_images, true_labels = select_random_images(train_dataset, 9)

# Predecir las etiquetas de las imágenes seleccionadas
predicted_labels = []
for img in random_images:
    img_array = np.expand_dims(img, axis=0)  # Añadir una dimensión (batch size 1)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Obtener la clase con mayor probabilidad
    predicted_labels.append(predicted_class[0])

# Mostrar las imágenes junto con las etiquetas verdaderas y las predicciones
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(random_images[i].astype("uint8"))
    true_class_name = class_names[true_labels[i]]
    predicted_class_name = class_names[predicted_labels[i]]
    plt.title(f"Verdadero: {true_class_name}\nPredicción: {predicted_class_name}")
    plt.axis("off")
plt.show()
