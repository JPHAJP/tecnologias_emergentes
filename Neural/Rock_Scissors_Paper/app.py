import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Definir los directorios de entrenamiento, validación y prueba
Train_dir = 'Rock_Scissors_Paper/images/train'
#Validation_dir = 'Rock_Scissors_Paper/images/validation'
Test_dir = 'Rock_Scissors_Paper/images/test'

# Cargar los datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(Train_dir, 
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

train_counts = np.bincount([label for batch in train_dataset for label in batch[1].numpy()])
print(f"Distribución de clases en entrenamiento: {train_counts}")

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
    tf.keras.layers.Dense(3, activation='softmax', name='output')
]) 

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Imprimir el resumen del modelo
model.summary()

# Definir EarlyStopping para evitar sobreentrenamiento
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Entrenar el modelo
model_history = model.fit(train_dataset, validation_data=test_dataset, epochs=10, callbacks=[early_stop])

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Acc: {test_acc}')

# Mostrar la gráfica de accuracy y pérdida
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(model_history.history['accuracy'], label='Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.ylabel('Acc')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(model_history.history['loss'], label='Loss')
plt.plot(model_history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('\nTest accuracy:', test_acc)

# Obtener las etiquetas verdaderas y predichas
y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())  # Etiquetas verdaderas
    y_pred.extend(np.argmax(predictions, axis=1))  # Etiquetas predichas

# Calcular la matriz de confusión
conf_mat = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:")
print(conf_mat)

# Calcular el reporte de clasificación (incluye precisión por clase)
class_report = classification_report(y_true, y_pred, target_names=class_names)
print("Reporte de clasificación por clase:")
print(class_report)

# Extraer la precisión (accuracy) por clase del reporte de clasificación
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
accuracy_per_class = [report_dict[cls]['precision'] for cls in class_names]

# Graficar el accuracy por clase
plt.figure(figsize=(8, 6))
sns.barplot(x=class_names, y=accuracy_per_class, palette="Blues_d")
plt.title("Accuracy por clase")
plt.xlabel("Clases")
plt.ylabel("Precisión (Accuracy)")
plt.show()

# Guardar el modelo
model.save('Rock_Scissors_Paper/model_1.h5')