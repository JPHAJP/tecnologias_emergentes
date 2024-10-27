import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num = int(input("Train num: "))

# Función para leer el archivo de mapeo y crear un diccionario
def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            index, ascii_value = line.split()
            mapping[int(index)] = chr(int(ascii_value))  # Convertir ASCII a carácter
    return mapping

# Ruta de los archivos (cámbialo a la ruta correspondiente)
train_path = "Segundo_Examen/archive/emnist-balanced-train.csv"
test_path = "Segundo_Examen/archive/emnist-balanced-test.csv"
mapping_path = "Segundo_Examen/archive/emnist-balanced-mapping.txt"

# Cargar los datos
train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)

# Cargar el mapeo de etiquetas
label_mapping = load_mapping(mapping_path)

# Separar características y etiquetas
X_train = train_data.iloc[:, 1:].values  # Características (pixeles)
y_train = train_data.iloc[:, 0].values   # Etiquetas (dígitos)
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Normalizar los datos (de 0-255 a 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Cambiar la forma para que sea compatible con TensorFlow (imagen 28x28 en escala de grises)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convertir las etiquetas a one-hot encoding
num_classes = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# EarlyStopping para evitar sobreentrenamiento
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)


datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Aplicar a los datos de entrenamiento
datagen.fit(X_train)

# Modelo con mayor número de filtros
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compilación del modelo con learning rate scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con data augmentation
model_history = model.fit(datagen.flow(X_train, y_train, batch_size=128), epochs=2,
                          validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])


# Crear dataset de prueba usando tf.data
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128)

# Mostrar la gráfica de accuracy y pérdida y guardarla en un archivo
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
plt.savefig(f'Segundo_Examen/trains/T{num}/accuracy_loss.png')
print("Gráfica de accuracy y pérdida guardada como 'accuracy_loss.png'")

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('\nTest accuracy:', test_acc)

# Obtener las etiquetas verdaderas y predichas
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

# Calcular la matriz de confusión con seaborn y guardarlo en un archivo de imagen
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicciones')
plt.ylabel('Verdaderos')
plt.title('Matriz de confusión')
plt.savefig(f'Segundo_Examen/trains/T{num}/matriz_conf.png')
print("Matriz de confusión guardada como 'matriz_conf.png'")


# Calcular el reporte de clasificación (incluye precisión por clase)
class_names = [label_mapping[i] for i in range(num_classes)]
class_report = classification_report(y_true, y_pred, target_names=class_names)
print("Reporte de clasificación por clase:")
print(class_report)
# Guardar el reporte de clasificación en un archivo de texto
with open(f'Segundo_Examen/trains/T{num}/reporte_clasificacion_emnist.txt', 'w') as f:
    f.write(class_report)

# Extraer la precisión (accuracy) por clase del reporte de clasificación
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
accuracy_per_class = [report_dict[cls]['precision'] for cls in class_names]

# Graficar el accuracy por clase y guardar la imagen
plt.figure(figsize=(8, 6))
sns.barplot(x=class_names, y=accuracy_per_class, palette="Blues_d")
plt.title("Accuracy por clase")
plt.xlabel("Clases")
plt.ylabel("Precisión (Accuracy)")
plt.savefig(f'Segundo_Examen/trains/T{num}/accuracy_por_clase.png')

# Guardar el modelo entrenado en formato .h5
model.save(f'Segundo_Examen/trains/T{num}/modelo_emnist.h5')
print("Modelo guardado como 'modelo_emnist.h5'")