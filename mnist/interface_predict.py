import tensorflow as tf
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

# Cargar el modelo
model = tf.keras.models.load_model('mnist/neural.h5')

# Inicializar una imagen en blanco para dibujar
canvas_image = Image.new("L", (280, 280), 255)  # Cambiar a 280x280 para el lienzo
draw = ImageDraw.Draw(canvas_image)

# Función para crear canvas para dibujar 
def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)  # Grosor del pincel reducido
    x2, y2 = (event.x + 5), (event.y + 5)
    cv.create_oval(x1, y1, x2, y2, fill='black', width=5)  # Grosor del pincel en el canvas
    draw.ellipse([x1, y1, x2, y2], fill='black')

def clear():
    cv.delete('all')
    global canvas_image, draw
    canvas_image = Image.new("L", (280, 280), 255)  # Reset canvas image
    draw = ImageDraw.Draw(canvas_image)  # Reiniciar el objeto de dibujo
    prediction_label.config(text="Predicción: ")  # Limpiar la predicción mostrada

def save_image():
    # Redimensionar la imagen del lienzo a 28x28 para guardar
    resized_image = canvas_image.resize((28, 28))
    file_path = "mnist/imagen_28x28.png"  # Nombre del archivo a guardar
    resized_image.save(file_path)
    print(f"Imagen guardada como {file_path}")

def predict_image():
    # Obtener la imagen del lienzo (canvas)
    image = canvas_image.resize((28, 28))  # Redimensionar a 28x28

    # Invertir la imagen (blanco a negro y viceversa)
    image = ImageOps.invert(image)

    # Convertir la imagen a un array numpy
    image_array = np.array(image)

    # Normalizar la imagen
    image_array = image_array / 255.0

    # Asegurarse de que tenga 1 canal y añadir dimensión de lote
    image_array = np.expand_dims(image_array, axis=-1)  # Añadir el canal
    image_array = np.expand_dims(image_array, axis=0)  # Añadir la dimensión de lote

    # Asegurarse de que la imagen sea del tipo float32
    image_array = tf.convert_to_tensor(image_array, dtype=tf.float32)

    # Verificar la forma de la imagen antes de predecir
    print(f"Forma de la imagen antes de predecir: {image_array.shape}")

    # Hacer la predicción
    prediction = model.predict(image_array)
    predicted_digit = prediction[0].argmax()

    # Mostrar la imagen invertida y la predicción
    plt.imshow(image_array[0, :, :, 0], cmap='gray')  # Mostrar la imagen en escala de grises
    plt.axis('off')
    plt.title(f'Predicción: {predicted_digit}')
    plt.show()

    # Mostrar la predicción en la etiqueta
    prediction_label.config(text=f'Predicción: {predicted_digit}')  # Mostrar la predicción en la etiqueta

# Configurar la GUI de Tkinter
root = Tk()
root.title('Predicción de imagen')
root.geometry('400x500')  # Ajustar altura para incluir más botones

# Crear un lienzo para dibujar
cv = Canvas(root, bg='white', width=280, height=280)
cv.pack(pady=20)

# Vincular eventos del ratón al lienzo
cv.bind("<B1-Motion>", paint)  # Dibujar mientras se mantiene presionado el botón izquierdo del ratón
cv.bind("<Button-3>", clear)    # Limpiar el lienzo con el botón derecho del ratón

# Botón para predecir la imagen dibujada
button_predict = Button(root, text='Predecir imagen', command=predict_image)
button_predict.pack()

# Botón para limpiar el lienzo
button_clear = Button(root, text='Limpiar', command=clear)
button_clear.pack()

# Botón para guardar la imagen
button_save = Button(root, text='Guardar imagen', command=save_image)
button_save.pack()

# Etiqueta para mostrar la predicción
prediction_label = Label(root, text='Predicción: ')
prediction_label.pack(pady=10)

root.mainloop()