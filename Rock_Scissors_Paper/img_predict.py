import tensorflow as tf
import numpy as np
import cv2
from tkinter import *
from PIL import Image, ImageTk
from time import sleep

# Cargar el modelo de piedra, papel o tijera
model = tf.keras.models.load_model('Rock_Scissors_Paper/model_1.h5')

# Mapeo de clases numéricas a texto
class_names = {0: 'Papel', 1: 'Piedra', 2: 'Tijera'}

# Variable global para almacenar el último frame capturado
frame = None

# Función para actualizar el frame de la cámara en la interfaz
def update_frame():
    global frame  # Declarar frame como variable global
    ret, frame = cap.read()  # Capturar un frame de la cámara
    if ret:
        # Convertir el frame de BGR a RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(10, update_frame)  # Actualizar el frame cada 10 ms

# Función para capturar y predecir la imagen
def predict_image():
    global frame  # Usar el último frame capturado
    
    # Borrar el texto de la predicción antes de realizar la nueva
    prediction_label.config(text='Predicción: Procesando...')
    root.update()  # Actualizar la ventana para mostrar el texto
    #sleep(0.5)  # Esperar medio segundo para dar tiempo a la actualización de la ventana
    
    if frame is not None:
        image_for_model = cv2.resize(frame, (150, 150))  # Redimensionar a 150x150 para el modelo
        image_for_model = cv2.cvtColor(image_for_model, cv2.COLOR_BGR2RGB)  # Convertir a RGB
        image_for_model = np.array(image_for_model) / 255.0  # Normalizar la imagen
        image_for_model = np.expand_dims(image_for_model, axis=0)  # Añadir batch dimension

        # Hacer la predicción con el modelo
        prediction = model.predict(image_for_model)
        print(f"Salidas del modelo: {prediction}")
        predicted_class = prediction[0].argmax()  # Obtener la clase con mayor probabilidad
        confidence = prediction[0].max()  # Obtener la confianza

        # Convertir la clase predicha de número a texto
        predicted_class_text = class_names.get(predicted_class, "Desconocido")

        # Mostrar la nueva predicción en la etiqueta
        prediction_label.config(text=f'Predicción: {predicted_class_text}, Confianza: {confidence * 100:.2f}%')
        print(f'Predicción: {predicted_class_text}, Confianza: {confidence * 100:.2f}%')
    else:
        prediction_label.config(text="Error: No se pudo capturar la imagen.")

# Configurar la GUI de Tkinter
root = Tk()
root.title('Piedra, papel o tijera')
root.geometry('600x600')

# Crear un label para mostrar la cámara
camera_label = Label(root)
camera_label.pack()

# Botón para realizar la predicción
button_predict = Button(root, text='Predecir Imagen', command=predict_image)
button_predict.pack(pady=10)

# Etiqueta para mostrar la predicción
prediction_label = Label(root, text='Predicción: ')
prediction_label.pack(pady=10)

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
update_frame()  # Comenzar la actualización de frames

# Ejecutar la interfaz
root.mainloop()

# Cerrar la cámara al cerrar la ventana
cap.release()
cv2.destroyAllWindows()
