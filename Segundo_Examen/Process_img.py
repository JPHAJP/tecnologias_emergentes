#Importar una imagen de una placa de un coche para procesarla y recortar la placa y sus caracteres.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_hub as hub

def recortar_imagen(img, x1, y1, x2, y2):
    img_a = img.copy()
    img_a = img_a[y1:y2, x1:x2]
    return img_a

# Funcion para cambiar a escala de grises
def gris(img):
    img_a = img.copy()
    m, n, c = img_a.shape
    for i in range(m):
        for j in range(n):
            pixel = img[i, j]
            promedio = (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) // 3
            img_a[i, j] = promedio, promedio, promedio
    return img_a

def binarizar_imagen(img, canal, umbral):
    img_a = img.copy()
    m, n, c = img_a.shape
    for i in range(m):
        for j in range(n):
            pixel = img[i, j]
            if pixel[canal] > umbral:
                img_a[i, j] = 255, 255, 255
            else:
                img_a[i, j] = 0, 0, 0
    return img_a

def filtro_mediana(img):
    return cv2.medianBlur(img, 5)

def invertir_imagen(img):
    return cv2.bitwise_not(img)

# Función para redimensionar y rellenar caracteres al mismo tamaño
def ajustar_caracteres(chars, target_size=(20, 20)):
    chars_a = []
    for char in chars:
        # Redimensionar al tamaño objetivo, manteniendo la proporción
        char_resized = cv2.resize(char, target_size, interpolation=cv2.INTER_AREA)
        chars_a.append(char_resized)
    return chars_a

#Función para mostrar los caracteres de la placa en un solo plot con 6 subplots
def mostrar_caracteres(chars):
    fig, axs = plt.subplots(1, len(chars), figsize=(15, 5))
    for i, char in enumerate(chars):
        axs[i].imshow(char, cmap='gray')
        #axs[i].axis('off')
    plt.show()

# Función para agregar un contorno negro de 8px a los caracteres
def contorno(chars, border_size=8):
    chars_a = []
    for char in chars:
        # Añadir un borde negro de 8 píxeles alrededor de la imagen del carácter
        char_with_border = cv2.copyMakeBorder(char, border_size, border_size, border_size, border_size, 
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
        chars_a.append(char_with_border)
    return chars_a

        



# Cargar la imagen de la placa con matplolib
img_1 = plt.imread('Segundo_Examen/placa_1.jpg')
plt.imshow(img_1)
plt.axis('off')
plt.show()

# Cargar la placa 2
img_2 = plt.imread('Segundo_Examen/placa_2.webp')
plt.imshow(img_2)
plt.axis('off')
plt.show()

# Recortar la región de interés (placa) de la imagen 1
plate_1 = recortar_imagen(img_1, 470, 642, 780, 700)
plate_1 = gris(plate_1)
# Aplicar filtro de mediana para eliminar ruido
plate_1 = filtro_mediana(plate_1)
plate_1 = binarizar_imagen(plate_1, 0, 50)
plate_1 = invertir_imagen(plate_1)
# plt.imshow(plate_1)
# plt.axis('off')
# plt.show()

# Recortar la región de interés (placa) de la imagen 2
plate_2 = recortar_imagen(img_2, 290, 334, 1025, 476)
plate_2 = gris(plate_2)
plate_2 = binarizar_imagen(plate_2, 0, 100)
plate_2 = invertir_imagen(plate_2)
# plt.imshow(plate_2)
# plt.axis('off')
# plt.show()

# Recortar los caracteres de la placa 1 con 6 caracteres
char_1 = plate_1[:, 25:63]
char_2 = plate_1[:, 70:103]
char_3 = plate_1[:, 113:150]
char_4 = plate_1[:, 167:204]
char_5 = plate_1[:, 212:246]
char_6 = plate_1[:, 250:300]

#Guardar lista de caracteres
chars_1 = [char_1, char_2, char_3, char_4, char_5, char_6]

# Recortar los caracteres de la placa 2 con 7 caracteres
char_1 = plate_2[:, 8:87]
char_2 = plate_2[:, 99:175]
char_3 = plate_2[:, 186:265]
char_4 = plate_2[:, 280:352]
char_5 = plate_2[:, 460:535]
char_6 = plate_2[:, 550:625]
char_7 = plate_2[:, 638:712]

#Guardar lista de caracteres
chars_2 = [char_1, char_2, char_3, char_4, char_5, char_6, char_7]

chars_1 = ajustar_caracteres(chars_1)
chars_2 = ajustar_caracteres(chars_2)
chars_1 = contorno(chars_1)
chars_2 = contorno(chars_2)

#Caragar modelo EMNIST para hacer predicción
#model = tf.keras.models.load_model('Segundo_Examen/emnist_model.h5')

# #Hacer predicción de los caracteres de la placa 1
# for char in chars_1:
#     char = np.expand_dims(char, axis=0)
#     char = char / 255.0
#     prediction = model.predict(char)
#     label = np.argmax(prediction)
#     print(chr(label + 65))

# #Hacer predicción de los caracteres de la placa 2
# for char in chars_2:
#     char = np.expand_dims(char, axis=0)
#     char = char / 255.0
#     prediction = model.predict(char)
#     label = np.argmax(prediction)
#     print(chr(label + 65))


mostrar_caracteres(chars_1)
mostrar_caracteres(chars_2)




# # Recortar los caracteres de la placa 2 con 6 caracteres
# char_1 = plate_2[:, 0:50]
# char_2 = plate_2[:, 50:100]

#Función para mostrar los caracteres de la placa
