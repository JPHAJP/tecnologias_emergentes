import matplotlib.pyplot as plt
import matplotlib.image as mimg
import cv2
#import numpy as np

# Leer la imagen
img = mimg.imread('image_process/lena.tif')
img_a = img.copy()

# Revisar si la imagen es rgb o bgr

# Mostrar la imagen
#plt.imshow(img)
#plt.show()

# Leer con OpenCV
#img = cv2.imread('image_process/lena.tif')

# Mostrar la imagen
# cv2.namedWindow('Lena', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('Lena', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

m, n, c = img_a.shape

# Imprimir linea por linea de la imagen
# for i in range(m):
#     for j in range(n):
#         print(img[i, j], end='')
#     print()


# COPIA DE img Para evitar modificar la imagen original "img_a"
# Manipular el canal rojo de color
# Bajar la intensidad del canal rojo
for i in range(m):
    for j in range(n):
        img_a[i, j, 0] = img[i, j, 0] * 0.5



# Crear la figura con subplots
plt.figure(figsize=(50, 20))

# Primer subplot: Imagen original
plt.subplot(2, 2, 1)  # 2 filas, 2 columnas, posición 1
plt.imshow(img)
plt.title('Imagen Original')
plt.axis('off')  # Ocultar los ejes

# Segundo subplot: Histograma de colores
plt.subplot(2, 2, 2)  # 2 filas, 2 columnas, posición 2
color = ('r', 'g', 'b')  # Colores para el histograma
plt.title('Histograma de colores')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])  # Limitar el rango del eje X

# Tercer subplot: Imagen en escala de grises
plt.subplot(2, 2, 3)  # 2 filas, 1 columna, posición 3 (abajo)
plt.imshow(img_a)
plt.title('Imagen modificada')
plt.axis('off')  # Ocultar los ejes

# Cuarto subplot: Histograma de colores
plt.subplot(2, 2, 4)  # 2 filas, 2 columnas, posición 4
color = ('r', 'g', 'b')  # Colores para el histograma
plt.title('Histograma de colores')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')
for i, col in enumerate(color):
    histr = cv2.calcHist([img_a], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])  # Limitar el rango del eje X


# Mostrar la figura completa
plt.tight_layout()
plt.show()



