import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('Segundo_Examen/placa_1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB para matplotlib

# Convertir a espacio de color HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir el rango de color para segmentar la placa
lower_color = np.array([0, 0, 120])
upper_color = np.array([180, 50, 255])

# Aplicar la máscara para extraer la placa
mask = cv2.inRange(hsv, lower_color, upper_color)
segmented = cv2.bitwise_and(image, image, mask=mask)

# Convertir la imagen segmentada a escala de grises
gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

# Aplicar un desenfoque para reducir ruido
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detectar bordes usando Canny
edges = cv2.Canny(blur, 50, 150)

# Encontrar contornos en los bordes detectados
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Variables para almacenar la placa detectada
plate_contour = None
max_area = 0

# Obtener las dimensiones de la imagen para usar como referencia
image_height, image_width = gray.shape

# Filtrar los contornos para encontrar el más probable que sea la placa
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    aspect_ratio = w / h

    # Filtrar por área, relación de aspecto, y posición esperada de la placa
    if 2 < aspect_ratio < 5 and area > 500:
        if (0.3 * image_width < x < 0.7 * image_width) and (0.5 * image_height < y < 0.9 * image_height):
            if area > max_area:
                max_area = area
                plate_contour = contour

# Recortar la región de la placa si se encontró un contorno válido
if plate_contour is not None:
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_image = image_rgb[y:y+h, x:x+w]

    # Mostrar la placa recortada
    plt.figure(figsize=(6, 4))
    plt.imshow(plate_image)
    plt.title('Placa Recortada')
    plt.axis('off')
    plt.show()
else:
    print("No se encontró la placa.")

# Mostrar los pasos del procesamiento en una figura
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Segmentación y Detección de Placa Paso a Paso')

# Imagen original
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Imagen Original')
axes[0, 0].axis('off')

# Segmentación por color
axes[0, 1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Segmentación por Color')
axes[0, 1].axis('off')

# Escala de grises
axes[0, 2].imshow(gray, cmap='gray')
axes[0, 2].set_title('Escala de Grises')
axes[0, 2].axis('off')

# Desenfoque Gaussiano
axes[1, 0].imshow(blur, cmap='gray')
axes[1, 0].set_title('Desenfoque Gaussiano')
axes[1, 0].axis('off')

# Bordes detectados
axes[1, 1].imshow(edges, cmap='gray')
axes[1, 1].set_title('Bordes Detectados')
axes[1, 1].axis('off')

# Mostrar placa recortada en el último cuadro si se encontró
if plate_contour is not None:
    axes[1, 2].imshow(plate_image)
    axes[1, 2].set_title('Placa Recortada')
else:
    axes[1, 2].text(0.5, 0.5, 'Placa no encontrada', 
                    ha='center', va='center', fontsize=12)
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
