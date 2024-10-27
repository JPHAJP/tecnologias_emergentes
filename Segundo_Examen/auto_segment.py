import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('Segundo_Examen/placa_1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB para matplotlib

# Convertir a espacio de color HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir el rango de color para segmentar la placa (ajustar según la imagen)
lower_color = np.array([0, 0, 120])  # Ajuste para detectar colores claros
upper_color = np.array([180, 50, 255])  # Ajuste máximo para colores blancos

# Aplicar la máscara para extraer la placa
mask = cv2.inRange(hsv, lower_color, upper_color)
segmented = cv2.bitwise_and(image, image, mask=mask)

# Convertir la imagen segmentada a escala de grises
gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

# Aplicar un desenfoque Gaussiano para reducir el ruido
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detectar bordes usando el algoritmo de Canny
edges = cv2.Canny(blur, 50, 150)

# Aplicar la transformada de Hough para detectar líneas rectas
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, 
                        minLineLength=50, maxLineGap=10)

# Dibujar las líneas detectadas sobre la imagen original
image_lines = image_rgb.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Crear una figura y mostrar cada paso del procesamiento
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Segmentación de Placas Paso a Paso')

# Mostrar la imagen original
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Imagen Original')
axes[0, 0].axis('off')

# Mostrar la imagen segmentada por color
axes[0, 1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Segmentación por Color')
axes[0, 1].axis('off')

# Mostrar la imagen en escala de grises
axes[0, 2].imshow(gray, cmap='gray')
axes[0, 2].set_title('Escala de Grises')
axes[0, 2].axis('off')

# Mostrar la imagen con desenfoque Gaussiano
axes[1, 0].imshow(blur, cmap='gray')
axes[1, 0].set_title('Desenfoque Gaussiano')
axes[1, 0].axis('off')

# Mostrar los bordes detectados
axes[1, 1].imshow(edges, cmap='gray')
axes[1, 1].set_title('Bordes Detectados')
axes[1, 1].axis('off')

# Mostrar la imagen con las líneas detectadas
axes[1, 2].imshow(image_lines)
axes[1, 2].set_title('Líneas Detectadas (Hough)')
axes[1, 2].axis('off')

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()
