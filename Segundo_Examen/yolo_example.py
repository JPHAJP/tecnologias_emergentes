import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('Autosegment/license_plate_detector.pt')  

# Cargar la imagen
image_path = "Segundo_Examen/placa_1.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB para Matplotlib

# Realizar la detección
results = model(image)

# Iterar sobre los resultados para extraer las coordenadas de cada objeto detectado
for result in results:
    boxes = result.boxes  # Coordenadas de las cajas detectadas
    for box in boxes:
        # Extraer coordenadas (x1, y1) y (x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Recortar la región de la imagen correspondiente a la detección
        plate_image = image_rgb[y1:y2, x1:x2]

        # Mostrar la imagen recortada usando Matplotlib
        plt.figure(figsize=(6, 4))
        plt.imshow(plate_image)
        plt.title('Placa Detectada y Recortada')
        plt.axis('off')  # Ocultar ejes
        plt.savefig('license_plate.jpg')

# Anotar las detecciones sobre la imagen original
annotated_image = results[0].plot()

# Mostrar la imagen original con las anotaciones usando Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(annotated_image)
plt.title('Detección de Placas con YOLOv8')
plt.axis('off')  # Ocultar ejes
plt.savefig('detection_result.jpg')  # Guardar la imagen con las anotaciones
