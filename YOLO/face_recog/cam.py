import cv2
from ultralytics import YOLO

# Cargar el modelo entrenado, cambia 'ruta_al_modelo.pt' por la ruta de tu archivo de modelo
model = YOLO('face_recog/train/weights/best.pt')

# Iniciar la captura de video de la cámara (0 para la cámara principal)
cap = cv2.VideoCapture(0)

# Verifica si la cámara está abierta
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Bucle para capturar los cuadros de la cámara en tiempo real
while True:
    # Leer el frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la predicción con YOLOv8 en el frame
    results = model(frame)

    # Dibujar las predicciones directamente en el frame original
    for box in results[0].boxes:
        # Extraer las coordenadas, clase y confianza de la detección
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordenadas del bounding box
        class_id = int(box.cls[0])  # ID de la clase detectada
        score = box.conf[0]  # confianza del modelo para la detección

        # Obtener el nombre de la clase (si está disponible en el modelo)
        label = model.names[class_id] if model.names else f"ID {class_id}"

        # Dibujar el bounding box y la etiqueta en el frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame con anotaciones en una sola ventana
    cv2.imshow("YOLOv8 - Detección en Tiempo Real", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()