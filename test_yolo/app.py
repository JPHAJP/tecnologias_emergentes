from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('yolov8n.pt')

IMAGE_PATH = 'test_yolo/test.jpg'
img = cv2.imread(IMAGE_PATH)



results = model.predict(img)

#print(results)

for result in results:
    boxes = result.boxes  # Coordenadas de las cajas detectadas
    for box in boxes:
        # Extraer coordenadas (x1, y1) y (x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = box.cls[0]
        class_name = model.names[int(class_id)]

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig('test_yolo/test_predict_1.jpg')



        # Recortar la región de la imagen correspondiente a la detección
        #plate_image = image_rgb[y1:y2, x1:x2]

        # Mostrar la imagen recortada usando Matplotlib
        # plt.figure(figsize=(6, 4))
        # plt.imshow(img)
        # plt.title('Placa Detectada y Recortada')
        # plt.axis('off')  # Ocultar ejes
        #plt.savefig('Segundo_Examen/license_plate.jpg')
