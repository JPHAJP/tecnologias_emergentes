#Run YOLO with my webcam with openCV

import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model.predict(frame)
    for result in results:
        boxes = result.boxes  # Coordenadas de las cajas detectadas
        for box in boxes:
            # Extraer coordenadas (x1, y1) y (x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = model.names[int(class_id)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    # Display the resulting frame on a big window
    frame = cv2.resize(frame, (1280, 920))
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

