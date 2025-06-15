import cv2
import dlib
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

print("Завантаження моделей...")

MODEL_PATH = "Models/emotion_vgg16_model.keras"

try:
    detector = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"Помилка завантаження детектора dlib: {e}")
    exit()

try:
    emotion_model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Не вдалося завантажити модель з файлу '{MODEL_PATH}'.")
    print("Переконайтеся, що ви спочатку запустили скрипт навчання та модель було успішно збережено.")
    exit()

IMG_WIDTH, IMG_HEIGHT = 48, 48
EMOTION_LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

print("Запуск веб-камери... Натисніть 'q' для виходу.")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Помилка: не вдалося відкрити веб-камеру.")
    exit()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face_rect in faces:
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        
        face_roi = frame[y:y+h, x:x+w]

        if face_roi.size > 0:
            roi_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
            roi_processed = roi_resized.astype("float") / 255.0
            roi_processed = img_to_array(roi_processed)
            roi_processed = np.expand_dims(roi_processed, axis=0)
            
            predictions = emotion_model.predict(roi_processed, verbose=0)
            
            predicted_class_index = np.argmax(predictions[0])
            predicted_emotion = EMOTION_LABELS[predicted_class_index]
            confidence = np.max(predictions[0])

            print(f"Raw detection: {predicted_emotion}, Confidence: {confidence:.2f}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text_to_show = f"{predicted_emotion} ({confidence:.2f})"
            cv2.putText(frame, text_to_show, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    curr_time = time.time()
    if prev_time > 0:
        fps = 1 / (curr_time - prev_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2)
    prev_time = curr_time

    cv2.imshow("Emotion Recognition - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()