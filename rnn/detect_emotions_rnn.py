import cv2
import dlib
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque

print("Завантаження моделей...")
MODEL_PATH = "../Models/emotion_cnnlstm_model.keras"
detector = dlib.get_frontal_face_detector()

try:
    emotion_model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Не вдалося завантажити модель з файлу '{MODEL_PATH}'. Переконайтеся, що навчання було проведено.")
    exit()

IMG_WIDTH, IMG_HEIGHT = 48, 48
SEQUENCE_LENGTH = 10 
EMOTION_LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

print("Запуск веб-камери... Натисніть 'q' для виходу.")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Помилка: не вдалося відкрити веб-камеру.")
    exit()

prev_time = 0
last_prediction = "Neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    roi_to_process = frame
    if faces:
        face_rect = faces[0]
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        roi_to_process = frame[y:y+h, x:x+w]

    if roi_to_process.size > 0:
        roi_resized = cv2.resize(roi_to_process, (IMG_WIDTH, IMG_HEIGHT))
        roi_processed = (roi_resized.astype("float") / 255.0)
        frame_buffer.append(roi_processed)
    
    if len(frame_buffer) == SEQUENCE_LENGTH:
        if faces:
            sequence_to_predict = np.expand_dims(np.array(frame_buffer), axis=0)
            
            predictions = emotion_model.predict(sequence_to_predict, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            last_prediction = EMOTION_LABELS[predicted_class_index]
            
            print(f"RNN Raw detection: {last_prediction}")

    if faces:
        face_rect = faces[0]
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, last_prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    curr_time = time.time()
    if prev_time > 0:
        fps = 1 / (curr_time - prev_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 0), 2)
    prev_time = curr_time

    cv2.imshow("Emotion Recognition (CNN-LSTM) - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()