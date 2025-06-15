# app_gui_ultimate.py

# --- 1. ІМПОРТИ ---
import cv2
import dlib
import numpy as np
import time
import psutil
import os
import tkinter as tk
from tkinter import ttk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque, Counter

# --- 2. КОНФІГУРАЦІЯ МОДЕЛЕЙ ---
MODELS = {
    "heuristic": {"type": "dlib_heuristic", "predictor_path": "Models/shape_predictor_68_face_landmarks.dat", "description": "Евристичний аналіз лендмарків Dlib."},
    "vgg16": {"type": "keras_cnn", "model_path": "Models/emotion_vgg16_model.keras", "img_size": (48, 48), "color_mode": "rgb", "labels": {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}, "description": "Згорткова мережа VGG16."},
    "resnet50": {"type": "keras_cnn", "model_path": "Models/emotion_resnet50_model.keras", "img_size": (48, 48), "color_mode": "rgb", "labels": {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}, "description": "Згорткова мережа ResNet50."},
    "rnn": {"type": "keras_rnn", "model_path": "Models/emotion_cnnlstm_model.keras", "img_size": (48, 48), "color_mode": "grayscale", "labels": {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}, "description": "Комбінована CNN+RNN (LSTM) модель."}
}

# --- 3. ПАРАМЕТРИ ДЛЯ РЕЖИМУ АДАПТИВНОГО НАВАНТАЖЕННЯ ---
FPS_ESTIMATE = 20  # Орієнтовна кількість кадрів для розрахунку таймерів
# Параметри вікна прийняття рішень
DECISION_BUFFER_SIZE = 100 # Кількість кадрів для аналізу (наприклад, 5 секунд при 20 FPS)
# Параметри "охолодження" після зміни рівня
COOLDOWN_SECONDS = 15
COOLDOWN_FRAMES = COOLDOWN_SECONDS * FPS_ESTIMATE
# Пороги для прийняття рішень
NEGATIVE_THRESHOLD = 0.25  # Якщо 25%+ кадрів негативні -> знизити навантаження
POSITIVE_THRESHOLD = 0.75  # Якщо 75%+ кадрів позитивні/нейтральні -> підвищити
# Класифікація емоцій для логіки навантаження
POSITIVE_NEUTRAL_EMOTIONS = {"happy", "neutral", "surprise"}
NEGATIVE_STRAIN_EMOTIONS = {"sad", "angry", "fear", "disgust"}


# --- 4. ДОПОМІЖНІ ФУНКЦІЇ ---

def recognize_emotion_heuristic(landmarks):
    # (Код цієї функції залишається без змін, як у попередньому варіанті)
    if landmarks.num_parts != 68: return "Neutral"
    try:
        eye_dist_x = landmarks.part(45).x - landmarks.part(36).x
        if eye_dist_x <= 1: eye_dist_x = 1
        mouth_corners_y_avg = (landmarks.part(48).y + landmarks.part(54).y) / 2
        inner_lips_y_avg = (landmarks.part(62).y + landmarks.part(66).y) / 2
        smile_indicator = (inner_lips_y_avg - mouth_corners_y_avg) / eye_dist_x
        if smile_indicator > 0.12: return "Happy"
        eyebrow_left_y_avg = (landmarks.part(19).y + landmarks.part(20).y) / 2
        eye_top_left_y_avg = (landmarks.part(37).y + landmarks.part(38).y) / 2
        brow_raise_left = (eye_top_left_y_avg - eyebrow_left_y_avg) / eye_dist_x
        if brow_raise_left > 0.25: return "Surprised"
        if smile_indicator < -0.02: return "Sad"
    except Exception: return "Neutral"
    return "Neutral"

def draw_performance_metrics(frame, fps, cpu_usage, ram_usage):
    """Відображає метрики продуктивності (FPS, CPU, RAM)."""
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"CPU: {cpu_usage:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"RAM: {ram_usage:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def draw_adaptive_load_info(frame, level, decision, cooldown):
    """Відображає інформацію про рівень навантаження."""
    cv2.putText(frame, f"Difficulty Level: {level}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Decision: {decision}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if cooldown > 0:
        cooldown_text = f"Cooldown: {cooldown // FPS_ESTIMATE}s"
        cv2.putText(frame, cooldown_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


# --- 5. ОСНОВНА ЛОГІКА ДОДАТКУ ---

def run_application(selected_model_name, is_adaptive_mode):
    config = MODELS[selected_model_name]
    print(f"[INFO] Обрано модель: '{selected_model_name}'. Режим адаптивного навантаження: {'Увімкнено' if is_adaptive_mode else 'Вимкнено'}")
    print(f"[INFO] Завантаження ресурсів...")
    
    try: # Завантаження моделей
        dlib_face_detector = dlib.get_frontal_face_detector()
        dlib_landmark_predictor = keras_emotion_model = None
        if config["type"] == "dlib_heuristic": dlib_landmark_predictor = dlib.shape_predictor(config["predictor_path"])
        elif config["type"] in ["keras_cnn", "keras_rnn"]: keras_emotion_model = load_model(config["model_path"])
    except Exception as e:
        print(f"[ERROR] Помилка завантаження ресурсів: {e}")
        error_root = tk.Tk(); error_root.title("Помилка"); ttk.Label(error_root, text=f"Не вдалося завантажити модель:\n{config.get('model_path') or config.get('predictor_path')}", padding=20).pack(); ttk.Button(error_root, text="OK", command=error_root.destroy).pack(pady=10); error_root.mainloop()
        return

    print("[INFO] Завантаження завершено. Запуск відеопотоку...")
    cap = cv2.VideoCapture(0)
    prev_time = 0
    process = psutil.Process(os.getpid())

    # Змінні стану
    emotion_history = deque(maxlen=DECISION_BUFFER_SIZE)
    smoothed_emotion = ""
    # Змінні для адаптивного режиму
    current_level = 3
    cooldown_timer = 0
    last_decision = "Maintain"
    
    while True:
        ret, frame = cap.read();
        if not ret: break

        cpu_usage = process.cpu_percent(); ram_usage = process.memory_percent()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dlib_face_detector(gray, 0)
        current_emotion = "unknown"

        if faces:
            face_rect = faces[0] # Обробляємо лише перше знайдене обличчя
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
            
            # --- УНІВЕРСАЛЬНЕ РОЗПІЗНАВАННЯ ЕМОЦІЇ ---
            if config["type"] == "dlib_heuristic":
                landmarks = dlib_landmark_predictor(gray, face_rect)
                current_emotion = recognize_emotion_heuristic(landmarks)
            elif config["type"] in ["keras_cnn", "keras_rnn"]:
                face_roi_orig = frame[y:y+h, x:x+w]
                if face_roi_orig.size > 0:
                    img_size = config["img_size"]
                    face_roi = cv2.cvtColor(face_roi_orig, cv2.COLOR_BGR2GRAY) if config["color_mode"] == "grayscale" else cv2.resize(face_roi_orig, img_size)
                    if config["color_mode"] == "grayscale": face_roi = cv2.resize(face_roi, img_size)
                    face_roi_processed = img_to_array(face_roi.astype("float") / 255.0)
                    face_roi_processed = np.expand_dims(face_roi_processed, axis=0)
                    predictions = keras_emotion_model.predict(face_roi_processed, verbose=0)
                    current_emotion = config["labels"][np.argmax(predictions[0])]
            
            emotion_history.append(current_emotion)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, current_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # --- ЗАСТОСУВАННЯ ЛОГІКИ ЗАЛЕЖНО ВІД РЕЖИМУ ---
        if is_adaptive_mode:
            if cooldown_timer > 0: cooldown_timer -= 1
            if cooldown_timer == 0 and len(emotion_history) == DECISION_BUFFER_SIZE:
                negative_count = sum(1 for emo in emotion_history if emo in NEGATIVE_STRAIN_EMOTIONS)
                positive_count = sum(1 for emo in emotion_history if emo in POSITIVE_NEUTRAL_EMOTIONS)
                negative_ratio = negative_count / DECISION_BUFFER_SIZE
                positive_ratio = positive_count / DECISION_BUFFER_SIZE
                
                print(f"[DEBUG] neg_ratio={negative_ratio:.2f}, pos_ratio={positive_ratio:.2f}")

                new_decision_made = False
                if negative_ratio >= NEGATIVE_THRESHOLD and current_level > 1:
                    current_level -= 1; last_decision = "DECREASE"; new_decision_made = True
                elif positive_ratio >= POSITIVE_THRESHOLD and negative_ratio == 0 and current_level < 5:
                    current_level += 1; last_decision = "INCREASE"; new_decision_made = True
                else:
                    last_decision = "Maintain"
                
                if new_decision_made:
                    cooldown_timer = COOLDOWN_FRAMES
                    print(f"[DECISION] {last_decision} Load to Level {current_level}")
                emotion_history.clear() # Очищуємо буфер після прийняття рішення
            draw_adaptive_load_info(frame, current_level, last_decision, cooldown_timer)
        else:
            # Логіка для звичайного режиму (згладжена емоція)
            if len(emotion_history) > 0:
                smoothed_emotion = Counter(emotion_history).most_common(1)[0][0]
                cv2.putText(frame, f"Smoothed: {smoothed_emotion}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Відображення метрик продуктивності (завжди)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        draw_performance_metrics(frame, fps, cpu_usage, ram_usage)
        
        cv2.imshow("Ultimate Emotion Recognition System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


# --- 6. ГРАФІЧНИЙ ІНТЕРФЕЙС (GUI) ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Налаштування запуску")
    root.geometry("450x220")
    root.resizable(False, False)

    def start_button_clicked():
        selected_model = model_combobox.get()
        is_adaptive = adaptive_mode_var.get()
        if selected_model:
            root.destroy()
            run_application(selected_model, is_adaptive)
        else: print("[ERROR] Модель не обрано!")

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    ttk.Label(main_frame, text="1. Оберіть модель для розпізнавання:").pack(anchor="w")
    model_keys = list(MODELS.keys())
    model_combobox = ttk.Combobox(main_frame, values=model_keys, state="readonly", width=35)
    if model_keys: model_combobox.current(0)
    model_combobox.pack(pady=(5, 20), anchor="w")

    ttk.Label(main_frame, text="2. Оберіть режим роботи:").pack(anchor="w")
    adaptive_mode_var = tk.BooleanVar()
    adaptive_check = ttk.Checkbutton(main_frame, text="Увімкнути режим адаптивного навантаження", variable=adaptive_mode_var)
    adaptive_check.pack(pady=5, anchor="w")

    start_button = ttk.Button(main_frame, text="Запустити аналіз", command=start_button_clicked)
    start_button.pack(pady=20)

    root.mainloop()