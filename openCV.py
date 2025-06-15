import cv2
import dlib
import numpy as np
import time
from collections import deque, Counter

DLIB_LANDMARK_PREDICTOR_PATH = "./Models/shape_predictor_68_face_landmarks.dat"

try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_LANDMARK_PREDICTOR_PATH)
except RuntimeError as e:
    print(f"Error loading Dlib model: {e}")
    print(f"Please ensure '{DLIB_LANDMARK_PREDICTOR_PATH}' exists and is accessible.")
    exit()

EMOTIONS = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Fear", "Disgust"]

def recognize_emotion_heuristic(landmarks):
    if landmarks.num_parts != 68:
        return "Neutral"

    try:
        eye_dist_x = landmarks.part(45).x - landmarks.part(36).x
        if eye_dist_x <= 1:
            eye_dist_x = 1

        mouth_opening_y = (landmarks.part(66).y - landmarks.part(62).y)
        norm_mouth_opening = mouth_opening_y / eye_dist_x

        mouth_corners_y_avg = (landmarks.part(48).y + landmarks.part(54).y) / 2
        inner_lips_y_avg = (landmarks.part(62).y + landmarks.part(66).y) / 2
        smile_indicator = (inner_lips_y_avg - mouth_corners_y_avg) / eye_dist_x

        eyebrow_left_y_avg = (landmarks.part(19).y + landmarks.part(20).y) / 2
        eyebrow_right_y_avg = (landmarks.part(23).y + landmarks.part(24).y) / 2
        eye_top_left_y_avg = (landmarks.part(37).y + landmarks.part(38).y) / 2
        eye_top_right_y_avg = (landmarks.part(43).y + landmarks.part(44).y) / 2

        brow_raise_left = (eye_top_left_y_avg - eyebrow_left_y_avg) / eye_dist_x
        brow_raise_right = (eye_top_right_y_avg - eyebrow_right_y_avg) / eye_dist_x
        avg_brow_raise = (brow_raise_left + brow_raise_right) / 2

        inner_brow_dist_x = landmarks.part(22).x - landmarks.part(21).x
        norm_inner_brow_dist = inner_brow_dist_x / eye_dist_x

        HAPPY_SMILE_THRESHOLD = 0.12
        HAPPY_MOUTH_OPEN_MIN = 0.03

        SURPRISED_MOUTH_OPEN_THRESHOLD = 0.25
        SURPRISED_BROW_RAISE_THRESHOLD = 0.20

        ANGRY_BROW_LOWERED_THRESHOLD = 0.08
        ANGRY_BROW_FURROW_THRESHOLD = 0.28

        SAD_SMILE_THRESHOLD = -0.02

        if avg_brow_raise > SURPRISED_BROW_RAISE_THRESHOLD and \
           norm_mouth_opening > SURPRISED_MOUTH_OPEN_THRESHOLD:
            return "Surprised"

        if smile_indicator > HAPPY_SMILE_THRESHOLD and \
           norm_mouth_opening > HAPPY_MOUTH_OPEN_MIN:
            return "Happy"

        if avg_brow_raise < ANGRY_BROW_LOWERED_THRESHOLD and \
           norm_inner_brow_dist < ANGRY_BROW_FURROW_THRESHOLD:
            return "Angry"

        if smile_indicator < SAD_SMILE_THRESHOLD:
            if not (avg_brow_raise < ANGRY_BROW_LOWERED_THRESHOLD and \
                    norm_inner_brow_dist < ANGRY_BROW_FURROW_THRESHOLD + 0.05):
                return "Sad"

    except Exception as e:
        return "Neutral"

    return "Neutral"

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    for i in range(0, landmarks.num_parts):
        point = (landmarks.part(i).x, landmarks.part(i).y)
        cv2.circle(image, point, radius, color, -1)

def draw_face_box(image, rect, color=(255, 0, 0), thickness=2):
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

def process_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    EMOTION_HISTORY_LENGTH = 15
    emotion_history = deque(maxlen=EMOTION_HISTORY_LENGTH)
    smoothed_emotion = "Neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_dlib = detector(gray, 0)

        current_raw_emotion_for_frame = "Neutral"

        for face_rect in faces_dlib:
            draw_face_box(frame, face_rect, color=(0, 0, 255))

            landmarks = predictor(gray, face_rect)
            draw_landmarks(frame, landmarks, color=(0, 255, 0))

            current_raw_emotion_for_frame = recognize_emotion_heuristic(landmarks)

        emotion_history.append(current_raw_emotion_for_frame)

        if len(emotion_history) == EMOTION_HISTORY_LENGTH:
            emotion_counts = Counter(emotion_history)
            if emotion_counts:
                most_common_emotion_in_window = emotion_counts.most_common(1)[0][0]
                smoothed_emotion = most_common_emotion_in_window
        elif not emotion_history:
            smoothed_emotion = "Neutral"

        cv2.putText(frame, f"Emotion: {smoothed_emotion}", (10, 70), font, 0.9, (36,255,12), 2)

        curr_time = time.time()
        if prev_time > 0:
            fps = 1 / (curr_time - prev_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), font, 0.7, (100, 255, 0), 2, cv2.LINE_AA)
        prev_time = curr_time

        cv2.imshow('Facial Emotion Recognition - Press Q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Facial Emotion Recognition Demo...")
    print("Press 'q' in the video window to quit.")
    print("Make sure you have 'shape_predictor_68_face_landmarks.dat' in the same directory or provide the correct path.")
    print("Note: Emotion recognition is heuristic-based and has limitations. Output is smoothed.")
    process_video()