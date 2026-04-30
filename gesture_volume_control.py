import cv2
import mediapipe as mp
import math
import pyautogui

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumb tip (4) and Index tip (8)
            x1 = int(hand_landmarks.landmark[4].x * w)
            y1 = int(hand_landmarks.landmark[4].y * h)

            x2 = int(hand_landmarks.landmark[8].x * w)
            y2 = int(hand_landmarks.landmark[8].y * h)

            # Draw circles
            cv2.circle(frame, (x1, y1), 10, (255,0,0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, (255,0,0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Distance between fingers
            length = math.hypot(x2 - x1, y2 - y1)

            # Map distance to volume (0 to 100)
            volume = int((length - 30) / (200 - 30) * 100)
            volume = max(0, min(100, volume))

            # Set system volume (simple scroll simulation)
            if volume > 70:
                pyautogui.press("volumeup")
            elif volume < 30:
                pyautogui.press("volumedown")

            cv2.putText(frame, f"Volume: {volume}%", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
