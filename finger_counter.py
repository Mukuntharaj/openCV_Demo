"""
========================================================
  HAND GESTURE FINGER COUNTER  (1 – 10)
  Uses: OpenCV + MediaPipe Hands
  Author: Generated for Prof. Mukunth
  Python 3.8+
========================================================

SETUP (run once):
    pip install opencv-python mediapipe numpy

USAGE:
    python finger_counter.py

CONTROLS:
    q  → quit
    s  → save screenshot
    r  → reset counter
    m  → toggle mirror mode
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ─────────────────────────────────────────────
#  MediaPipe initialisation
# ─────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode        = False,
    max_num_hands            = 2,          # detect both hands → counts up to 10
    min_detection_confidence = 0.75,
    min_tracking_confidence  = 0.60,
)

# ─────────────────────────────────────────────
#  Finger landmark indices (MediaPipe Hands)
#
#  Each finger has 4 landmarks (0 = wrist):
#  Tip indices  : [4, 8, 12, 16, 20]  → thumb, index, middle, ring, pinky
#  MCP indices  : [2, 5,  9, 13, 17]  → base knuckles (used for thumb)
#  PIP indices  : [3, 6, 10, 14, 18]  → middle knuckles
# ─────────────────────────────────────────────
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

# ─────────────────────────────────────────────
#  Finger number words
# ─────────────────────────────────────────────
NUMBER_WORDS = {
    0:  "ZERO",
    1:  "ONE",
    2:  "TWO",
    3:  "THREE",
    4:  "FOUR",
    5:  "FIVE",
    6:  "SIX",
    7:  "SEVEN",
    8:  "EIGHT",
    9:  "NINE",
    10: "TEN",
}

# ─────────────────────────────────────────────
#  Color palette  (BGR)
# ─────────────────────────────────────────────
CLR_BG      = (15,  15,  20)
CLR_GREEN   = (50, 220,  80)
CLR_CYAN    = (200, 220, 50)
CLR_ORANGE  = (30, 140, 255)
CLR_WHITE   = (240, 240, 240)
CLR_GREY    = (90,  90, 100)
CLR_RED     = (50,  50, 220)


# ─────────────────────────────────────────────
#  Core logic: count extended fingers
# ─────────────────────────────────────────────
def count_fingers(landmarks, hand_label: str) -> int:
    """
    Returns the number of extended fingers for one detected hand.

    Thumb logic:
        - For a RIGHT hand shown on-screen (mirrored camera feed it is
          actually the user's LEFT hand), the thumb tip (lm[4].x) should
          be to the LEFT  of lm[3].x to be extended.
        - For a LEFT hand on-screen (user's RIGHT hand), tip should be
          to the RIGHT of lm[3].x.
    Fingers 2-5: tip_y < pip_y  (tip above the second knuckle → extended).
    """
    lm    = landmarks.landmark
    count = 0

    # ── Thumb ─────────────────────────────────
    if hand_label == "Right":
        # mirrored: user's left hand appears on right side
        if lm[TIP_IDS[0]].x < lm[TIP_IDS[0] - 1].x:
            count += 1
    else:
        if lm[TIP_IDS[0]].x > lm[TIP_IDS[0] - 1].x:
            count += 1

    # ── Fingers 2–5 ───────────────────────────
    for i in range(1, 5):
        if lm[TIP_IDS[i]].y < lm[PIP_IDS[i]].y:
            count += 1

    return count


# ─────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1):
    """Draw a filled or outlined rounded rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)


def draw_big_number(img, total: int, h: int, w: int):
    """Render large centred number + word label on the frame."""
    word = NUMBER_WORDS.get(total, str(total))

    # big digit
    font_scale = 5.0
    thickness  = 10
    text       = str(total)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    cx = (w - tw) // 2
    cy = h // 2 + th // 2

    # shadow
    cv2.putText(img, text, (cx + 4, cy + 4),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 4)
    # main
    color = CLR_GREEN if total > 0 else CLR_GREY
    cv2.putText(img, text, (cx, cy),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)

    # word below
    font_scale2  = 1.4
    (ww, wh), _  = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale2, 3)
    cv2.putText(img, word, ((w - ww) // 2, cy + wh + 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale2, CLR_CYAN, 3)


def draw_hand_badges(img, hand_counts: list, h: int, w: int):
    """Show per-hand counts at the bottom."""
    labels = ["LEFT HAND", "RIGHT HAND"]
    for i, (label, cnt) in enumerate(zip(labels, hand_counts)):
        x = 30 + i * (w // 2)
        y = h - 60
        draw_rounded_rect(img, x, y - 30, x + 200, y + 10, 8,
                          (30, 60, 30), -1)
        cv2.putText(img, f"{label}: {cnt}", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_GREEN, 2)


def draw_hud(img, fps: float, mirror: bool, h: int, w: int):
    """Heads-up display: FPS + controls."""
    # top bar
    cv2.rectangle(img, (0, 0), (w, 40), (20, 20, 30), -1)
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_CYAN, 2)
    title = "FINGER COUNTER  |  1-10  |  Both Hands"
    cv2.putText(img, title, (w // 2 - 220, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_WHITE, 1)
    mirror_txt = "MIRROR: ON" if mirror else "MIRROR: OFF"
    cv2.putText(img, mirror_txt, (w - 160, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_ORANGE, 2)

    # bottom hint bar
    cv2.rectangle(img, (0, h - 28), (w, h), (20, 20, 30), -1)
    hint = "Q: Quit   S: Screenshot   R: Reset   M: Mirror"
    cv2.putText(img, hint, (w // 2 - 250, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_GREY, 1)


# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mirror       = True    # flip webcam so it feels natural
    prev_time    = time.time()
    screenshot_n = 0
    last_total   = 0

    print("[INFO] Starting Finger Counter. Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read from webcam.")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # ── dark overlay background ────────────
        overlay = np.zeros_like(frame)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # ── MediaPipe processing ───────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        total_fingers = 0
        hand_counts   = [0, 0]   # [left, right] from user's perspective

        if res.multi_hand_landmarks:
            for hand_lm, hand_info in zip(
                    res.multi_hand_landmarks, res.multi_handedness):

                label = hand_info.classification[0].label   # "Left" or "Right"

                # ── draw skeleton ──────────────
                mp_drawing.draw_landmarks(
                    frame, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                # ── highlight fingertips ───────
                for tip_id in TIP_IDS:
                    lm  = hand_lm.landmark[tip_id]
                    cx_ = int(lm.x * w)
                    cy_ = int(lm.y * h)
                    cv2.circle(frame, (cx_, cy_), 12, CLR_GREEN, -1)
                    cv2.circle(frame, (cx_, cy_), 14, CLR_WHITE,  2)

                # ── count ─────────────────────
                n = count_fingers(hand_lm, label)
                total_fingers += n

                # store per-hand (mirror flips Left↔Right label)
                if label == "Left":
                    hand_counts[0] = n
                else:
                    hand_counts[1] = n

        total_fingers = min(total_fingers, 10)

        # ── big number display ─────────────────
        draw_big_number(frame, total_fingers, h, w)

        # ── per-hand badges ────────────────────
        if res.multi_hand_landmarks:
            draw_hand_badges(frame, hand_counts, h, w)

        # ── FPS calc ──────────────────────────
        cur_time  = time.time()
        fps       = 1.0 / (cur_time - prev_time + 1e-9)
        prev_time = cur_time

        draw_hud(frame, fps, mirror, h, w)

        # ── show ──────────────────────────────
        cv2.imshow("Finger Counter (1-10)", frame)

        # ── key handling ──────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{screenshot_n:03d}.png"
            cv2.imwrite(fname, frame)
            screenshot_n += 1
            print(f"[SAVED] {fname}")
        elif key == ord('m'):
            mirror = not mirror
        elif key == ord('r'):
            print("[RESET]")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
