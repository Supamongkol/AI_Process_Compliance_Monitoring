import cv2
import mediapipe as mp
import numpy as np
import time
import json
import pandas as pd

CONFIG_JSON = "workflow_config.json"
LOG_CSV = "assembly_log.csv"
HAND_OPEN_THRESHOLD = 0.20  # threshold สำหรับ open/close

# Mediapipe hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# โหลด workflow
with open(CONFIG_JSON, "r") as f:
    config = json.load(f)

workflow = config["workflow"]
station_name = config.get("station", "STATION_01")

# Initialize state
state = {step["id"]: {"timer_start": None, "elapsed": 0.0, "completed": False} for step in workflow}
logs = []

def hand_ratio(hand_landmarks):
    tips = [4, 8, 12]
    wrist = hand_landmarks.landmark[0]
    distances = [np.linalg.norm([hand_landmarks.landmark[i].x - wrist.x,
                                 hand_landmarks.landmark[i].y - wrist.y]) for i in tips]
    return np.mean(distances)

def all_landmarks_in_roi(hand_landmarks, rect, w, h):
    x1, y1, x2, y2 = rect
    for lm in hand_landmarks.landmark:
        px, py = lm.x * w, lm.y * h
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return False
    return True

cap = cv2.VideoCapture(0)
cv2.namedWindow("Assembly Monitor")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    disp = frame.copy()

    # วาด ROI
    for step in workflow:
        x1 = int(step["rect_norm"][0] * w)
        y1 = int(step["rect_norm"][1] * h)
        x2 = int(step["rect_norm"][2] * w)
        y2 = int(step["rect_norm"][3] * h)
        st = state[step["id"]]
        color = (0, 200, 0) if st["completed"] else (200, 200, 200)
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        cv2.putText(disp, f"{step['label']} | {st['elapsed']:.2f}s",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(disp, hand, mp_hands.HAND_CONNECTIONS)
            ratio = hand_ratio(hand)
            hand_state = "OPEN" if ratio > HAND_OPEN_THRESHOLD else "CLOSED"
            cx = int(np.mean([lm.x for lm in hand.landmark]) * w)
            cy = int(np.mean([lm.y for lm in hand.landmark]) * h)
            col = (0,255,0) if hand_state == "OPEN" else (0,0,255)
            cv2.circle(disp, (cx, cy), 8, col, -1)
            cv2.putText(disp, f"{hand_state} {ratio:.2f}", (cx+10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            # ตรวจแต่ละ step
            for step in workflow:
                x1 = int(step["rect_norm"][0] * w)
                y1 = int(step["rect_norm"][1] * h)
                x2 = int(step["rect_norm"][2] * w)
                y2 = int(step["rect_norm"][3] * h)
                st = state[step["id"]]

                if all_landmarks_in_roi(hand, (x1,y1,x2,y2), w, h) and hand_state == step["hand_condition"]:
                    # เริ่มจับเวลาหรือสะสม
                    if st["timer_start"] is None:
                        st["timer_start"] = time.time()
                    elapsed = time.time() - st["timer_start"] + st["elapsed"]
                    st["elapsed"] = elapsed

                    # เช็ค dwell_time
                    if not st["completed"] and elapsed >= step["dwell_time"]:
                        st["completed"] = True
                        logs.append({
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "roi": step["id"],
                            "label": step["label"],
                            "rect_norm": step["rect_norm"],
                            "dwell_time": round(elapsed, 3),
                            "station": station_name,
                            "hand_state": hand_state
                        })
                        print(f"[{station_name}] Step {step['id']} completed - {elapsed:.2f}s")
                else:
                    # หยุดจับเวลา แต่ไม่ reset elapsed
                    if st["timer_start"] is not None:
                        st["elapsed"] += time.time() - st["timer_start"]
                        st["timer_start"] = None

    cv2.imshow("Assembly Monitor", disp)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if logs:
            pd.DataFrame(logs).to_csv(LOG_CSV, index=False)
            print(f"Log saved → {LOG_CSV}")

cap.release()
cv2.destroyAllWindows()
