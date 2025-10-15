# updated 11-Oct-2025 (fixed ROI0 template matching + match value display)

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import pandas as pd

# ---------- CONFIG ----------
CONFIG_JSON = r"D:\4.Machine Learning(AI)\Image process\4. Activities Monitoring\AI_Process_Compliance_Monitoring\workflow_config.json"
LOG_CSV = "assembly_log.csv"
TEMPLATE_IMG = r"D:\4.Machine Learning(AI)\Image process\4. Activities Monitoring\roi0_capture.jpg"  # template ของชิ้นงานใน ROI0
HAND_OPEN_THRESHOLD = 0.2
# ----------------------------

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

# โหลด template
roi0_norm = workflow[0]["rect_norm"]
roi0_template = cv2.imread(TEMPLATE_IMG, cv2.IMREAD_GRAYSCALE)
if roi0_template is None:
    raise FileNotFoundError(f"Template not found: {TEMPLATE_IMG}")
template_h, template_w = roi0_template.shape[:2]

# Initialize state
state = {step["id"]: {"timer_start": None, "elapsed": 0.0, "completed": False} for step in workflow}
logs = []

workflow_active = False
work_in_progress = False   # state flag ป้องกัน start/stop ซ้อนกัน

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

def template_match_roi0(frame_gray, roi_rect):
    """Template matching เฉพาะ ROI0 โดย resize template ให้ตรงกับ ROI"""
    x1, y1, x2, y2 = roi_rect
    roi = frame_gray[y1:y2, x1:x2]

    if roi.size == 0:
        return False, 0.0

    # Resize template ให้ขนาดเท่ากับ ROI0 ปัจจุบัน
    template_resized = cv2.resize(roi0_template, (roi.shape[1], roi.shape[0]))
    res = cv2.matchTemplate(roi, template_resized, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return (max_val > 0.3), max_val  # threshold 0.6 ปรับได้

# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(1)
cv2.namedWindow("Assembly Monitor")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = hands.process(rgb)
    disp = frame.copy()

    # ROI0 pixel
    roi0_px = [int(roi0_norm[0]*w), int(roi0_norm[1]*h), int(roi0_norm[2]*w), int(roi0_norm[3]*h)]
    x0_1, y0_1, x0_2, y0_2 = roi0_px

    # Template Matching ROI0
    trigger_detected, match_val = template_match_roi0(frame_gray, roi0_px)

    # ตรวจว่ามีมือบัง ROI0 หรือไม่
    hand_on_roi0 = False
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            if all_landmarks_in_roi(hand, roi0_px, w, h):
                hand_on_roi0 = True
                break

    # ---- Logic เริ่ม/จบงาน ----
    if not work_in_progress and trigger_detected:
        work_in_progress = True
        workflow_active = True
        print(f"[{station_name}] Workflow started")

    elif work_in_progress and not trigger_detected and not hand_on_roi0:
        work_in_progress = False
        workflow_active = False
        print(f"[{station_name}] Workflow ended")

        # บันทึก log และ reset state
        for step in workflow:
            st = state[step["id"]]
            if st["elapsed"] > 0:
                logs.append({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "roi": step["id"],
                    "label": step["label"],
                    "rect_norm": step["rect_norm"],
                    "dwell_time": round(st["elapsed"], 3),
                    "station": station_name,
                    "hand_state": None
                })
            st["timer_start"] = None
            st["elapsed"] = 0.0
            st["completed"] = False

    # วาด ROI
    for step in workflow:
        x1 = int(step["rect_norm"][0]*w)
        y1 = int(step["rect_norm"][1]*h)
        x2 = int(step["rect_norm"][2]*w)
        y2 = int(step["rect_norm"][3]*h)
        st = state[step["id"]]
        color = (0,200,0) if st["completed"] else (200,200,200)
        cv2.rectangle(disp, (x1,y1),(x2,y2), color,2)
        cv2.putText(disp, f"{step['label']} | {st['elapsed']:.2f}s", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,2)

    # แสดง match value บนจอ
    cv2.putText(disp, f"ROI0 Match: {match_val:.3f}", (x0_1, y0_1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ---- ตรวจมือระหว่าง workflow ----
    if workflow_active and results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(disp, hand, mp_hands.HAND_CONNECTIONS)
            ratio = hand_ratio(hand)
            hand_state = "OPEN" if ratio > HAND_OPEN_THRESHOLD else "CLOSED"
            cx = int(np.mean([lm.x for lm in hand.landmark])*w)
            cy = int(np.mean([lm.y for lm in hand.landmark])*h)
            col = (0,255,0) if hand_state=="OPEN" else (0,0,255)
            cv2.circle(disp,(cx,cy),8,col,-1)
            cv2.putText(disp,f"{hand_state} {ratio:.2f}",(cx+10,cy),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)

            for step in workflow:
                x1 = int(step["rect_norm"][0]*w)
                y1 = int(step["rect_norm"][1]*h)
                x2 = int(step["rect_norm"][2]*w)
                y2 = int(step["rect_norm"][3]*h)
                st = state[step["id"]]

                # check dwell
                if all_landmarks_in_roi(hand,(x1,y1,x2,y2),w,h) and hand_state==step["hand_condition"]:
                    if st["timer_start"] is None:
                        st["timer_start"] = time.time()
                    elapsed = time.time() - st["timer_start"] + st["elapsed"]
                    st["elapsed"] = elapsed

                    if not st["completed"] and elapsed>=step["dwell_time"]:
                        st["completed"] = True
                        print(f"[{station_name}] Step {step['id']} completed - {elapsed:.2f}s")
                else:
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
