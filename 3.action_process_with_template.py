import cv2
import mediapipe as mp
import numpy as np
import time
import json
import pandas as pd

CONFIG_JSON = "workflow_config.json"
LOG_CSV = "assembly_log.csv"
TEMPLATE_IMG = r"D:\255441\Project 2025\1.Process Activity\IMG20251010152443.jpg"  # template ของชิ้นงานใน ROI0
HAND_OPEN_THRESHOLD = 0.5  # threshold สำหรับ open/close

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

# ROI0 สำหรับ template matching
roi0_norm = workflow[0]["rect_norm"]
roi0_template = cv2.imread(TEMPLATE_IMG, cv2.IMREAD_GRAYSCALE)
if roi0_template is None:
    print(f"Warning: Template {TEMPLATE_IMG} not found.")
template_h, template_w = roi0_template.shape[:2]

# Initialize state
state = {step["id"]: {"timer_start": None, "elapsed": 0.0, "completed": False} for step in workflow}
logs = []

workflow_active = False  # track if workflow is active

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
    x1, y1, x2, y2 = roi_rect
    roi = frame_gray[y1:y2, x1:x2]
    if roi.shape[0] < template_h or roi.shape[1] < template_w:
        return False
    res = cv2.matchTemplate(roi, roi0_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val > 0.7  # threshold ปรับได้

cap = cv2.VideoCapture(0)
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

    # แปลง ROI0 เป็น pixel
    roi0_px = [int(roi0_norm[0]*w), int(roi0_norm[1]*h), int(roi0_norm[2]*w), int(roi0_norm[3]*h)]
    x0_1, y0_1, x0_2, y0_2 = roi0_px

    # ตรวจ Template Matching ROI0
    trigger_detected = template_match_roi0(frame_gray, roi0_px)

    # ตรวจมือว่าปิดบัง ROI0
    hand_on_roi0 = False
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            if all_landmarks_in_roi(hand, roi0_px, w, h):
                hand_on_roi0 = True
                break

    # เริ่ม workflow
    if trigger_detected and not workflow_active:
        workflow_active = True
        print(f"[{station_name}] Workflow started")

    # จบ workflow
    if workflow_active and (not trigger_detected and not hand_on_roi0):
        workflow_active = False
        print(f"[{station_name}] Workflow ended")
        # บันทึก log และ reset time
        for step in workflow:
            step_id = step["id"]
            st = state[step_id]
            if st["elapsed"] > 0:
                logs.append({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "roi": step_id,
                    "label": step["label"],
                    "rect_norm": step["rect_norm"],
                    "dwell_time": round(st["elapsed"],3),
                    "station": station_name,
                    "hand_state": None
                })
            st["timer_start"] = None
            st["elapsed"] = 0.0
            st["completed"] = False

    # วาด ROI และ dwell
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

    # ประมวลผลมือถ้า workflow active
    if workflow_active and results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(disp, hand, mp_hands.HAND_CONNECTIONS)
            ratio = hand_ratio(hand)
            hand_state = "OPEN" if ratio > HAND_OPEN_THRESHOLD else "CLOSED"
            cx = int(np.mean([lm.x for lm in hand.landmark])*w)
            cy = int(np.mean([lm.y for lm in hand.landmark])*h)
            col = (0,255,0) if hand_state=="OPEN" else (0,0,255)
            cv2.circle(disp,(cx,cy),8,col,-1)
            cv2.putText(disp,f"{hand_state} {ratio:.2f}",(cx+10,cy),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)

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
    if key==ord('q'):
        break
    elif key==ord('s'):
        if logs:
            pd.DataFrame(logs).to_csv(LOG_CSV,index=False)
            print(f"Log saved → {LOG_CSV}")

cap.release()
cv2.destroyAllWindows()
