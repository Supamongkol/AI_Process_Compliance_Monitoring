import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from collections import OrderedDict

# ---------- CONFIG ----------
THRESH_PERCENT = 50
DEFAULT_DWELL_TIME = 0.5
LOG_CSV = "assembly_log.csv"
STATION_NAME = input("Enter station name (e.g. STATION_01): ") or "STATION_01"
# ----------------------------

# ROI storage
ROIS = []
NEXT_ROI_ID = 1
WORKFLOW = []
CURRENT_STEP_INDEX = 0

state = {}
logs = []

# mediapipe hands init
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# mouse draw ROI
drawing = False
ix, iy = -1, -1

def mouse_draw(event, x, y, flags, param):
    global drawing, ix, iy, NEXT_ROI_ID, ROIS, WORKFLOW
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        roi = {
            "id": NEXT_ROI_ID,
            "rect": (x1, y1, x2, y2),
            "label": f"Step {NEXT_ROI_ID}",
            "completed": False
        }
        ROIS.append(roi)
        WORKFLOW.append(NEXT_ROI_ID)
        state[NEXT_ROI_ID] = {
            "timer_start": None,
            "completed": False,
            "last_hit": False,
            "hand_state": "OPEN",
            "current_dwell": 0.0
        }
        print(f"ROI added: {roi['label']} rect={roi['rect']}")
        NEXT_ROI_ID += 1

def point_in_roi(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

# ตรวจจับระดับการกางของมือ (0 = หุบ, 1 = กาง)
def hand_open_ratio(hand_landmarks):
    tips = [4, 8, 12, 16, 20]  # ปลายนิ้ว
    palm_base = hand_landmarks.landmark[0]
    distances = []
    for tip in tips:
        tip_point = hand_landmarks.landmark[tip]
        dist = np.sqrt(
            (tip_point.x - palm_base.x) ** 2 +
            (tip_point.y - palm_base.y) ** 2
        )
        distances.append(dist)
    avg_dist = np.mean(distances)
    return avg_dist  # ค่านี้คือ ratio

def draw_checklist(img):
    h, w = img.shape[:2]
    panel_w = 300
    panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 40
    cv2.putText(panel, f"Checklist ({STATION_NAME})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    y = 60
    for i, roi in enumerate(ROIS):
        rid = roi["id"]
        label = roi["label"]
        st = state[rid]
        completed = st["completed"]
        dwell = st["current_dwell"]
        color = (180,180,180)
        text_color = (255,255,255)
        if completed:
            idx_in_workflow = WORKFLOW.index(rid) if rid in WORKFLOW else -1
            if idx_in_workflow == CURRENT_STEP_INDEX - 1:
                color = (0,200,0)
            else:
                color = (0,0,255)
        cv2.rectangle(panel, (10, y), (panel_w-10, y+40), color, -1)
        cv2.putText(panel, f"{label} | {dwell:.2f}s", (20, y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        y += 50
    combined = np.hstack((img, panel))
    return combined

def main():
    global CURRENT_STEP_INDEX, logs
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Assembly Monitor")
    cv2.setMouseCallback("Assembly Monitor", mouse_draw)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        frame_disp = frame.copy()

        # draw ROI
        for roi in ROIS:
            x1, y1, x2, y2 = roi["rect"]
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (200,200,200), 2)
            cv2.putText(frame_disp, roi["label"], (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        hand_centers = []
        hand_states = []
        hand_ratios = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame_disp, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # center of palm
                cx = int(np.mean([hand_landmarks.landmark[i].x for i in [0,5,9,13,17]]) * w)
                cy = int(np.mean([hand_landmarks.landmark[i].y for i in [0,5,9,13,17]]) * h)
                ratio = hand_open_ratio(hand_landmarks)
                state_label = "OPEN" if ratio > 0.4 else "CLOSED"
                hand_centers.append((cx, cy))
                hand_states.append(state_label)
                hand_ratios.append(ratio)
                color = (0,255,0) if state_label == "OPEN" else (0,0,255)
                cv2.circle(frame_disp, (cx, cy), 10, color, -1)
                cv2.putText(frame_disp, f"{state_label} {ratio:.2f}", (cx+15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # check each ROI
        for roi in ROIS:
            rid = roi["id"]
            st = state[rid]
            hit = False
            current_hand_state = "OPEN"
            current_ratio = 0.0
            for i, (cx, cy) in enumerate(hand_centers):
                if point_in_roi(cx, cy, roi["rect"]):
                    hit = True
                    current_hand_state = hand_states[i]
                    current_ratio = hand_ratios[i]
                    break

            if hit and current_hand_state == "CLOSED":
                if not st["last_hit"]:
                    st["timer_start"] = time.time()
                    st["last_hit"] = True
                else:
                    elapsed = time.time() - (st["timer_start"] or time.time())
                    st["current_dwell"] = elapsed
                    if not st["completed"] and elapsed >= DEFAULT_DWELL_TIME:
                        expected_rid = WORKFLOW[CURRENT_STEP_INDEX] if CURRENT_STEP_INDEX < len(WORKFLOW) else None
                        dwell = round(elapsed, 3)
                        roi_rect = roi["rect"]
                        log_entry = {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "roi": rid,
                            "rect": roi_rect,
                            "dwell_time": dwell,
                            "result": "OK" if rid == expected_rid else "OUT_OF_ORDER",
                            "station": STATION_NAME,
                            "hand_state": current_hand_state,
                            "hand_ratio": round(current_ratio, 3)
                        }
                        logs.append(log_entry)
                        st["completed"] = True
                        state[rid]["completed"] = True
                        if rid == expected_rid:
                            CURRENT_STEP_INDEX += 1
                        print(f"[{STATION_NAME}] Step {rid} completed ({log_entry['result']}) - Dwell {dwell}s - Ratio {current_ratio:.3f}")
            else:
                if st["last_hit"]:
                    # เมื่อมือออกจาก ROI ให้รีเซ็ตเวลา
                    if st["timer_start"]:
                        elapsed = time.time() - st["timer_start"]
                        st["current_dwell"] = elapsed
                    else:
                        st["current_dwell"] = 0.0
                st["timer_start"] = None
                st["last_hit"] = False

        out = draw_checklist(frame_disp)
        cv2.imshow("Assembly Monitor", out)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            for rid in state:
                state[rid] = {"timer_start": None, "completed": False, "last_hit": False, "hand_state": "OPEN", "current_dwell": 0.0}
            CURRENT_STEP_INDEX = 0
            logs.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "roi": None, "result": "RESET", "station": STATION_NAME})
            print("Reset all states")
        elif key == ord('s'):
            if logs:
                df = pd.DataFrame(logs)
                df.to_csv(LOG_CSV, index=False)
                print(f"Log saved to {LOG_CSV}")

    cap.release()
    cv2.destroyAllWindows()
    if logs:
        pd.DataFrame(logs).to_csv(LOG_CSV, index=False)
        print(f"Log saved to {LOG_CSV}")

if __name__ == "__main__":
    main()
