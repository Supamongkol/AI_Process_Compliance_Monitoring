import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import json

# ---------- CONFIG ----------
DEFAULT_DWELL_TIME = 0.5
LOG_CSV = "assembly_log.csv"
ROI_JSON = "roi_config.json"
STATION_NAME = input("Enter station name (e.g. STATION_01): ") or "STATION_01"

# ✅ FIX: ปรับ threshold ให้ stable ตาม palm-normalized ratio
OPEN_THRESHOLD = 0.20
# ----------------------------

ROIS = []
ROIS_NORM = []
NEXT_ROI_ID = 1
WORKFLOW = []
CURRENT_STEP_INDEX = 0

state = {}
logs = []

# mediapipe hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

drawing = False
ix, iy = -1, -1

def mouse_draw(event, x, y, flags, param):
    global drawing, ix, iy, NEXT_ROI_ID, ROIS, WORKFLOW
    if param is None:
        return
    h, w = param.shape[:2]
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        roi = {"id": NEXT_ROI_ID, "rect": (x1, y1, x2, y2), "label": f"Step {NEXT_ROI_ID}", "completed": False}
        roi_norm = {"id": NEXT_ROI_ID, "rect_norm": (x1/w, y1/h, x2/w, y2/h)}
        ROIS.append(roi)
        ROIS_NORM.append(roi_norm)
        WORKFLOW.append(NEXT_ROI_ID)
        state[NEXT_ROI_ID] = {
            "timer_start": None,
            "elapsed": 0.0,  # ✅ FIX: เก็บเวลาสะสม
            "completed": False
        }
        print(f"ROI added: {roi['label']} rect={roi['rect']} | normalized={roi_norm['rect_norm']}")
        NEXT_ROI_ID += 1

def all_landmarks_in_roi(hand_landmarks, rect, w, h):
    x1, y1, x2, y2 = rect
    for lm in hand_landmarks.landmark:
        px, py = lm.x * w, lm.y * h
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return False
    return True

def hand_grip_ratio(hand_landmarks):
    """วัดระยะเฉลี่ยของนิ้วที่เกี่ยวกับการหยิบจับ (4,8,12) จากข้อมือ"""
    tips = [4, 8, 12]
    wrist = hand_landmarks.landmark[0]
    distances = []
    for tip in tips:
        tip_point = hand_landmarks.landmark[tip]
        dist = np.linalg.norm([tip_point.x - wrist.x, tip_point.y - wrist.y])
        distances.append(dist)
    return np.mean(distances)  # ไม่ normalize ตาม orientation



def draw_checklist(img):
    h, w = img.shape[:2]
    panel = np.ones((h, 300, 3), dtype=np.uint8) * 40
    cv2.putText(panel, f"Checklist ({STATION_NAME})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    y = 60
    for roi in ROIS:
        rid = roi["id"]
        st = state[rid]
        dwell = st["elapsed"]
        color = (0,200,0) if st["completed"] else (180,180,180)
        cv2.rectangle(panel, (10, y), (280, y+40), color, -1)
        cv2.putText(panel, f"{roi['label']} | {dwell:.2f}s", (20, y+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += 50
    return np.hstack((img, panel))

def save_roi_config():
    data = {"station": STATION_NAME, "rois": ROIS_NORM}
    with open(ROI_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved ROI config → {ROI_JSON}")

def main():
    global CURRENT_STEP_INDEX, logs
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Assembly Monitor")
    cv2.setMouseCallback("Assembly Monitor", mouse_draw, None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        disp = frame.copy()
        cv2.setMouseCallback("Assembly Monitor", mouse_draw, disp)

        # วาด ROI
        for roi, roi_norm in zip(ROIS, ROIS_NORM):
            x1, y1, x2, y2 = roi["rect"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (200,200,200), 2)
            cv2.putText(disp, roi["label"], (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            nx1, ny1, nx2, ny2 = roi_norm["rect_norm"]
            cv2.putText(disp, f"({nx1:.2f},{ny1:.2f})", (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,255), 1)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(disp, hand, mp_hands.HAND_CONNECTIONS)
                ratio = hand_grip_ratio(hand)
                hand_state = "OPEN" if ratio > OPEN_THRESHOLD else "CLOSED"
                cx = int(np.mean([lm.x for lm in hand.landmark]) * w)
                cy = int(np.mean([lm.y for lm in hand.landmark]) * h)
                col = (0,255,0) if hand_state == "OPEN" else (0,0,255)
                cv2.circle(disp, (cx, cy), 8, col, -1)
                cv2.putText(disp, f"{hand_state} {ratio:.2f}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                for roi, roi_norm in zip(ROIS, ROIS_NORM):
                    rid = roi["id"]
                    st = state[rid]
                    if all_landmarks_in_roi(hand, roi["rect"], w, h) and hand_state == "CLOSED":
                        # ✅ FIX: นับต่อได้ ไม่ reset ถ้าออกแล้วกลับเข้า ROI
                        if st["timer_start"] is None:
                            st["timer_start"] = time.time()
                        elapsed = time.time() - st["timer_start"] + st["elapsed"]
                        st["elapsed"] = elapsed

                        if not st["completed"] and elapsed >= DEFAULT_DWELL_TIME:
                            st["completed"] = True
                            expected = WORKFLOW[CURRENT_STEP_INDEX] if CURRENT_STEP_INDEX < len(WORKFLOW) else None
                            nx1, ny1, nx2, ny2 = roi_norm["rect_norm"]
                            logs.append({
                                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "roi": rid,
                                "rect_norm": (nx1, ny1, nx2, ny2),
                                "dwell_time": round(elapsed, 3),
                                "station": STATION_NAME,
                                "hand_state": hand_state,
                                "hand_ratio": round(ratio, 3),
                                "result": "OK" if rid == expected else "OUT_OF_ORDER"
                            })
                            CURRENT_STEP_INDEX += 1
                            print(f"[{STATION_NAME}] Step {rid} completed ({logs[-1]['result']}) - {elapsed:.2f}s")
                    else:
                        # ✅ FIX: หยุดจับเวลา แต่ไม่ reset ค่า elapsed เดิม
                        if st["timer_start"] is not None:
                            st["elapsed"] += time.time() - st["timer_start"]
                            st["timer_start"] = None

        out = draw_checklist(disp)
        cv2.imshow("Assembly Monitor", out)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            for rid in state:
                state[rid].update({"elapsed": 0, "timer_start": None, "completed": False})
            CURRENT_STEP_INDEX = 0
            print("Reset all states")
        elif key == ord('s'):
            if logs:
                pd.DataFrame(logs).to_csv(LOG_CSV, index=False)
                print(f"Log saved to {LOG_CSV}")
            save_roi_config()

    cap.release()
    cv2.destroyAllWindows()
    if logs:
        pd.DataFrame(logs).to_csv(LOG_CSV, index=False)
        print(f"Final log saved to {LOG_CSV}")
    save_roi_config()

if __name__ == "__main__":
    main()
