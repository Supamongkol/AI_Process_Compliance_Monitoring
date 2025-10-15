import cv2
import json
import time

# ---------- CONFIG ----------
CONFIG_JSON = r"D:\4.Machine Learning(AI)\Image process\4. Activities Monitoring\AI_Process_Compliance_Monitoring\workflow_config.json"
SAVE_PATH = "roi0_capture.jpg"
# ----------------------------

# โหลด config และ ROI0
with open(CONFIG_JSON, "r") as f:
    config = json.load(f)

workflow = config["workflow"]
roi0_norm = workflow[0]["rect_norm"]

# เปิดกล้อง
cap = cv2.VideoCapture(1)
cv2.namedWindow("Capture ROI0")

print("📷 Press 'c' to capture ROI0 | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ไม่พบกล้องหรือเปิดไม่สำเร็จ")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # แปลง ROI0 เป็น pixel
    x1 = int(roi0_norm[0] * w)
    y1 = int(roi0_norm[1] * h)
    x2 = int(roi0_norm[2] * w)
    y2 = int(roi0_norm[3] * h)

    # วาดกรอบ ROI0 บนหน้าจอ
    disp = frame.copy()
    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(disp, "ROI0", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Capture ROI0", disp)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        roi_crop = frame[y1:y2, x1:x2]
        cv2.imwrite(SAVE_PATH, roi_crop)
        print(f"✅ ROI0 saved → {SAVE_PATH}")
        time.sleep(1)  # เว้นให้เห็นผล
        break

cap.release()
cv2.destroyAllWindows()
