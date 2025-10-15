# capture_roi0_auto.py
import cv2
import numpy as np
import time
import json
import os

# ====== CONFIG ======
CONFIG_JSON = r"D:\4.Machine Learning(AI)\Image process\4. Activities Monitoring\AI_Process_Compliance_Monitoring\workflow_config.json"
TEMPLATE_SAVE_PATH = r"D:\4.Machine Learning(AI)\Image process\4. Activities Monitoring\AI_Process_Compliance_Monitoring\roi0_template_auto.jpg"
OLD_TEMPLATE = None  # ถ้ามี template เดิม เช่น r"path\to\old_template.jpg"
# ====================

# โหลด config
with open(CONFIG_JSON, "r") as f:
    config = json.load(f)

workflow = config["workflow"]

# ROI0 = พื้นที่วางชิ้นงาน
roi0_norm = workflow[0]["rect_norm"]

cap = cv2.VideoCapture(1)
cv2.namedWindow("Capture ROI0")

print("Press 'c' to capture single frame.")
print("Press 's' to auto capture 10 frames and select best.")
print("Press 'q' to quit.")

def laplacian_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def match_score(image, template):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val

old_template_gray = None
if OLD_TEMPLATE and os.path.exists(OLD_TEMPLATE):
    old_template_gray = cv2.imread(OLD_TEMPLATE, cv2.IMREAD_GRAYSCALE)
    print("Loaded existing template for matching reference.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ROI0 rect
    x1 = int(roi0_norm[0] * w)
    y1 = int(roi0_norm[1] * h)
    x2 = int(roi0_norm[2] * w)
    y2 = int(roi0_norm[3] * h)

    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(frame, "ROI0", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Capture ROI0", frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        # ถ่ายภาพเดียว
        cv2.imwrite(TEMPLATE_SAVE_PATH, roi)
        print(f"Saved ROI0 template → {TEMPLATE_SAVE_PATH}")

    elif key == ord('s'):
        # Auto capture 10 เฟรม
        print("Auto capturing 10 frames...")
        frames = []
        scores = []

        for i in range(10):
            ret, f = cap.read()
            if not ret:
                continue
            f = cv2.flip(f, 1)
            roi_crop = f[y1:y2, x1:x2]
            sharp = laplacian_sharpness(roi_crop)
            match_val = 0
            if old_template_gray is not None:
                match_val = match_score(roi_crop, old_template_gray)

            # รวมคะแนนจากความคมและ match (ถ้ามี)
            total_score = sharp + (match_val * 100 if old_template_gray is not None else 0)
            frames.append(roi_crop)
            scores.append(total_score)
            print(f"Frame {i+1}: sharp={sharp:.2f}, match={match_val:.3f}, total={total_score:.2f}")
            time.sleep(0.2)

        best_idx = int(np.argmax(scores))
        best_frame = frames[best_idx]
        cv2.imwrite(TEMPLATE_SAVE_PATH, best_frame)
        print(f"✅ Best frame selected (#{best_idx+1}) and saved to {TEMPLATE_SAVE_PATH}")

cap.release()
cv2.destroyAllWindows()
