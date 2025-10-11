import cv2
import numpy as np
import time

# ======= ตั้งชื่อไฟล์ =======
image_path = r"C:\Users\supamongkol\Downloads\IMG20251010151339.jpg"       # รูปที่ต้องการค้นหา (scene)
template_path = r"D:\255441\Project 2025\1.Process Activity\IMG20251010152443.jpg" # รูปอ้างอิง (reference)

# ======= โหลดภาพ =======
img = cv2.imread(image_path)
template = cv2.imread(template_path)

# ตรวจสอบการโหลด
if img is None or template is None:
    print("❌ ไม่พบไฟล์รูป กรุณาตรวจสอบชื่อไฟล์หรือ path")
    exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
h, w = template_gray.shape[:2]

# ======= เริ่มจับเวลา =======
start = time.time()

# ======= ทำ Template Matching =======
result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# ======= หาค่าความตรงกันสูงสุด =======
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
end = time.time()

# ======= วาดกรอบตรงตำแหน่งที่เจอ =======
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)

# ======= คำนวณประสิทธิภาพ =======
elapsed_time = end - start
confidence = max_val * 100  # เปลี่ยนเป็นเปอร์เซ็นต์

# ======= แสดงผลลัพธ์ใน console =======
print("=== Template Matching Result ===")
print(f"Match confidence : {confidence:.2f}%")
print(f"Best location    : {top_left}")
print(f"Time taken       : {elapsed_time:.4f} sec")

# ======= ใส่ข้อความบนภาพ =======
cv2.putText(img, f"Conf: {confidence:.1f}%", (top_left[0], top_left[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# ======= ปรับขนาดภาพก่อนแสดงให้พอดีจอ =======
screen_res = 1280, 720   # ความละเอียดจอโดยประมาณ (ปรับได้)
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)

window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

resized_img = cv2.resize(img, (window_width, window_height))

# ======= แสดงภาพผลลัพธ์ =======
cv2.imshow("Template Matching Result", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
