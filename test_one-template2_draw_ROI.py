import cv2
import time

# ======= โหลดภาพ =======
image_path = r"D:\255441\Project 2025\1.Process Activity\IMG20251010152443.jpg"
template_path = r"D:\255441\Project 2025\1.Process Activity\IMG20251010152443.jpg"

img = cv2.imread(image_path)
template_img = cv2.imread(template_path)

if img is None or template_img is None:
    print("❌ ไม่พบไฟล์รูป")
    exit()

# ======= Resize สำหรับ display =======
def resize_for_display(image, max_size=(1280,720)):
    h, w = image.shape[:2]
    scale = min(max_size[0]/w, max_size[1]/h, 1.0)
    return cv2.resize(image, (int(w*scale), int(h*scale))), scale

img_display, scale_scene = resize_for_display(img)
template_display, scale_template = resize_for_display(template_img)

# ======= เลือก ROI ของ scene =======
print("➡️ เลือก ROI บน scene (ลากกรอบแล้ว Enter)")
roi = cv2.selectROI("Select ROI - Scene", img_display, showCrosshair=True, fromCenter=False)
x, y, w, h = [int(coord/scale_scene) for coord in roi]  # แปลงกลับเป็นขนาดจริง
scene_roi = img[y:y+h, x:x+w]
cv2.destroyWindow("Select ROI - Scene")

# ======= เลือก ROI ของ template =======
print("➡️ เลือก ROI บน template (ลากกรอบแล้ว Enter)")
roi_template = cv2.selectROI("Select ROI - Template", template_display, showCrosshair=True, fromCenter=False)
tx, ty, tw, th = [int(coord/scale_template) for coord in roi_template]
template_roi = template_img[ty:ty+th, tx:tx+tw]
cv2.destroyWindow("Select ROI - Template")

# ======= Auto-scale template ให้เท่ากับ scene ROI =======
template_resized = cv2.resize(template_roi, (w, h))  # w,h ของ scene ROI

# ======= แปลงเป็น Gray =======
scene_gray = cv2.cvtColor(scene_roi, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

# ======= Template Matching =======
start = time.time()
result = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
end = time.time()

# ======= วาดกรอบ =======
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(scene_roi, top_left, bottom_right, (0,255,0), 2)
cv2.putText(scene_roi, f"Conf: {max_val*100:.1f}%", 
            (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# ======= ปรับขนาดภาพสำหรับแสดง =======
scene_display, _ = resize_for_display(scene_roi)
cv2.imshow("Template Matching Auto-Scaled", scene_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ======= แสดงผลใน console =======
print("=== Template Matching Result ===")
print(f"Match confidence : {max_val*100:.2f}%")
print(f"Best location    : {top_left}")
print(f"Time taken       : {end-start:.4f} sec")
