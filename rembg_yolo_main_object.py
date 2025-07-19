from ultralytics import YOLO
from rembg import remove
from PIL import Image
import numpy as np
import cv2

# === Load Image ===
image_path = "myimg.jpg"
input_img = Image.open(image_path).convert("RGB")
image_np = np.array(input_img)

# === Step 1: YOLO Detection ===
model = YOLO("yolo11x.pt")  # or 'yolov8n.pt' for lighter model
results = model(image_np)

# === Find Largest Object (by area) ===
boxes = results[0].boxes.xyxy.cpu().numpy()
areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
largest_idx = np.argmax(areas)
x1, y1, x2, y2 = boxes[0].astype(int)

# === Step 2: Use rembg to remove background ===
fg_img = remove(input_img)  # RGBA image with transparency

# === Step 3: Crop just the main object from alpha mask ===
fg_np = np.array(fg_img)
alpha = fg_np[:, :, 3]  # Alpha channel
cropped_rgba = fg_np[y1:y2, x1:x2]

# === Optional: Remove floating fragments (mask again) ===
gray_alpha = (cropped_rgba[:, :, 3] > 10).astype(np.uint8) * 255
cnts, _ = cv2.findContours(gray_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    # Use bounding box of biggest segment in the crop
    cnt = max(cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    cropped_rgba = cropped_rgba[by:by+bh, bx:bx+bw]

# === Step 4: Save result ===
result = Image.fromarray(cropped_rgba)
result.save("main_object_rembg_yolo.png")
print("✅ Saved: main_object_rembg_yolo.png")
