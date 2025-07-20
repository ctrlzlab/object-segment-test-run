import torch
import numpy as np
import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# --- Load image ---
image_path = "myimg.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image_rgb.shape[:2]

# --- YOLO Detection ---
yolo_model = YOLO('yolo11x.pt')  # Use yolov8x.pt if yolo11x.pt is not available
results = yolo_model(image_rgb)

boxes = results[0].boxes.xyxy.cpu().numpy()
scores = results[0].boxes.conf.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()
class_names = results[0].names

# --- Sort boxes by confidence * area to prioritize prominent objects ---
areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
importance = scores * areas
sorted_indices = np.argsort(-importance)
boxes = boxes[sorted_indices]

# --- Load SAM ---
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to('cuda' if torch.cuda.is_available() else 'cpu')
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# --- Create a blank combined mask ---
combined_mask = np.zeros((height, width), dtype=bool)

# --- Segment top N most prominent objects (e.g., top 3) ---
N = 3  # or change to len(boxes) to segment all
for i, box in enumerate(boxes):
    print(f"Segmenting object {i+1} at box {box}")
    input_box = np.array(box, dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False
    )
    combined_mask |= masks[0]

# --- Convert combined mask to alpha channel ---
mask_uint8 = (combined_mask * 255).astype(np.uint8)
mask_pil = Image.fromarray(mask_uint8)

# --- Create final RGBA output ---
image_pil = Image.fromarray(image_rgb).convert("RGBA")
image_pil.putalpha(mask_pil)

# --- Save result ---
image_pil.save("main_objects_mask_auto.png")
print("✅ Saved: main_objects_mask_auto.png")
