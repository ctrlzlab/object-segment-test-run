import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image

# Load image
image_path = "myimg2.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM model
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.cuda() if torch.cuda.is_available() else sam

# Create a mask generator and generate masks
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# Sort masks by area (descending)
sorted_masks = sorted(masks, key=lambda m: m['area'], reverse=True)

# How many main objects to save? (Top N by area)
N = 1  # Change this number as needed (e.g., 1, 2, 3, etc.)

# Convert the original image to RGBA
image_pil = Image.fromarray(image).convert("RGBA")

for idx, m in enumerate(sorted_masks[:N]):
    mask = m['segmentation']
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8)

    # Apply mask as alpha channel
    obj_img = image_pil.copy()
    obj_img.putalpha(mask_pil)

    # Optional: crop to bounding box of object (for tight PNGs)
    bbox = mask_pil.getbbox()
    if bbox:
        obj_img = obj_img.crop(bbox)

    out_path = f"main_object_{idx+1}.png"
    obj_img.save(out_path)
    print(f"Saved {out_path}")

print(f"Saved top {N} main object(s)!")
