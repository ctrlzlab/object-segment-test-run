from rembg import remove
from PIL import Image

input_path = "myimg2.jpg"
output_path = f"{input_path.split(".")[0]}_output.png"

input_img = Image.open(input_path)
output_img = remove(input_img)  # returns RGBA
output_img.save(output_path)
