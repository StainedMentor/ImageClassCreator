import os
from PIL import Image

source_folder = "imgc"
output_folder = os.path.join(source_folder, "compressed")

os.makedirs(output_folder, exist_ok=True)

# Target height in pixels
target_height = 360

for filename in os.listdir(source_folder):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        image_path = os.path.join(source_folder, filename)
        with Image.open(image_path) as img:
            width, height = img.size
            aspect_ratio = width / height
            new_width = int(target_height * aspect_ratio)
            resized_img = img.resize((new_width, target_height), Image.LANCZOS)
            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path, quality=85, optimize=True)

print("Compression complete.")