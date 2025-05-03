import os
from PIL import Image

source_folder = "minibark"

# Set the target dimension and mode: 'width' or 'height'
target_size = 360
resize_mode = 'width'  # Change to 'height' to fix height instead

def compress_images_recursively(input_dir):
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg")):
                image_path = os.path.join(root, filename)

                with Image.open(image_path) as img:
                    width, height = img.size

                    if resize_mode == 'width':
                        aspect_ratio = height / width
                        new_width = target_size
                        new_height = int(target_size * aspect_ratio)
                    elif resize_mode == 'height':
                        aspect_ratio = width / height
                        new_height = target_size
                        new_width = int(target_size * aspect_ratio)
                    else:
                        raise ValueError("resize_mode must be 'width' or 'height'")

                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                    resized_img.save(image_path, quality=85, optimize=True)

compress_images_recursively(source_folder)
print("Recursive compression complete.")
