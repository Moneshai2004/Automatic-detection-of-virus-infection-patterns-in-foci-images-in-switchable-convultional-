import os
import numpy as np
from PIL import Image

def create_dummy_images(dir_path, num_images=10, img_size=(256, 256)):
    os.makedirs(dir_path, exist_ok=True)
    for i in range(num_images):
        # Create a white image
        image = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 255
        img = Image.fromarray(image)
        img.save(os.path.join(dir_path, f'healthy_image_{i}.jpg'))

# Create dummy healthy images
create_dummy_images('/home/moni/Desktop/virus-infection-detection/data/healthy', num_images=10)
