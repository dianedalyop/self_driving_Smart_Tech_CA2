

import cv2
import numpy as np

# Crop, resize, normalize image
def preprocess_image(img):
    if img is None:
        raise ValueError("None image recived in preprocess image")
    
    # Cropping (top 50px, bottom 20px off)
    img = img[50:-20, :, :]

    # NVIDIA expected input size
    img = cv2.resize(img, (200, 66))

    # Convert BGR OpenCV -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    img = img / 255.0 - 0.5

    return img
