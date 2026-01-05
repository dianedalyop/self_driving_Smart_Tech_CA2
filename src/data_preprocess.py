

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


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



TRACK1_MAIN = "data/track1_main/Udacity_sim_Track1_rec/driving_log.csv"
TRACK1_RECOVERY = "data/track1_recovery/Udacity_sim_Track1_recovery/driving_log.csv"


def load_data():
    df1 = pd.read_csv(TRACK1_MAIN)
    df2 = pd.read_csv(TRACK1_RECOVERY)

    df = pd.concat([df1, df2], ignore_index=True)

    print("Total number of samples:", len(df))
    print("Columns:", df.columns)

    return df

if __name__ == "__main__":
    df = load_data()
