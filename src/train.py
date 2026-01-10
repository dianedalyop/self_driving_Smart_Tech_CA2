import os 
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from model import build_model
from data_preprocess import preprocess_image


TRACK1_MAIN = "data/track1_main/Udacity_sim_Track1_rec/driving_log.csv"
TRACK1_RECOVERY = "data/track1_recovery/Udacity_sim_track1_recovery/driving_log.csv"

BATCH_SIZE = 32
EPOCHS = 3 
#increased epochs. more time to learn steering behaviour



def load_samples():
    samples = []


    for log_path in [TRACK1_MAIN, TRACK1_RECOVERY]:
        if not os.path.exists(log_path):
            print(f"Missing: {log_path}")
            continue

        with open(log_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)

    print(f"Total samples loaded: {len(samples)}")
    return samples


def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)

    while True:
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch = samples[offset:offset + batch_size]

            images = []
            steerings = []

            for row in batch:
                center_path = row[0]      # center image path
                steering = float(row[3])  # steering angle

                image = cv2.imread(center_path)
                if image is None:
                    # Skip if image can't be read
                    continue

                image = preprocess_image(image)

                images.append(image)
                steerings.append(steering)

            if len(images) == 0:
                
                continue

            X = np.array(images)
            y = np.array(steerings)

           
            X, y = sklearn.utils.shuffle(X, y)
            yield X, y


if __name__ == "__main__":

    print("Loading samples...")
    samples = load_samples()

    train_samples, val_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples)
    val_generator = generator(val_samples)

    print("Building model...")
    model = build_model()

    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=1e-4)
    )

    checkpoint = ModelCheckpoint(
        "model.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True
    )

    print("Training...")
    model.fit(
        train_generator,
        steps_per_epoch=len(train_samples) // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=len(val_samples) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose=1
    )

    print("Training finished — model saved as model.h5")
