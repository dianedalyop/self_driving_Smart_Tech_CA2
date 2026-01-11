# src/train.py
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from model import build_model
from data_preprocess import preprocess_image, build_training_df


TRACK1_MAIN = "data/track1_main/Udacity_sim_Track1_rec/driving_log.csv"
TRACK1_RECOVERY = "data/track1_recovery/Udacity_sim_track1_recovery/driving_log.csv"

BATCH_SIZE = 32
EPOCHS = 6


def generator(df, batch_size=BATCH_SIZE, augment=True):
    n = len(df)

    while True:
        df = df.sample(frac=1.0).reset_index(drop=True)  # shuffling each epoch

        for offset in range(0, n, batch_size):
            batch = df.iloc[offset:offset + batch_size]

            images = []
            steerings = []

            for _, row in batch.iterrows():
                center_path = row["center"]
                steering = float(row["steering"])

                image = cv2.imread(center_path)
                if image is None:
                    continue

                # preprocess_image converts BGR->RGB 
                
                image = preprocess_image(image)

                
                if augment and np.random.rand() < 0.5:
                    image = np.fliplr(image)
                    steering = -steering

                images.append(image)
                steerings.append(steering)

            if len(images) == 0:
                continue

            X = np.array(images, dtype=np.float32)
            y = np.array(steerings, dtype=np.float32)

            yield X, y


if __name__ == "__main__":
    print("Building balanced training dataframe...")
    df = build_training_df(
    

        TRACK1_MAIN,
        TRACK1_RECOVERY,
        near_zero_threshold=0.03,
        near_zero_keep_prob=0.10,
        bins=25,
        max_per_bin=300,
        seed=42
    )

    print("Steering mean:", df["steering"].mean())
    print("Steering median:", df["steering"].median())

    print("Final samples:", len(df))
    print("Near-zero %:", (df["steering"].abs() < 0.03).mean() * 100)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_generator = generator(train_df, augment=True)
    val_generator = generator(val_df, augment=False)

    print("Building model...")
    model = build_model()

    model.compile(loss="mse", optimizer=Adam(learning_rate=1e-4))

    checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    print("Training...")
    model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_df) // BATCH_SIZE),
        validation_data=val_generator,
        validation_steps=max(1, len(val_df) // BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )

    print("Training finished — model saved as model.h5")
