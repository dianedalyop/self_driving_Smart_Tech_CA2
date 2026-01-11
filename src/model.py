# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout


def build_model(input_shape=(66, 200, 3)):
    model = Sequential()

   
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=input_shape))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))

    model.add(Flatten())

    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))

    #  steering angle
    model.add(Dense(1))
    return model
