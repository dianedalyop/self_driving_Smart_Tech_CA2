import base64
import io
import os
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow.keras.models import load_model
from wsgiref import simple_server
import eventlet
from io import BytesIO
import cv2

from data_preprocess import preprocess_image

# Initialize the Socket.IO server and Flask app
sio = socketio.Server(async_mode='eventlet', cors_allowed_origins='*')
app = Flask(__name__)

speed_limit = 10

# Load the model in the global scope so the telemetry function can always access it
# Ensure 'nvidia_elu_augmented.keras' is in your project root folder
model = load_model("nvidia_elu_augmented.keras")

def img_preprocess(img):
    """
    Pre-processes the image to match the format used during training in bc.py.
    """
    img = img[60:135, :, :] # Crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # Convert to YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Decode the image from the simulator
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)
        image = np.expand_dims(image, axis=0)

        # Get current speed and predict steering angle
        speed = float(data['speed'])
        steering_angle = float(model.predict(image)[0][0])

        # Control logic for throttle
        if speed < 1.0:
            throttle = 0.8
        else:
            throttle = max(0.2, 1.0 - speed / speed_limit)

        print(f"Steering: {steering_angle:.4f} Throttle: {throttle:.4f} Speed: {speed:.4f}")
        send_control(steering_angle, throttle)
    else:
        sio.emit('manual', data={}, skip_sid=sid)

@sio.on('connect')
def connect(sid, environ):
    print("Connected to Simulator!")
    send_control(0.0, 0.0)

def send_control(steering_angle, throttle):
    """
    Sends steering and throttle commands back to the simulator.
    """
    sio.emit(
        'steer',
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        }
    )

if __name__ == "__main__":
    # Wrap the Flask app with Socket.IO middleware
    app = socketio.Middleware(sio, app)
    # Start the eventlet WSGI server on port 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)