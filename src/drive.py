import socketio
import eventlet
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

# IMPORTANT: simple server, no engineio hacks
sio = socketio.Server(async_mode='eventlet', cors_allowed_origins='*')
app = Flask(__name__)

speed_limit = 10

def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    print("Telemetry received")

    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.expand_dims(image, axis=0)

    speed = float(data['speed'])
    steering_angle = float(model.predict(image)[0][0])

    # force startup movement
    if speed < 1.0:
        throttle = 0.8
    else:
        throttle = max(0.2, 1.0 - speed / speed_limit)

    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("Connected")
    send_control(0.0, 0.5)

def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        }
    )


# Load the model in the global scope so telemetry() can always find it
model = load_model("nvidia_elu_augmented.keras")

if __name__ == "__main__":
    # Wrap flask app with socketio's middleware
    app = socketio.Middleware(sio, app)
    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
