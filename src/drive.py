import base64
import io
import os

import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow.keras.models import load_model
from wsgiref import simple_server

from data_preprocess import preprocess_image


sio = socketio.Server(
    async_mode="threading",
    cors_allowed_origins="*"
)

flask_app = Flask(__name__)



app = socketio.WSGIApp(sio, flask_app)



MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.h5")
print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH, safe_mode=False, compile=False)
print("Model loaded.")


MAX_SPEED = 20.0
MIN_SPEED = 5.0
current_speed = 0.0




@sio.event
def connect(sid, environ):
    print("Client connected:", sid)
    # Send neutral control on connect
    send_control(0.0, 0.0)


@sio.on("telemetry")
def telemetry(sid, data):
    global current_speed

    if not data:
        print("No telemetry data received.")
        return

    
    try:
        current_speed = float(data["speed"])
        steering_angle = float(data["steering_angle"])
    except Exception as e:
        print("Error reading speed/angle:", e, "raw data:", data)
        return

    print(f"Telemetry: speed={current_speed:.2f}, sim_angle={steering_angle:.4f}")


    img_str = data.get("image")
    if not img_str:
        print("No image data found in telemetry.")
        return

    try:
        image = Image.open(io.BytesIO(base64.b64decode(img_str)))
        image = np.asarray(image)
    except Exception as e:
        print("Error decoding image:", e)
        return

    processed = preprocess_image(image)
    processed = np.expand_dims(processed, axis=0)

   
    steering_pred = float(model.predict(processed)[0][0])

    if current_speed < MIN_SPEED:
        throttle = 0.5
    elif current_speed > MAX_SPEED:
        throttle = 0.1
    else:
        throttle = 0.2

    print(f"Sending control: steering={steering_pred:.4f}, throttle={throttle:.3f}")
    send_control(steering_pred, throttle)


def send_control(steering, throttle):
    # Emit to simulator
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering),
            "throttle": str(throttle),
        },
    )



if __name__ == "__main__":
    from wsgiref import simple_server
    import socketio

    
    app_wrapper = socketio.WSGIApp(sio, app)

    print("Starting drive server on port 4567")
    server = simple_server.make_server("", 4567, app_wrapper)
    server.serve_forever()