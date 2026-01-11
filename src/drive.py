import base64
import io
import os

import eventlet
eventlet.monkey_patch()

import numpy as np
import socketio
from PIL import Image
from tensorflow.keras.models import load_model

from data_preprocess import preprocess_image


sio = socketio.Server(
    async_mode="eventlet",
    cors_allowed_origins="*"
)

app = socketio.WSGIApp(sio)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.h5")
print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH, safe_mode=False, compile=False)
print("Model loaded.")

MAX_SPEED = 20.0
MIN_SPEED = 5.0
current_speed = 0.0

# --- NEW: steering controls --- ( suggestion)
STEERING_AMPLIFICATION = 1.0   # keep if under-steering; reduce to 1.0 if twitchy
STEERING_DEADZONE = 0.03        # ignore tiny steering noise (helps stop drifting)


def send_control(steering: float, throttle: float) -> None:
    """Send steering + throttle to the simulator."""
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering),
            "throttle": str(throttle),
        },
    )


@sio.event
def connect(sid, environ, auth=None):
    print("Client connected:", sid)
    # Giving  the car a small push so it starts moving
    send_control(0.0, 0.35)


@sio.event
def disconnect(sid):
    print("Client disconnected:", sid)


@sio.on("*")
def catch_all(event, sid, data):
    if event != "telemetry":
        print("Event received:", event)


@sio.on("telemetry")
def telemetry(sid, data):
    global current_speed

    if not data:
        print("No telemetry data received.")
        return

    # Reading speed & steering from simulator
    try:
        current_speed = float(data.get("speed", 0.0))
        sim_angle = float(data.get("steering_angle", 0.0))
    except Exception as e:
        print("Error reading telemetry:", e, "raw:", data)
        return

    # Reading camera image
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

    try:
        processed = preprocess_image(image)
        processed = np.expand_dims(processed, axis=0)
        steering_pred = float(model.predict(processed, verbose=0).squeeze())
    except Exception as e:
        print("Preprocess/predict error:", e)
        send_control(0.0, 0.25)
        return

    # Safety checks
    if np.isnan(steering_pred) or np.isinf(steering_pred):
        steering_pred = 0.0

    # Amplify steering 
    steering_pred *= STEERING_AMPLIFICATION

    # Clamp to simulator range
    steering_pred = float(np.clip(steering_pred, -1.0, 1.0))

    # --- NEW: deadzone to stop tiny constant turning / drift ---
    if abs(steering_pred) < STEERING_DEADZONE:
        steering_pred = 0.0

    # Throttle control
    if current_speed < 2.0:
        throttle = 0.6
    elif current_speed < MIN_SPEED:
        throttle = 0.45
    elif current_speed > MAX_SPEED:
        throttle = 0.05
    else:
        throttle = 0.2

    print(
        f"Telemetry: speed={current_speed:.2f}, sim_angle={sim_angle:.4f} | "
        f"send steering={steering_pred:.4f}, throttle={throttle:.3f}"
    )

    send_control(steering_pred, throttle)


if __name__ == "__main__":
    import eventlet.wsgi

    print("Starting drive server on port 4567")
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
