import cv2
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import firebase_admin
from firebase_admin import credentials, firestore
import tensorflow as tf
import os

# ---------------- CONFIG ----------------
VIDEO_PATH = "samp_vidd.mp4"
GPX_PATH = "20260228-100456.gpx"
MODEL_PATH = "pothole_detector_ref/yolov8n_train/weights/best_saved_model/best_int8.tflite"
SERVICE_KEY = "serviceAccountKey.json"

CONF_THRESHOLD = 0.60
FRAME_SKIP = 30
MIN_UPLOAD_INTERVAL = 3

COLLECTION = "artifacts/potholedetect-bb0a9/public/data/potholes"

# ⚠ SET THIS TO YOUR VIDEO START TIME (FROM OVERLAY)
VIDEO_START_IST = datetime(2026, 2, 28, 10, 8, 37)

# ----------------------------------------


# ---------- FILE CHECK ----------
for path in [VIDEO_PATH, GPX_PATH, MODEL_PATH, SERVICE_KEY]:
    if not os.path.exists(path):
        raise RuntimeError(f"File not found: {path}")

print("All required files found.\n")


# ---------- FIREBASE ----------
cred = credentials.Certificate(SERVICE_KEY)
firebase_admin.initialize_app(cred)
db = firestore.client()


# ---------- GPX LOADER ----------
def load_gpx(gpx_path):
    tree = ET.parse(gpx_path)
    root = tree.getroot()

    gps_points = []

    for elem in root.iter():
        if 'trkpt' in elem.tag:
            lat = float(elem.attrib.get('lat'))
            lon = float(elem.attrib.get('lon'))

            for child in elem:
                if 'time' in child.tag:
                    # GPX times are UTC
                    timestamp = datetime.fromisoformat(
                        child.text.replace("Z", "+00:00")
                    )
                    gps_points.append((timestamp, lat, lon))

    if not gps_points:
        raise RuntimeError("No GPS points found in GPX file.")

    print(f"Loaded {len(gps_points)} GPS points.\n")
    return gps_points


gps_data = load_gpx(GPX_PATH)


def get_nearest_gps(target_time_utc):
    return min(gps_data, key=lambda x: abs(x[0] - target_time_utc))


# ---------- MAIN ----------
def run_background_processing():

    print("Initializing TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        print("FPS not detected. Using default 30 FPS.\n")
        fps = 30

    # Convert video IST start time to UTC
    video_start_utc = VIDEO_START_IST.replace(
        tzinfo=timezone(timedelta(hours=5, minutes=30))
    ).astimezone(timezone.utc)

    frame_number = 0
    last_upload_time = None

    print("Starting background processing...\n")

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number % (FRAME_SKIP + 1) != 0:
            continue

        # Calculate frame timestamp (UTC)
        video_time_sec = frame_number / fps
        frame_timestamp_utc = video_start_utc + timedelta(seconds=video_time_sec)

        # Prepare input
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]

        resized = cv2.resize(frame, (width, height))
        input_data = np.float32(resized) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        confidences = output_data[4]
        max_score = float(np.max(confidences))

        print(f"Frame {frame_number} | Score: {max_score:.3f}")

        if max_score > CONF_THRESHOLD:

            if last_upload_time:
                delta = (frame_timestamp_utc - last_upload_time).total_seconds()
                if delta < MIN_UPLOAD_INTERVAL:
                    continue

            gps_time, lat, lon = get_nearest_gps(frame_timestamp_utc)

            print("\n[DETECTED]")
            print("Frame time UTC:", frame_timestamp_utc)
            print("Nearest GPS time UTC:", gps_time)
            print("Mapped to:", lat, lon, "\n")

            db.collection(COLLECTION).add({
                "latitude": lat,
                "longitude": lon,
                "timestamp": datetime.utcnow(),
                "confidence": max_score
            })

            last_upload_time = frame_timestamp_utc

    cap.release()
    print("\nProcessing complete.")


if __name__ == "__main__":
    run_background_processing()
