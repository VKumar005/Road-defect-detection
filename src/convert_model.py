from ultralytics import YOLO

# Path to trained model
MODEL_PATH = "pothole_detector_ref/yolov8n_train/weights/best.pt"

# Path to your training data.yaml
DATA_YAML = "C:/pothole/Reference_data/data.yaml"  # adjust if needed

print("Loading model...")
model = YOLO(MODEL_PATH)

print("Exporting INT8 TFLite model for Raspberry Pi...")

model.export(
    format="tflite",
    imgsz=320,
    int8=True,
    data=DATA_YAML,
    keras=False
)

print("Export complete.")
