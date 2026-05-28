# train.py - Run on your desktop/laptop or cloud VM

from ultralytics import YOLO
import os

# --- Configuration ---
# Start from the pre-trained 'nano' model for efficiency
MODEL_TO_USE = 'yolov8n.pt' 

DATA_YAML_PATH = 'C:/pothole/Reference_data/data.yaml' 

# Training parameters
TRAINING_EPOCHS = 100  # Increase for better accuracy, maybe 100-200
IMAGE_SIZE = 640
PROJECT_NAME = 'pothole_detector_ref'
RUN_NAME = 'yolov8n_train'
# ---------------------

print(f"Starting training with model: {MODEL_TO_USE}")
print(f"Using dataset config: {DATA_YAML_PATH}")

def main():
    # Check if dataset yaml exists
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: Dataset YAML file not found at '{DATA_YAML_PATH}'")
        print("Please ensure the path is correct and the dataset is unzipped.")
        return
        
    try:
        # Load the pre-trained model
        model = YOLO(MODEL_TO_USE)

        # Start the training process
        print(f"Training for {TRAINING_EPOCHS} epochs...")
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=TRAINING_EPOCHS,
            imgsz=IMAGE_SIZE,
            project=PROJECT_NAME,
            name=RUN_NAME,
            exist_ok=True # Overwrite previous run if name exists
        )
        
        print("\n--- Training Complete! ---")
        best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'best.pt')
        print(f"Best model saved to: {best_model_path}")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == '__main__':
    main()
