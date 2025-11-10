import os
from ultralytics import YOLO
import cv2
from pathlib import Path

def predict_germination(model_path='yolov8n.pt', image_dir='data/test_images', output_dir='predictions'):
    """
    Run YOLOv8 inference on germination test images.
    Args:
        model_path (str): Path to YOLO model (pretrained or fine-tuned)
        image_dir (str): Folder containing test images
        output_dir (str): Where to save prediction images
    """

    # Load model
    model = YOLO(model_path)

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Run inference
    results = model.predict(source=image_dir, save=True, project=output_dir, name='yolo_results', show=False)

    print(f"✅ Predictions saved in: {os.path.join(output_dir, 'yolo_results')}")
    print("✅ Done! You can view annotated images inside that folder.")

if __name__ == "__main__":
    predict_germination()
