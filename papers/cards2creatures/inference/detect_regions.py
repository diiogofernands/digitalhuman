from ultralytics import YOLO
import cv2
import os
from datetime import datetime
from PIL import Image
import numpy as np

def run_yolo_detection(model_path, image_path, save=True, imgsz=576, conf=0.3):
    model = YOLO(model_path)
    results = model.predict(image_path, save=save, imgsz=imgsz, conf=conf)
    return results

def extract_and_save_bboxes(results, original_image_path, output_dir="detected_objects"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"Could not load image from {original_image_path}")
    
    saved_paths = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                cropped_image = original_image[y1:y2, x1:x2]
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
                filename = f"{class_name}_{timestamp}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                cv2.imwrite(filepath, cropped_image)
                
                saved_paths.append(filepath)
                
                print(f"Saved {class_name} (conf: {confidence:.2f}) to {filepath}")
    
    return saved_paths

def detect_and_extract_objects(model_path, image_path, output_dir="detected_objects", save=True, imgsz=576, conf=0.3):
    results = run_yolo_detection(model_path, image_path, save, imgsz, conf)
    saved_paths = extract_and_save_bboxes(results, image_path, output_dir)
    
    return results, saved_paths

if __name__ == "__main__":
    model_path = "best.onnx"
    image_path = "togepi.jpeg"
    
    results, saved_paths = detect_and_extract_objects(model_path, image_path)

    print(f"\nSaved {len(saved_paths)} detected objects:")
    for path in saved_paths:
        print(f"  - {path}")