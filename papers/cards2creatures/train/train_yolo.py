"""Train YOLO to detect card_num and card_set regions."""

from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    results = model.train(data="dataset/data.yaml", epochs=30, imgsz=640)

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best_path))
    trained.export(format="onnx")
    print(f"Trained weights: {best_path}")
