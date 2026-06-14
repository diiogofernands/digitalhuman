from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime
from detect_regions import detect_and_extract_objects
from read_text import perform_ocr_inference
import uvicorn
import traceback

app = FastAPI()

class ImageRequest(BaseModel):
    image_base64: str

class InferenceResponse(BaseModel):
    card_set: str
    card_num: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: ImageRequest):
    temp_files = []
    saved_images = []

    # Create output directory for saving images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"api_outputs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)

        # Create temporary file for input image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_input:
            temp_input.write(image_data)
            temp_input_path = temp_input.name
            temp_files.append(temp_input_path)

        # Save the received image
        received_image_path = os.path.join(output_dir, "received_image.png")
        with open(received_image_path, 'wb') as f:
            f.write(image_data)
        saved_images.append(received_image_path)

        # Create temporary directory for detected objects
        temp_output_dir = tempfile.mkdtemp()
        temp_files.append(temp_output_dir)

        # Run YOLO detection and extract bboxes
        model_path = "final_models/best.pt"
        results, saved_paths = detect_and_extract_objects(
            model_path=model_path,
            image_path=temp_input_path,
            output_dir=temp_output_dir,
            save=False,
            conf=0.3
        )

        # Initialize results
        card_set = ""
        card_num = ""

        # Process each detected object with OCR
        for bbox_path in saved_paths:
            temp_files.append(bbox_path)

            # Save the detected object image
            filename = os.path.basename(bbox_path)
            saved_bbox_path = os.path.join(output_dir, filename)
            shutil.copy2(bbox_path, saved_bbox_path)
            saved_images.append(saved_bbox_path)

            # Run OCR on the bbox image
            ocr_result, inf_time = perform_ocr_inference(bbox_path)
            print(f"Inference time for {bbox_path}: {inf_time:.4f} seconds - {inf_time*1000:.1f} ms")

            # Extract text from OCR result
            extracted_text = ""
            for page_result in ocr_result:
                if 'rec_texts' in page_result:
                    for text in page_result['rec_texts']:
                        extracted_text += text + " "

            # Determine if it's card_set or card_num based on filename
            filename = os.path.basename(bbox_path)
            if "card_set" in filename:
                card_set = extracted_text.strip()
            elif "card_num" in filename:
                card_num = extracted_text.strip()

        return InferenceResponse(card_set=card_set, card_num=card_num)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("❌ Exception during inference:\n", traceback_str)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                elif os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete {temp_file}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")