from paddleocr import PaddleOCR
import time

ocr = PaddleOCR(
    lang='en',
    ocr_version="PP-OCRv5",
    device="gpu:0",
)

def perform_ocr_inference(input_file):
    start_time = time.time()
    result = ocr.predict(input=input_file,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)
    end_time = time.time()
    inference_time = end_time - start_time
    
    return result, inference_time

if __name__ == "__main__":
    input_file_set = "./detected_objects/card_set_20250621_145903_528.jpg"
    result_set, inference_time_set = perform_ocr_inference(input_file_set)
    input_file_num = "./detected_objects/card_num_20250621_145903_528.jpg"
    result_num, inference_time_num = perform_ocr_inference(input_file_num)
    
    print(f"Inference time: {inference_time_set:.3f} seconds")
    print(f"Inference time: {inference_time_num:.3f} seconds")

    for page_result in result_set:
        if 'rec_texts' in page_result:
            for text in page_result['rec_texts']:
                print(text[:3])
    
    for page_result in result_num:
        if 'rec_texts' in page_result:
            for text in page_result['rec_texts']:
                print(text)