from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
import numpy as np
from pathlib import Path
import cv2
import base64

from utils import cv2_to_base64, base64_to_cv2
from text_detector import TextDetector
from trocr import TrOCR

# Create instance of text detector
ocr = TextDetector(
        use_angle_cls=True, 
        lang='en', #lang='german',char=True,
        #det_model_dir="./fce_model",#"./pse_model",
        det=True,        # Enable text detection
        drop_score=.7,
        #det_db_thresh=0.6,
        det_db_box_thresh=0.4, #box threshold for text detection.
        #det_db_unclip_ratio=4,
        det_algorithm='DB++',
        use_gpu=False,
)

# Create instance of text recognizer
rec = TrOCR()


app = Flask(__name__)

@app.route('/api/process_image/', methods=['POST'])
def process_image():
    """
    Detect text in received image

    Returns:
        Response: A Flask response object containing a JSON with 
        the boundary boxes found in image
    """
    print('processing query ...')
    
    # gets base64 encoded image 
    img_str = request.json['image']
    # Convert the received base64 encoded string to an opencv image
    img = base64_to_cv2(img_str)
    
    # Obtain bounding boxes
    bounding_boxes, binary_image = ocr(img)
    
    # Crop images from bounding boxes and put them in a list
    cropped_images = ocr.crop_bounding_boxes(binary_image, bounding_boxes)
    
    # Extract the text from each cropped image 
    image_strings = []
    for i in cropped_images:
        image_strings.append(rec(i))
    
    
    base64_bimg = cv2_to_base64(binary_image)
        
    # Return bounding boxes and converted text
    print('ready to make response')
    response = {
        "bboxes": bounding_boxes,
        "binary_image":base64_bimg,
        "text":image_strings
    }
    return jsonify(response)



def main():
    app.run()


if __name__ == '__main__':
    main()