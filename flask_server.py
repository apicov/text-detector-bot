from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
import numpy as np
from pathlib import Path
import cv2
import base64

from utils import cv2_to_base64, base64_to_cv2
from text_detector import TextDetector

# Create instance of text detector
ocr = TextDetector(
        use_angle_cls=True, 
        lang='en', #lang='german',char=True,
        #det_model_dir="./fce_model",#"./pse_model",
        det=True,        # Enable text detection
        drop_score=.6,
        #det_db_thresh=0.1,
        det_db_box_thresh=0.4,
        #det_db_unclip_ratio=4,
        det_algorithm='DB++',
        use_gpu=False,
)


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
    
    img_str = request.json['image']
    
    # Convert the received base64 encoded string to a PIL image
    img = base64_to_cv2(img_str)
    
    # Obtain bounding boxes
    bounding_boxes = ocr(img)
        
    # Return bounding boxes
    print('ready to make response')
    response = {
        "bboxes": bounding_boxes
    }
    return jsonify(response)



def main():
    app.run()


if __name__ == '__main__':
    main()