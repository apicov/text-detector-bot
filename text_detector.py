import os
import sys
import requests
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np

# add paddleocr folder to python path
sys.path.insert(0,'./PaddleOCR')
from paddleocr import PaddleOCR, draw_ocr



class TextDetector:
    '''ss
    '''
    def __init__(self, **kwargs):
        
        # optical character recognition using paddleocr library
        self.ocr = PaddleOCR(rec=False, **kwargs)
        
        
    def _preprocess_image(img):
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Apply thresholding
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_OTSU)
        
        return binary_image

    def __call__(self, img)
        ''' 
        Applies character detection to image.

        Parameters:
        img (numpy array): opencv image

        Returns:
        list of bounding boxes 
        '''
        result = self.ocr.ocr( self._preprocess_image(img), rec=False,)
        bounding_boxes = result[0]
        
        return bounding_boxes
    