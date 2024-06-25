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
    '''
    A class for detecting text in an image.
    '''
    def __init__(self, **kwargs):
        ''' 
        Initializes ocr detector 

        Parameters:
        kwargs(see PaddleOCR docs) 
        '''
        
        # optical character recognition using paddleocr library
        self.ocr = PaddleOCR(rec=False, **kwargs)
        
        self.binary_image = None
        
        
    def _preprocess_image(self, img):
        ''' 
        Applies Gaussianblur and binarizes input image

        Parameters:
        img (numpy array): opencv image

        Returns:
        black and white opencv image 
        '''
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        
        # Apply thresholding
        #_, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_OTSU)
        
        binary_img = cv2.adaptiveThreshold(gray_img, 
                                               255, # maximum value
                                               cv2.ADAPTIVE_THRESH_MEAN_C, 
                                               cv2.THRESH_BINARY,
                                               11, # size of block
                                               20)  # constant subtracted from the mean
        
        # The aim is to dilate black areas
        # erosion is used becaous eit works on white areas
        # Create a kernel
        kernel = np.ones((3, 3), np.uint8)
        # Apply dilation
        dilated_image = cv2.erode(binary_img, kernel, iterations=1)
        
        return dilated_image

    def __call__(self, img):
        ''' 
        Applies character detection to image.

        Parameters:
        img (numpy array): opencv image

        Returns:
        list of bounding boxes 
        '''
        binary_image = self._preprocess_image(img)
        result = self.ocr.ocr(binary_image , rec=False,)
        bounding_boxes =  [] if result[0] is None else result[0]
        
        
        
        return bounding_boxes, binary_image
    
    def draw(self, img, bboxes):
        ''' 
        Draws bounding boxes to image.

        Parameters:
        img (numpy array): opencv image
        bboxes (numpy array): list of bounding boxes

        Returns:
        image with bounding boxes
        '''
        bounding_boxes_img = img.copy()
        cv2.polylines(bounding_boxes_img, np.array(bboxes, dtype=np.int32), isClosed=True, color=(0, 255, 0), thickness=2)
        
        return cv2.cvtColor(bounding_boxes_img, cv2.COLOR_BGR2RGB)
    
    def crop_bounding_boxes(self, img, bboxes):
        ''' 
        Crops bounding boxes from image.

        Parameters:
        img (numpy array): opencv image
        bboxes (numpy array): list of bounding boxes

        Returns:
        list of cropped images
        '''
        cropped_images = []

        for box in bboxes:
            box = np.array(box, dtype=np.int32)
            # Get the bounding rectangle (minimum area rectangle) from the polygonal points
            x, y, w, h = cv2.boundingRect(box)
            # Crop the region of interest (ROI) from the image
            cropped_image = img[y:y+h, x:x+w]
            # Append the cropped image to the list
            cropped_images.append(cropped_image)
            
        return cropped_images
        
    