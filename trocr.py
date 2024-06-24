from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import torch
from utils import cv2_to_PIL


class TrOCR:
    def __init__(self):
        # Check if CUDA is available and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
        
    def __call__(self, img):
        
        # model needs pil image as input
        pil_img = cv2_to_PIL(img)
        
        # image preprocessing
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)
        
        # Inference
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        return generated_text