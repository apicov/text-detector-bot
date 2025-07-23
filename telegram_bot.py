"""
Telegram bot for image text detection and recognition.
Receives images, sends them to the Flask API, and returns results to the user.
"""
#from telegram.ext import *
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import InputFile
from io import BytesIO
import numpy as np

from utils import cv2_to_base64, base64_to_cv2, bytes_to_cv2, cv2_to_bytes
import requests
#from skimage.io import imread
from PIL import Image

import requests
import json
import cv2
import os

from pathlib import Path
import datetime

# load telegram token
with open("private_data.json", "r") as read_file:
    data = json.load(read_file)

TOKEN = data['telegram_token']



def list_of_lists_to_string(list_of_lists):
    """
    Convert a list of lists to a JSON string.

    Args:
        list_of_lists (list): List of lists to convert.

    Returns:
        str: JSON string representation.
    """
    return json.dumps(list_of_lists)


def start(update, context):
    """
    Handler for the /start command. Sends a greeting message.

    Args:
        update (telegram.Update): Incoming update.
        context (telegram.ext.CallbackContext): Context for the update.
    """
    update.message.reply_text("Hola")
    
 
def handle_photo(update, context):
    """
    Handler for incoming photo messages. Sends the image to the OCR server and returns results.

    Args:
        update (telegram.Update): Incoming update.
        context (telegram.ext.CallbackContext): Context for the update.
    """
    # Check if the message contains an image
    #if update.message.photo:
    # Save the image to disk
    #file.download('image.jpg')
     
    # Retrieve the file ID of the highest resolution photo
    file_id = update.message.photo[-1].file_id
    # Get the file object
    file = context.bot.get_file(file_id)

    # Download the image as bytes (in ram byte stream)
    image_bytes = BytesIO()
    file.download(out=image_bytes)
    
    image = bytes_to_cv2(image_bytes)
    #image_bytes.seek(0)

    # Convert bytes to NumPy array
    #np_arr = np.frombuffer(image_bytes.read(), np.uint8)
    
    # Decode NumPy array to OpenCV image
    #image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

   
    update.message.reply_text("Processing query. Please wait...")

    #send request server
    base64_img = cv2_to_base64(image)
    #send request to clip searcher
    url = 'http://127.0.0.1:5000/api/process_image/'
    # send  text to the server
    payload = {"image": base64_img}
    response = requests.post(url, json=payload)
    
    #obtain bounding boxes of processed image
    bounding_boxes = response.json()['bboxes']
    # obtain list of converted text
    str_list = response.json()['text']

    #update.message.reply_text(list_of_lists_to_string(bounding_boxes))
    
    # Draw bounding boxes to image
    bboxes_img = cv2.polylines(image, 
                               np.array(bounding_boxes, dtype=np.int32), 
                               isClosed=True, 
                               color=(0, 255, 0), 
                               thickness=2)
    
         
    # Convert  image to a byte stream
    bboxes_bytes = cv2_to_bytes(bboxes_img)
    bboxes_bytes.seek(0)

    # send image stream to client
    context.bot.send_photo(chat_id=update.message.chat_id, photo=bboxes_bytes,caption='')
    
    
    b_img_base64 = response.json()['binary_image']
    b_img = base64_to_cv2(b_img_base64)
    # Convert  image to a byte stream
    b_img_bytes = cv2_to_bytes(b_img)
    b_img_bytes.seek(0)

    # send image stream to client
    context.bot.send_photo(chat_id=update.message.chat_id, photo=b_img_bytes,caption='')
    
    # join strings of list in one single string separated by newlines
    joined_text = '\n'.join(str_list)
    # Send text to client
    if joined_text != '':
        update.message.reply_text(joined_text)

    
    

updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))

#dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))


updater.start_polling()
updater.idle()