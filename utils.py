from PIL import Image
import io
import base64
import random
import cv2
import numpy as np
from io import BytesIO

def cv2_to_PIL(image):
    """
    Converts an OpenCV image to a Pillow image.

    Parameters:
    image (numpy.ndarray): The OpenCV image to convert.

    Returns:
    Pillow image.
    """
    c_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(c_image)
    
    return pil_img
    
def cv2_to_bytes(image):
    """
    Converts an OpenCV image to a byte stream.

    Parameters:
    image (numpy.ndarray): The OpenCV image to convert.

    Returns:
    BytesIO: The byte stream of the image.
    """
    # Encode the image as a JPEG
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Could not encode image to JPEG format.")
    
    # Convert the encoded image to a byte stream
    image_bytes = BytesIO(encoded_image.tobytes())
    return image_bytes


def bytes_to_cv2(byte_stream):
    """
    Converts a byte stream back to an OpenCV image.

    Parameters:
    byte_stream (BytesIO or bytes): The byte stream containing the image data.

    Returns:
    numpy.ndarray: The OpenCV image array.
    """
    # Convert the byte stream to a numpy array
    byte_stream.seek(0)
    img_bytes =  np.frombuffer(byte_stream.read(), np.uint8)

    # Decode the numpy array into an OpenCV image
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return image


def cv2_to_base64(image):
    """
    Convert an OpenCV image (NumPy array) to base64 encoding.

    Args:
        image (numpy.ndarray): OpenCV image (BGR format).

    Returns:
        str: Base64 encoded image as a string.
    """
    # Convert image to JPEG format in memory
    _, buffer = cv2.imencode('.jpg', image)

    # Convert JPEG buffer to base64 string
    base64_encoded = base64.b64encode(buffer).decode('utf-8')

    return base64_encoded


def base64_to_cv2(base64_string):
    """
    Convert a base64 encoded string to an OpenCV image (NumPy array).

    Args:
        base64_string (str): Base64 encoded image as a string.

    Returns:
        numpy.ndarray: OpenCV image (BGR format).
    """
    # Decode base64 to bytes
    image_data = base64.b64decode(base64_string)

    # Convert bytes to numpy array
    image_array = np.frombuffer(image_data, np.uint8)

    # Decode image array to OpenCV format
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image


def pil_to_base64(pil_im):
    """
    Convert a PIL image to base64 encoding.

    Parameters:
        pil_im (PIL.Image.Image): PIL image object.

    Returns:
        str: Base64 encoded image as a string.
    """
    # Save PIL image to memory buffer as JPEG
    buffer = io.BytesIO()
    pil_im.save(buffer, format="JPEG")
    
    # Convert buffer to base64 string
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_str


def base64_to_pil(img_str):
    """
    Convert a base64 encoded string to a PIL image.

    Args:
        img_str (str): Base64 encoded image as a string.

    Returns:
        PIL.Image.Image: PIL image object.
    """
    # Decode base64 string to bytes
    img_data = base64.b64decode(img_str)
    
    # Convert bytes to PIL image
    pil_img = Image.open(io.BytesIO(img_data))
    
    return pil_img

