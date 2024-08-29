import os
from skimage import io
import cv2
from modules.model.params import *

def crop_resize_image(image_path):
    """ this function crops and resizes the image
    to the shape given in the params.py file"""
    img = io.imread(UPLOAD_PATH)
    img_cropped = img[290:752,:]
    img_resized = cv2.resize(img_cropped,(1456,350))
    return img_resized
