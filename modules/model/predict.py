import os
import shutil
import tensorflow as tf

from tensorflow import keras
from keras import models
from keras.models import load_model
from PIL import Image
from params import *
from modules.model.zoning import x9_from_img
from skimage import data, io
from catboost import CatBoostClassifier, Pool

def predict_city_countryside(img):
    model = load_model(os.path.join(PROJECT_PATH,'models','20240826-115504.h5'))

    X = x9_from_img(img)
    result = model.predict(X)
    return result[0] #if result[0] == 1 => city, if ==0 => countryside
