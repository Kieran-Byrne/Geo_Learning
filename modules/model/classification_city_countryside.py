import os
import shutil
import tensorflow as tf

from tensorflow import keras
from keras import models
# from keras.models import load_model
from PIL import Image
from params import *
from modules.model.zoning import x9_from_img
from skimage import data, io
from catboost import CatBoostClassifier, Pool
'''
Load a picture
Predict city or countryside
Copy that picture into a /data/classified_data/country/biome directory
Repeat
'''


def city_countryside_classification():
    model = CatBoostClassifier()
    model.load_model(os.path.join(PROJECT_PATH,'modules','model','20240826-174502.cbm')
    data_dir = os.path.join(PROJECT_PATH,'data')
    full_dir =  os.path.join(data_dir,'compressed_dataset')
    biome_dir = os.path.join(data_dir,'biome_data_by_biome')
    countryside_dir = os.path.join(biome_dir,'countryside')
    city_dir = os.path.join(biome_dir,'city')

    if not os.path.isdir(biome_dir):
        os.mkdir(biome_dir)

    if os.path.isdir(countryside_dir):
        shutil.rmtree(countryside_dir)
    if os.path.isdir(city_dir):
        shutil.rmtree(city_dir)

    os.mkdir(countryside_dir)
    os.mkdir(city_dir)

    count = 0
    for dir in os.listdir(full_dir):
        print(dir)

        countryside_subdirectories = os.path.join(countryside_dir,dir)
        city_subdirectories = os.path.join(city_dir,dir)


        os.mkdir(countryside_subdirectories)
        os.mkdir(city_subdirectories)

        for file in os.listdir(os.path.join(full_dir,dir)):
            image = io.imread(os.path.join(full_dir,dir,file))
            X = x9_from_img(image)
            result = model.predict(X)
            if result[0] == 1:
                shutil.copyfile(os.path.join(full_dir,dir,file), os.path.join(city_subdirectories,file))
            else:
                shutil.copyfile(os.path.join(full_dir,dir,file), os.path.join(countryside_subdirectories,file))
            count += 1
            print(file, ' has been transferred with a score of ', result[0],'. N°',count)

city_countryside_classification()
