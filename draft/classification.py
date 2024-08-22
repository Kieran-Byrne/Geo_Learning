import os
import shutil
import tensorflow as tf

from tensorflow import keras
from keras import models
from keras.models import load_model
from PIL import Image
from params import *

'''
Load a picture
Predict city or countryside
Copy that picture into a /data/classified_data/country/biome directory
Repeat
'''

model = load_model('20240821-175159.h5')

data_dir = os.path.join(PROJECT_PATH, "data")
tiny_dir = os.path.join(data_dir, 'tiny_dataset')
dest_dir = 'biome_data'

count = 0
for dir in os.listdir(tiny_dir):
    print(dir)
    country_directory = os.path.join(data_dir,dest_dir,dir)

    if os.path.isdir(country_directory):
        shutil.rmtree(country_directory)

    os.mkdir(country_directory)
    os.mkdir(os.path.join(country_directory,'city'))
    os.mkdir(os.path.join(country_directory,'countryside'))

    for file in os.listdir(os.path.join(tiny_dir,dir)):
        image = Image.open(os.path.join(tiny_dir,dir,file))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image.resize(66,153,3)
        image = tf.expand_dims(image, axis=0)
        result = model_city.predict(image)
        if result[0][0] < 0.5:
            shutil.copyfile(os.path.join(tiny_dir,dir,file), os.path.join(country_directory,'countryside',file))
        else:
            shutil.copyfile(os.path.join(tiny_dir,dir,file), os.path.join(country_directory,'city',file))
        count += 1
        print(file, ' has been transferred with a score of ', result[0][0],'. N°',count)
