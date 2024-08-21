import os
import shutil
import numpy as np
import pandas as pd

from ml_logic.params import *

from tensorflow.keras.utils import image_dataset_from_directory

data_dir = os.path.join(PROJECT_PATH, 'data')
raw_data_path= os.path.join(data_dir, 'compressed_dataset')

def load_dataset(min_count:int =0,
                 max_count:int = 0,
                 image_resize:tuple =(66,153)
                 ):

    ''' Create the directory with the wanted data.'''
    dest_dir = 'temp_data'
    dest_path = os.path.join(data_dir, dest_dir)

    if os.path.isdir(dest_path):
        shutil.rmtree(dest_path) #remove old temporary directory if existent

    os.mkdir(dest_path) #create a temporary directory for the data used for the dataset

    for dir in os.listdir(raw_data_path):
        if min_count < len(os.listdir(os.path.join(raw_data_path, dir))) < max_count:
            shutil.copytree(os.path.join(raw_data_path, dir), os.path.join(data_dir, dest_dir, dir))

    ''' Create the BatchDataset from the directory.'''

    train_ds = image_dataset_from_directory(
        dest_path,
        labels='inferred',
        label_mode='int',
        seed=123,
        image_size=image_resize,
        validation_split=0.3,
        subset='training'
        )

    val_ds = image_dataset_from_directory(
        dest_path,
        labels='inferred',
        label_mode='int',
        seed=123,
        image_size=(66,153),
        validation_split=0.3,
        subset = 'validation'
        )

    '''Get output layer shape for the model'''

    classes_number = len(os.listdir(os.path.join(data_dir, dest_dir)))

    return train_ds, val_ds, classes_number
