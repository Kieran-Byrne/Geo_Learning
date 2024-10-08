
import shutil
import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt


from modules.model.params import *

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers
from modules.model.dataset import load_dataset
from modules.model.model import initialize_model, compile_model, train_model, save_model, save_results, plot_training

def launch_overnight_training(min_count:int =0,
                              max_count:int = 0,
                              image_resize:tuple =(66,153),
                              kernel_shape:tuple=(3,3),
                              regularizer_value:float=0.001,
                              learning_rate:float=0.001,
                              batch_size:int=32,
                              epochs:int=50):

    train_ds, val_ds, classes_number = load_dataset(min_count=min_count,
                                    max_count=max_count,
                                    image_resize=image_resize)

    model = initialize_model(output_layer=classes_number,
                             kernel_shape=kernel_shape,
                             regularizer_value=regularizer_value,
                             )

    model = compile_model(model,
                          learning_rate=learning_rate)

    model, history, modelCheckpoint = train_model(model,
                                 train_ds=train_ds,
                                 val_ds=val_ds,
                                 batch_size=batch_size,
                                 epochs=epochs)

    save_results(history.params, history.history)

    save_model(model)

    plot_training(history)


launch_overnight_training(min_count=MIN_COUNT,
                          max_count=MAX_COUNT,
                          image_resize=IMAGE_RESIZE,
                          kernel_shape=KERNEL_SHAPE,
                          learning_rate=LEARNING_RATE)
