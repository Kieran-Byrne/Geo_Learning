import time
import pickle

from params import *
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

import os

PROJECT_PATH = os.getcwd()

def initialize_model(output_layer:int,
                     kernel_shape:tuple=(3,5),
                     regularizer_value:float=0.001
                     ):


    model = models.Sequential()

    # Bloc Convolutionnel 1
    model.add(layers.Conv2D(64,kernel_shape, activation='relu', input_shape=(662,1536, 3)))
    #model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(66, 153, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    # Bloc Convolutionnel 2
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))

    # Bloc Convolutionnel 3
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Bloc Convolutionnel 4
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Bloc Résiduel (Inspiré de ResNet)
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())

    # Ajout de la connexion de raccourci (skip connection)
    #model.add(layers.Add())

    # Bloc final
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_layer, activation='softmax'))

    return model


def compile_model(model,
                  learning_rate:float=0.01):

    opt = optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['Accuracy'])

    return model

def train_model(model,
                train_ds,
                val_ds,
                batch_size:int=16,
                epochs:int=200
                ):

    model_name = "model_1"

    modelCheckpoint = ModelCheckpoint(f"{model_name}.h5", monitor="val_Accuracy", verbose=1, save_best_only=True, mode='max')
    #LRreducer = ReduceLROnPlateau(monitor="val_Accuracy", factor = 1.0, patience=5, verbose=1, min_lr=0)
    EarlyStopper = EarlyStopping(monitor='val_Accuracy', patience=10, verbose=1, restore_best_weights=True, mode='max')

    history = model.fit(train_ds, validation_data=val_ds,
              epochs=200,
              batch_size=16,
              callbacks=[modelCheckpoint, EarlyStopper])

    return model, history, modelCheckpoint


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Save params locally
    if params is not None:
        params_path = os.path.join(PROJECT_PATH, "../../params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)
    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(PROJECT_PATH, "../../metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)



def save_model(model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(PROJECT_PATH, "../../models", f"{timestamp}.h5")
    model.save(model_path)
    return None

def plot_training(history):
    # Plot accuracy
    if 'Accuracy' in history.history:
        plt.plot(history.history['Accuracy'])
    if 'val_Accuracy' in history.history:
        plt.plot(history.history['val_Accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Plot loss
    if 'loss' in history.history:
        plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
