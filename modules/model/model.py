import time
import pickle
from model.params import *
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt


def initialize_model(output_layer:int,
                     kernel_shape:tuple=(3,3),
                     regularizer_value:float=0.001
                     ):

    k_regularizer=regularizers.l2(0.001)

    model = models.Sequential()
    model.add(layers.Conv2D(64,kernel_shape, activation='relu', input_shape=(66,153, 3))),
    model.add(layers.MaxPooling2D(pool_size=(2, 2))),
    model.add(layers.Conv2D(128, kernel_shape, activation='relu')),
    # model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D(pool_size=(2,2))),
    # model.add(layers.Conv2D(256, kernel_shape, activation='relu')),
    # model.add(layers.MaxPooling2D(pool_size=(2,2))),
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=k_regularizer))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_layer, activation='softmax'))

    return model

def compile_model(model,
                  learning_rate:float=0.01):

    opt = optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def train_model(model,
                train_ds,
                val_ds,
                batch_size:int=32,
                epochs:int=50
                ):

    model_name = "model_1"

    modelCheckpoint = ModelCheckpoint(f"{model_name}.h5", monitor="accuracy", verbose=1, save_best_only=True)
    LRreducer = ReduceLROnPlateau(monitor="accuracy", factor = 0.2, patience=3, verbose=1, min_lr=0)
    EarlyStopper = EarlyStopping(monitor='accuracy', patience=4, verbose=1, restore_best_weights=True)

    # es = EarlyStopping(patience= 4, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=val_ds,
              epochs=50,
              batch_size=32,
              callbacks=[modelCheckpoint, EarlyStopper, LRreducer])




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
        params_path = os.path.join(PROJECT_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)
    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(PROJECT_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)



def save_model(model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(PROJECT_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)
    return None

def plot_training(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
