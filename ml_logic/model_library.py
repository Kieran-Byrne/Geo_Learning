from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers


def model_baseline(train_ds, val_ds):
    k_regularizer=regularizers.l2(0.001)

    model = models.Sequential()
    model.add(layers.Conv2D(64,(3, 3), activation='relu', input_shape=(66,153, 3))),
    model.add(layers.MaxPooling2D(pool_size=(2, 2))),
    model.add(layers.Conv2D(128, (3, 3), activation='relu')),
    # model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D(pool_size=(2,2))),
    model.add(layers.Conv2D(256, (3, 3), activation='relu')),
    model.add(layers.MaxPooling2D(pool_size=(2,2))),
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=k_regularizer))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dense(6, activation='softmax'))

    opt = optimizers.Adam(learning_rate=0.01)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    es = EarlyStopping(patience= 4, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=val_ds,
              epochs=50,
              batch_size=32,
              callbacks=es)

    '''
    Epoch 1/50
    132/132 [==============================] - 40s 297ms/step - loss: 5.6508 - accuracy: 0.2577 - val_loss: 1.7068 - val_accuracy: 0.2938
    Epoch 2/50
    132/132 [==============================] - 45s 342ms/step - loss: 1.6573 - accuracy: 0.3519 - val_loss: 1.7000 - val_accuracy: 0.3137
    Epoch 3/50
    132/132 [==============================] - 50s 379ms/step - loss: 1.6096 - accuracy: 0.3728 - val_loss: 1.5289 - val_accuracy: 0.4096
    Epoch 4/50
    132/132 [==============================] - 51s 383ms/step - loss: 1.5053 - accuracy: 0.4329 - val_loss: 1.5922 - val_accuracy: 0.3875
    Epoch 5/50
    132/132 [==============================] - 51s 385ms/step - loss: 1.4433 - accuracy: 0.4602 - val_loss: 1.5147 - val_accuracy: 0.4368
    Epoch 6/50
    132/132 [==============================] - 50s 380ms/step - loss: 1.3431 - accuracy: 0.5129 - val_loss: 1.5813 - val_accuracy: 0.4296
    Epoch 7/50
    132/132 [==============================] - 51s 384ms/step - loss: 1.2956 - accuracy: 0.5315 - val_loss: 1.5342 - val_accuracy: 0.4590
    Epoch 8/50
    132/132 [==============================] - 52s 390ms/step - loss: 1.1603 - accuracy: 0.5868 - val_loss: 1.5946 - val_accuracy: 0.4728
    Epoch 9/50
    132/132 [==============================] - 50s 380ms/step - loss: 1.0357 - accuracy: 0.6419 - val_loss: 1.8248 - val_accuracy: 0.4579
    '''
