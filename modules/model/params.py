import os

PROJECT_PATH = os.environ.get('PROJECT_PATH')
UPLOAD_PATH = os.path.join(PROJECT_PATH, 'data/uploaded_images/temp.jpg')
CATBOOST_MODEL_PATH = os.path.join(PROJECT_PATH,'saved_models/trained_models/catboost_cropped.cbm')
COUNTRYSIDE_MODEL_PATH = os.path.join(PROJECT_PATH, 'model_1.h5')
CITY_MODEL_PATH = os.path.join(PROJECT_PATH, 'model_1.h5')

'''Model params'''
LEARNING_RATE = 1.0
KERNEL_SHAPE = (3,3)
MIN_COUNT = 800
MAX_COUNT = 1400
BATCH_SIZE = 32
IMAGE_RESIZE = (70,291) #CHANGE ONLY FLOAT (BETWEEN 0 AND 1)
EPOCHS = 50
L2_REGULARIZER_VALUE = 0.01
