import os

PROJECT_PATH = os.environ.get("PROJECT_PATH")

'''Model params'''
LEARNING_RATE = 1.0
KERNEL_SHAPE = (3,3)
MIN_COUNT = 800
MAX_COUNT = 1400
BATCH_SIZE = 32
IMAGE_RESIZE = (662,1536) * 0.1 #CHANGE ONLY FLOAT (BETWEEN 0 AND 1)
EPOCHS = 50
L2_REGULARIZER_VALUE = 0.01