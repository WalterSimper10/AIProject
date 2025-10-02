import tensorflow as tf
from tensorflow.keras import layers
import os
import cv2

#Loading, parsing and getting rid of bad data
import filetype
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

    #Load data

#Load data from directory 'wild card search', data set API
#Build data pipeline
data = tf.keras.utils.image_dataset_from_directory('data', image_size=(640,640))
#Access data pipeline
data_iterator = data.as_numpy_iterator()
#Getting the batch of data
batch = data_iterator.next()
#^Images represented as numpy array


#limit tensorflow from using all the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)