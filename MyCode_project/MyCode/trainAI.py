import tensorflow as tf
import os
import cv2

import magic
import filetype
import imghdr



fileName = os.listdir('mushroom-dataset-1')
print(fileName)

#limit tensorflow from using all the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)