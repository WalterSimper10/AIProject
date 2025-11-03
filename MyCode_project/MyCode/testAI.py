import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt


image_size = (180, 180)
batch_size = 128

model = keras.models.load_model(r'C:\Users\SkillsHub-Learner-09\.vscode\VSCODE Projects\AI\AIProject\save_at_1.keras')

img = keras.utils.load_img(r"C:\Users\SkillsHub-Learner-09\.vscode\VSCODE Projects\AI\AIProject\kagglecatsanddogs_5340\PetImages\Dog\12495.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
print(score)