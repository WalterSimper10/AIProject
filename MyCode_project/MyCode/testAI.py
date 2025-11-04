import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

#List of classes
categories = os.listdir("C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/dataset_for_model/train")
categories.sort()

#Load the model
path_for_saved_model = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/dataset_for_model/mushroomV2.keras"
model = tf.keras.models.load_model(path_for_saved_model)


def classify_image(imageFile):
    x = []

    img = Image.open(imageFile)
    img.load()
    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    x = image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    print(x.shape)

    pred = model.predict(x)
    categoryValue=np.argmax(pred, axis=1)
    print(categoryValue)

    categoryValue = categoryValue[0]
    print(categoryValue)

    result = categories[categoryValue]

    return result

imagePath = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/Mushroom Classification.v1i.folder/Entoloma/100_DOEuA90u0n4_jpg.rf.d2d49341b664db177b3986a73b4f9832.jpg"
resultText = classify_image(imagePath)
print(resultText)

img = cv2.imread(imagePath)
img = cv2.putText(img, resultText, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()