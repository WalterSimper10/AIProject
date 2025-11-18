#Code for testing the AI
#Takes image, resizes and preprocesses them before passing them to AI
#Converts image to usuable numPy array for AI, then pulls out AI predictions and prints result to screen

import os
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input # type: ignore
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

#Function to classify the image
def classify_image(imageFile):
    x = []

    #Load the image and resize it to input to mobileNetv2
    img = Image.open(imageFile)
    img.load()
    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    #Convert image to numPy array
    x = image.img_to_array(img)

    #Add a batch dimension (i.e there is ONE image in this batch)
    x=np.expand_dims(x,axis=0)

    #Preprocesses the shape of the image for the AI
    x=preprocess_input(x)

    #Pass image through the model and predict
    #Take out highest value and convert to 1
    pred = model.predict(x)
    categoryValue=np.argmax(pred, axis=1)

    #Pulls the value 1 from the array
    categoryValue = categoryValue[0]

    #Returns corresponding index for this 1 value as class name
    result = categories[categoryValue]
    return result

#Run function with desired image
imagePath = r"C:\Users\SkillsHub-Learner-09\.vscode\VSCODE Projects\AI\AIProject\Mushroom Classification.v1i.folder\Mushroom Classification.v1i.folder\Hygrocybe\015_DthsqGQxQHY_jpg.rf.7fa20cc0544ff3e63542060e88811027.jpg"
resultText = classify_image(imagePath)
print(resultText)

#Create GUI using cv2, display image and text
img = cv2.imread(imagePath)
img = cv2.putText(img, resultText, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

#Logic for GUI window, ensure 'open-cv python' is installed to run this
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()