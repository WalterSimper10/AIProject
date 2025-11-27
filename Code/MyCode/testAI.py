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
categories = os.listdir(r"/home/killsub-earner-09/datasets/mushroom/train")
categories.sort()

#Load the model
path_for_saved_model = r"/mnt/c/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/models/finalmodel.keras"
model = tf.keras.models.load_model(path_for_saved_model)

#Function to classify the image
def classify_image(imageFile, top_k=5):
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
    pred = model.predict(x)[0]

    # Get top-k indices sorted by confidence
    top_indices = pred.argsort()[-top_k:][::-1]

    # Map indices to class names and confidence
    results = [(categories[i], float(pred[i])) for i in top_indices]

    return results


#Run function with desired image
imagePath = r"/mnt/c/Users/SkillsHub-Learner-09/Downloads/IMG_823.JPG"
top_predictions = classify_image(imagePath, top_k=5)

# Print results
print("Top 5 predictions:")
for cls, prob in top_predictions:
    print(f"{cls}: {prob*100:.2f}%")

#Create Cv2 GUI
img = cv2.imread(imagePath)
y0, dy = 50, 30
for i, (cls, prob) in enumerate(top_predictions):
    text = f"{cls}: {prob*100:.1f}%"
    y = y0 + i*dy
    img = cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

cv2.imshow("Predictions", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
