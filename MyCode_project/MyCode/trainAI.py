import tensorflow as tf
import os
import cv2

#I need to use one of these to replace imghdr, but I don't know which one to use yet
import filetype

data_dir = 'data' #directory to dataset
image_exts = ['jpeg','jpg','bmp', 'png'] #literally just a list
#^set up to loop into it

#Pass through directory for 'data' folder and concatenate train + image to complete path
os.listdir(os.path.join(data_dir,'train', 'images')) 


#Checking for every folder in 'data' directory
for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        print(image)
        

#limit tensorflow from using all the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)