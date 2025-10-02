import os
import cv2
import filetype

data_dir = 'data' #directory to dataset
image_exts = ['jpeg','jpg','bmp', 'png'] #literally just a list
#^set up to loop into it
        #Removing dodgy images
#Checking for every folder in 'data' directory
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class) 
    #Checking every sub-folder in the data directory
    for image in os.listdir(os.path.join(class_path)):
        image_folder = os.path.join(class_path, image)
        if not os.path.isdir(image_folder): #Skip the annotations document
            continue

        #Opening all the image files
        for image_path in os.listdir(os.path.join(image_folder)):
            full_path = os.path.join(image_folder, image_path)
            try:    
                img = cv2.imread(full_path)
                kind = filetype.guess(full_path)
                tip = kind.extension if kind else None
                #Small change from imghdr, but basically is just checks the object to make sure it is returning something

                #If the type of image does not match any of the type in the list
                if tip not in image_exts:
                 print('Image not in ext list {}'.format(image_path))
                 os.remove(full_path) 
            except Exception as e:
                print('Issue with image {}'.format(image_path))
                os.remove(full_path) 
                #Shakaboom

#Load data from directory 'wild card search', data set API
#Build data pipeline
data = tf.keras.utils.image_dataset_from_directory('data', image_size=(640,640))
#Access data pipeline
data_iterator = data.as_numpy_iterator()
#Getting the batch of data
batch = data_iterator.next()
#^Images represented as numpy array
