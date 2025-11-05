#Organizes dataset
#Takes a folder in the following format:
#Dataset|
#       /Dataset|
#               /classes (i.e folders)
#And refactors them into:
#Dataset|
#       /dataset_for_model|
#                         /train
#                         /validate
#Filters out empty files as a quick check
#Splits data between train and validate by itself, creating 85% train and 15% validate
#**NOTE** modify 'splitsize' variable for desired split
#Recommended 0.80-0.85~
#Max value <.95~
#DO NOT exceed 1

import os
import random
import shutil

#Train on 85% of the data base and validate on the other 15%
splitsize = .85

#Array of all classes (mushroom genus')
categories = []

#Set root of dataset (C:... Dataset/Dataset)
source_folder = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects\AI\AIProject/Mushroom Classification.v1i.folder/Mushroom Classification.v1i.folder"
folders = os.listdir(source_folder)

#Loop and append categories to categories array
for subfolder in folders:
    if os.path.isdir(source_folder + "/" + subfolder):
        categories.append(subfolder)

categories.sort()

#Create a target folder (Root of model)
target_folder = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/dataset_for_model"
existDataSetPath = os.path.exists(target_folder)
if existDataSetPath == False:
    os.mkdir(target_folder)


#Create a function to split the data into train and validate
def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):

    #append all files to files array
    files=[]

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is length 0, ignore...")  
    print(len(files))  

    #Converts value to int and splits list based on split size
    trainingLength = int(len(files) * SPLIT_SIZE)

    #Shuffle dataset to reduce sample bias
    shuffleSet = random.sample(files, len(files))

    #Add training data to training set
    trainingSet = shuffleSet[0:trainingLength]

    #Add remaining data 
    validSet = shuffleSet[trainingLength:]

    #Copy the train images
    for filename in trainingSet:
        thisFile = SOURCE + filename
        destination = TRAINING + filename
        shutil.copyfile(thisFile, destination)

    #Copy validation images
    for filename in validSet:
        thisFile = SOURCE + filename
        destination = VALIDATION + filename
        shutil.copyfile(thisFile, destination)

#Create variable that contains target folder path for train and validate
trainPath = target_folder + "/train"
validatePath = target_folder + "/validate"

#Create target folders
exitsDataSetPth = os.path.exists(trainPath)
if not(exitsDataSetPth):
    os.mkdir(trainPath)

#Checks both to makes sure they dont exist first
exitsDataSetPth = os.path.exists(validatePath)
if exitsDataSetPth==False:
    os.mkdir(validatePath)

#Run the function for each of the folders
for category in categories:
    trainDestPath = trainPath + "/" + category
    validateDestPath = validatePath + "/" + category

    if os.path.exists(trainDestPath)==False:
        os.mkdir(trainDestPath)
    if os.path.exists(validateDestPath)==False:
        os.mkdir(validateDestPath)

    sourcePath = source_folder + "/" + category + "/"
    trainDestPath = trainDestPath + "/" 
    validateDestPath = validateDestPath + "/"

    print("copy from: " + sourcePath + " to: " + trainDestPath + " and " + validateDestPath)

    split_data(sourcePath, trainDestPath, validateDestPath, splitsize)





