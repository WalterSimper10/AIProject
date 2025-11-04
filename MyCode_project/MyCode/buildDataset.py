import os
import random
import shutil
#Train on 85% of the data base and validate on the other 15%
splitsize = .85

#Array of each of the classes (i.e. mushroom remus')
categories = []

source_folder = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects\AI\AIProject/Mushroom Classification.v1i.folder/Mushroom Classification.v1i.folder"
folders = os.listdir(source_folder)
print(folders)

for subfolder in folders:
    if os.path.isdir(source_folder + "/" + subfolder):
        categories.append(subfolder)

categories.sort()
print(categories)


#Create a target folder (Root of model dataset)
target_folder = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/dataset_for_model"
existDataSetPath = os.path.exists(target_folder)
if existDataSetPath == False:
    os.mkdir(target_folder)


#Create a function to split the data into train and validate
def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files=[]

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is length 0, ignore...")  
    print(len(files))  


    trainingLength = int(len(files) * SPLIT_SIZE)
    shuffleSet = random.sample(files, len(files))
    trainingSet = shuffleSet[0:trainingLength]
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

trainPath = target_folder + "/train"
print(trainPath)
validatePath = target_folder + "/validate"

#Create target folders
exitsDataSetPth = os.path.exists(trainPath)
print(exitsDataSetPth)
if not(exitsDataSetPth):
    print("inside")
    os.mkdir(trainPath)

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





