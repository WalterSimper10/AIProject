from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

train_path = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/dataset_for_model/train"
validation_path = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/dataset_for_model/validate"

trainGenerator = ImageDataGenerator(preprocessing_function = preprocess_input).flow_from_directory(train_path, target_size=(224,224), batch_size=30)
validGenerator = ImageDataGenerator(preprocessing_function = preprocess_input).flow_from_directory(validation_path, target_size=(224,224), batch_size=30)

#Build the model
baseModel = MobileNetV2(weights='imagenet', include_top = False) #Chop first layer

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)

predictLayer = Dense(9, activation='softmax')(x)

model = Model(inputs = baseModel.input, outputs=predictLayer)

print(model.summary())


#Freeze the pre-trained MobileNetV2 layers
#Up until the last layers we add 

for layer in model.layers[:-5]: 
    layer.trainable = False\
    
# Compile

epochs = 50
optimizer = Adam(learning_rate = 0.0001)

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])


#train
model.fit(trainGenerator, validation_data=validGenerator, epochs=epochs)

path_for_saved_model = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/dataset_for_model/mushroomV2.h5"
model.save(path_for_saved_model)