from tensorflow.keras import Model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

#Imports to fix overfitting and 'dropout' from 'tensorflow.keras.layers'
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

#Imports for class weights as not all classes have the same amount of images
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

train_path = r"C:\Users\SkillsHub-Learner-09\Downloads\Mushroom classification.v1i.folder (1)\train"
validation_path = r"C:\Users\SkillsHub-Learner-09\Downloads\Mushroom classification.v1i.folder (1)\valid"

trainGenerator = ImageDataGenerator(preprocessing_function = preprocess_input).flow_from_directory(train_path, target_size=(224,224), batch_size=30)
validGenerator = ImageDataGenerator(preprocessing_function = preprocess_input).flow_from_directory(validation_path, target_size=(224,224), batch_size=30)


#Class weights
num_classes = 100
classes = trainGenerator.classes  # array of class indices for each image

# Compute weights
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)
print(class_weights_array)
# Convert to dictionary {class_index: weight}
class_weights = dict(enumerate(class_weights_array))
print(class_weights)
#Build the model
baseModel = MobileNetV2(weights='imagenet', include_top = False) #Chop first layer

#Add dropouts throughout training. Basically, turn off layers randomly so the model cannot rely on specific nodes
x = GlobalAveragePooling2D()(baseModel.output)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.6)(x)

predictLayer = Dense(100, activation='softmax')(x)

model = Model(inputs=baseModel.input, outputs=predictLayer)




#Freeze the pre-trained MobileNetV2 layers
#Up until the last layers we add 

#Freeze layers, to fine tune the later layers so we can keep the pre-trained generalized layers that recognize colour shape etc untouched
for layer in baseModel.layers[:-30]:
    layer.trainable = False

# Compile

optimizer = Adam(learning_rate = 0.00001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

# Callbacks
#End the training early if overfitting is detected. Detected if val loss is increasing and val accuracy is increasing (i.e. ai is memorizing dataset and noise)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)


#train
# Phase 1: freeze all layers, train only the head
for layer in baseModel.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(trainGenerator,
          validation_data=validGenerator,
          epochs=3,
          callbacks=[early_stop, lr_schedule])

# Phase 2: unfreeze last 10–15 layers, retrain with low LR
for layer in baseModel.layers[-15:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(trainGenerator,
          validation_data=validGenerator,
          epochs=8,
          callbacks=[early_stop, lr_schedule])


path_for_saved_model = "C:/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/dataset_for_model/mushroomV2.2.h5"
model.save(path_for_saved_model)