from tensorflow.keras import Model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import mixed_precision # type: ignore
import keras


#Imports to fix overfitting and 'dropout' from 'tensorflow.keras.layers'
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

#Imports for class weights as not all classes have the same amount of images
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf

train_path = r"/home/killsub-earner-09/datasets/mushroom/train"
validation_path = r"/home/killsub-earner-09/datasets/mushroom/valid"

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

mixed_precision.set_global_policy("mixed_float16")

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=30
)

raw_valid_ds = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    image_size=(224, 224),
    batch_size=30
)

# Get class names before transformations
classes = raw_train_ds.class_names
num_classes = len(classes)

# apply preprocessing and pipeline optimizations
train_ds = raw_train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
valid_ds = raw_valid_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# Compute weights
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)

# Convert to dictionary {class_index: weight}
class_weights = dict(enumerate(class_weights_array))

#Build the model
baseModel = MobileNetV2(weights='imagenet', include_top = False) #Chop first layer

#Add dropouts throughout training. Basically, turn off layers randomly so the model cannot rely on specific nodes
x = GlobalAveragePooling2D()(baseModel.output)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.2)(x)

predictLayer = Dense(100, activation='softmax')(x)

model = Model(inputs=baseModel.input, outputs=predictLayer)




#Freeze the pre-trained MobileNetV2 layers
#Up until the last layers we add 

#Freeze layers, to fine tune the later layers so we can keep the pre-trained generalized layers that recognize colour shape etc untouched
for layer in baseModel.layers:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False


# Compile

optimizer = Adam(learning_rate = 0.00001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])


# Callbacks
#End the training early if overfitting is detected. Detected if val loss is increasing and val accuracy is increasing (i.e. ai is memorizing dataset and noise)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

#train
# Phase 1: freeze all layers, train only the head
for layer in baseModel.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds,
          validation_data=valid_ds,
          epochs=25,
          callbacks=[early_stop, lr_schedule])

# Phase 2: unfreeze last 10–15 layers, retrain with low LR
for layer in baseModel.layers[-25:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds,
          validation_data=valid_ds,
          epochs=75,
          callbacks=[early_stop, lr_schedule])


path_for_saved_model = "/mnt/c/Users/SkillsHub-Learner-09/.vscode/VSCODE Projects/AI/AIProject/Mushroom Classification.v1i.folder/models/finalmodel.keras"
model.save(path_for_saved_model)