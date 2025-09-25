import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Load training and test sets
train_ds = tf.keras.utils.image_dataset_from_directory(
    "mushroom-dataset-1/train",
    image_size=(640, 640),   # resize all images
    batch_size=32
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "mushroom-dataset-1/test",
    image_size=(640, 640),
    batch_size=32
)

# Normalize pixel values to [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./640)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
df = pd.read_csv("annotations.csv", header=None)
df.columns = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]

# Pick one row
row = df.iloc[0]

# Load the image
img = Image.open(r"C:\Users\SkillsHub-Learner-09\.vscode\VSCODE Projects\AIBackend\MyCode_project\MyCode\trainAI.py" + row["filename"])

# Plot
fig, ax = plt.subplots(1)
ax.imshow(img)

# Draw bounding box
rect = patches.Rectangle(
    (row["xmin"], row["ymin"]),
    row["xmax"] - row["xmin"],
    row["ymax"] - row["ymin"],
    linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(rect)

# Add label
plt.xlabel(row["class"])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)