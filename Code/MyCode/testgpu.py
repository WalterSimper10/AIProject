import tensorflow as tf
import os 
# Print TensorFlow version

tf.test.is_gpu_available()
train_path = r"/home/killsub-earner-09/datasets/mushroom/train"
valid_path = r"/home/killsub-earner-09/datasets/mushroom/valid"


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=32
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_path,
    image_size=(224, 224),
    batch_size=32
)

print("Train classes:", len(train_ds.class_names))
print("Valid classes:", len(valid_ds.class_names))
print("train names:", train_ds.class_names)
print("valid names:", valid_ds.class_names)

