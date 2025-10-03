import pandas as pd
import tensorflow as tf
import os
import keras_cv



def load_annotations(csv_path, img_dir):
    df = pd.read_csv(csv_path)

    # Convert class column to categorical
    df["class"] = df["class"].astype("category")

    # Get mapping
    class_names = list(df["class"].cat.categories)
    print("Detected classes:", class_names)
    print("Number of classes:", len(class_names))

    grouped = df.groupby("filename")
    records = []
    for filename, group in grouped:
        boxes = group[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")
        labels = group["class"].cat.codes.values.astype("int32")  # integer IDs
        records.append((os.path.join(img_dir, filename), boxes, labels))

    return records, class_names
# -----------------------------
# Dataset Builder
# -----------------------------
def parse_example(img_path, boxes, labels, target_size=(512, 512)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_size)
    img.set_shape([target_size[0], target_size[1], 3])

    h, w = target_size
    boxes = boxes / [w, h, w, h]

    return img, {"boxes": boxes, "classes": labels}


def make_dataset(records, batch_size=4, shuffle=True):
    def gen():
        for path, boxes, labels in records:
            yield path, boxes, labels

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(records))

    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            {"images": [512, 512, 3]},
            {"boxes": [None, 4], "classes": [None]}
        ),
        padding_values=(
            {"images": tf.constant(0.0, dtype=tf.float32)},
            {"boxes": tf.constant(-1.0, dtype=tf.float32),
             "classes": tf.constant(-1, dtype=tf.int32)}
        )
    )

    return ds.prefetch(tf.data.AUTOTUNE)




train_records, class_names = load_annotations(
    r"C:\Users\SkillsHub-Learner-09\.vscode\VSCODE Projects\AIBackend\data\train\train.csv",
    r"C:\Users\SkillsHub-Learner-09\.vscode\VSCODE Projects\AIBackend\data\train\images"
)

val_records, _ = load_annotations(
    r"C:\Users\SkillsHub-Learner-09\.vscode\VSCODE Projects\AIBackend\data\valid\valid.csv",
    r"C:\Users\SkillsHub-Learner-09\.vscode\VSCODE Projects\AIBackend\data\valid\images"
)

num_classes = len(class_names)
print("Training with", num_classes, "classes")

train_ds = make_dataset(train_records, batch_size=4)
val_ds   = make_dataset(val_records, batch_size=4, shuffle=False)

for x, y in train_ds.take(1):
    print(x.keys())          # should show dict_keys(['images'])
    print(x["images"].shape) # (batch, 512, 512, 3)
    print(y["boxes"].shape)  # (batch, N, 4)
    print(y["classes"].shape)# (batch, N)


model = keras_cv.models.RetinaNet.from_preset(
    "resnet50_imagenet",
    num_classes=num_classes,
    bounding_box_format="xyxy"
)



# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    classification_loss="focal",
    box_loss="smoothl1"
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10   # adjust as needed
)

# -----------------------------
# Save Model
# -----------------------------
model.save("mushroom_detector")
