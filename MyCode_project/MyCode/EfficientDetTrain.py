import pandas as pd
import tensorflow as tf
import os
import keras_cv



def load_annotations(csv_path, img_dir, class_names=None):
    df = pd.read_csv(csv_path)

    # Drop invalid rows
    df = df[df["class"].notna()]
    df = df[~df["class"].isin(["-------", "--------------------"])]

    if class_names is None:
        df["class"] = df["class"].astype("category")
        class_names = list(df["class"].cat.categories)
    else:
        df["class"] = pd.Categorical(df["class"], categories=class_names)

    labels = df["class"].cat.codes

    grouped = df.groupby("filename")
    records = []
    for filename, group in grouped:
        boxes = group[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")
        labels = group["class"].cat.codes.values.astype("int32")
        records.append((os.path.join(img_dir, filename), boxes, labels))
        print("Max label value:", labels.max())

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

    return img, {"boxes": boxes, "classes": labels}  # Return image tensor directly


def wrap_input(images, labels):
    return {"images": images}, labels



def make_dataset(records, batch_size=4, shuffle=True):
    # Separate your lists
    img_paths = [r[0] for r in records]
    boxes_list = [r[1] for r in records]
    labels_list = [r[2] for r in records]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, boxes_list, labels_list))
    if shuffle:
        ds = ds.shuffle(len(records))

    def _parse(img_path, boxes, labels):
        return parse_example(img_path, boxes, labels)

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(wrap_input, num_parallel_calls=tf.data.AUTOTUNE)  # Wrap input here

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




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
val_ds = make_dataset(val_records, batch_size=4, shuffle=False)

for batch in train_ds.take(1):
    boxes = batch[1]["boxes"]
    classes = batch[1]["classes"]
    print("Boxes:", boxes)
    print("Classes:", classes)
    print("Unique labels:", tf.unique(tf.concat(classes.flat_values, axis=0))[0])

for x, y in train_ds.take(1):
    print("x keys:", x.keys())  # ✅ Should be dict_keys(['images'])
    print("x['images'].shape:", x["images"].shape)
    print("y['boxes'].shape:", y["boxes"].shape)
    print("y['classes'].shape:", y["classes"].shape)

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
    epochs=5   # adjust as needed
)

# -----------------------------
# Save Model
# -----------------------------
model.save("mushroom_detector")
