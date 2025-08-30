import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE # Lets TensorFlow optimize the data loading speed
DATA_DIR = pathlib.Path('Data')

train_dir = DATA_DIR / 'Training'
test_dir = DATA_DIR / 'Testing'
train_image_paths = list(train_dir.glob('*/*.jpg')) + list(train_dir.glob('*/*.png'))
test_image_paths = list(test_dir.glob('*/*.jpg')) + list(test_dir.glob('*/*.png'))

# print(f"Found {len(train_image_paths)} training images")
# print(f"Found {len(test_image_paths)} testing images")

def parse_image(image_path):
    """
    Returns image and label as tf.Tensors
    Image is 3D tensor with 224,224,3 shape and float32
    Label is a 1,1,1 tensor with just the index of the class
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    label = tf.strings.split(image_path, os.sep)[-2]
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary'] # Indexing labels
    label = tf.argmax(label == class_names)
    return image, label


def prepare_dataset(image_paths, shuffle=False, batch_size=32):
    """
    We create pipeline
    First path to string
    Then map parse_image onto each path with multi-core optimisations via num_parallel_calls
    We want to shuffle if training and avoid shuffle if validation or testing
    Batch and Prefetch is trivial for any model we want to optimise
    """
    dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in image_paths])
    dataset = dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

# Create datasets
train_ds = prepare_dataset(train_image_paths, shuffle=True, batch_size=BATCH_SIZE)
val_ds = prepare_dataset(train_image_paths, shuffle=False, batch_size=BATCH_SIZE)
test_ds = prepare_dataset(test_image_paths, shuffle=False, batch_size=BATCH_SIZE)


def visualize_augmentations(dataset, num_images=5):
    plt.figure(figsize=(15, 5))
    for images, labels in dataset.take(1):
        for i in range(min(num_images, BATCH_SIZE)):
            ax = plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i])
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis('off')
    plt.show()

visualize_augmentations(train_ds)


# Preprocessing function specifically for EfficientNet
def preprocess_efficientnet(image, label):
    # EfficientNet expects inputs in [-1, 1] range instead of [0, 1]
    image = tf.keras.applications.efficientnet.preprocess_input(image * 255.0)
    return image, label

# Apply EfficientNet-specific preprocessing
train_ds = train_ds.map(preprocess_efficientnet, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess_efficientnet, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(preprocess_efficientnet, num_parallel_calls=AUTOTUNE)

# Build the model
def create_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # Freeze initially
    
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)

model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    verbose=1
)