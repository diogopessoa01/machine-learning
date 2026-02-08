#!/usr/bin/env python

# Load libraries.
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow_datasets as tfds

# Load dataset and assign to training/test variables.
data = tfds.load('horses_or_humans', split = 'train', as_supervised = True)
train_batches = data.shuffle(100).batch(10)

val_data = tfds.load('horses_or_humans', split = 'test', as_supervised = True)
validation_batches = val_data.batch(32)

# Build convolutional neural network model.
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (300, 300, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation = 'relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(units = 512, activation = 'relu'),
    layers.Dense(units = 1, activation = 'sigmoid') # output layer with 1 neuron with either 0 or 1 for each class/label
    ])

# Compile model.
model.compile(optimizer = "Adam",
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit(train_batches,
                    validation_data = validation_batches,
                    validation_steps = 1,
                    epochs = 5)

# Predict images.
test_path = './test'
test_files = os.listdir(test_path)

for test_file in test_files:
    img = image.load_img(os.path.join(test_path, test_file), target_size = (300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes)
    print(classes[0])
    if classes[0] > 0.5:
        print(f"{test_file} is a human")
    else:
        print(f"{test_file} is a horse")


