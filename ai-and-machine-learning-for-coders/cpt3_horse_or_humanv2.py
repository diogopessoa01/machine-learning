#!/usr/bin/env python

# Load libraries.
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Load dataset and assign to training/test variables.
train_path = '/home/ddopessoa/.cache/kagglehub/datasets/sanikamal/horses-or-humans-dataset/versions/1/horse-or-human/train'
val_path = '/home/ddopessoa/.cache/kagglehub/datasets/sanikamal/horses-or-humans-dataset/versions/1/horse-or-human/validation'

# All images will be rescaled by 1./255
# Apply multiple data augmentation transforms to the dataset.
train_datagen = ImageDataGenerator(rescale = 1/255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest'
                                  )

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size = (300, 300),
                                                    batch_size = 128,
                                                    class_mode = 'binary'
                                                    )

validation_datagen = ImageDataGenerator(rescale = 1/255)

val_generator = validation_datagen.flow_from_directory(val_path,
                                                       target_size = (300, 300),
                                                       class_mode = 'binary'
                                                      )

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
model.compile(optimizer = RMSprop(learning_rate = 0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# Fit model.
history = model.fit(train_generator,
                    validation_data = val_generator,
                    epochs = 5)

# Predict images.
test_path = '/home/ddopessoa/Documents/aimlc/test'
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


