#!/usr/bin/env python

# Load libraries.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset and assign to training/test variables.
data = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

# Normalize/scale data to 0 to 1 range.
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test / 255.0

# Build convolutional neural network model.
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)), # learn 64 convolutions (filter of weights to pixels), apply on a 3x3 size filter, on an input of 28x28x1 shape
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation = 'relu'), # learn 64 convolutions (filter of weights to pixels), apply on a 3x3 size filter
    layers.MaxPooling2D(2, 2),
    layers.Flatten(), # specify and flatten 28x28 pixel image to vector
    layers.Dense(units = 128, activation = 'relu'), # hidden layer with 128 neurons
    layers.Dense(units = 10, activation = 'softmax') # output layer with 10 neurons for each class/label
    ])

# Compile model.
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5)

print(model.evaluate(x_test, y_test))

classifications = model.predict(x_test)

print(classifications[0])
print(y_test[0])
