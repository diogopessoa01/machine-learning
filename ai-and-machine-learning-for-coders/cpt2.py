#!/usr/bin/env python

# Load libraries.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset and assign to training/test variables.
data = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

# Normalize/scale data to 0 to 1 range.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build neural network model.
model = models.Sequential([
    layers.Flatten(input_shape = (28, 28, 1)), # specify and flatten 28x28 pixel image to vector
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
