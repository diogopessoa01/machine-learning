#!/bin/bash python3

import numpy as np
from tensorflow.keras import datasets,utils
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

# Load CIFAR iamge dataset from keras.
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

NUM_CLASSES = 10

# Scale iamge dataset from 0 to 255 (pixels) to 0 to 1 scale for neural network input.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One hot encode the (10) labels for the images.
y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

# Build MLP using a functional model.
input_layer = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_layer)
x = layers.Dense(units=200, activation = 'relu')(x)
x = layers.Dense(units=150, activation = 'relu')(x)
output_layer = layers.Dense(units=10, activation = 'softmax')(x)
model = models.Model(input_layer, output_layer)

# Print model summary to visualize MLP architecture.
model.summary()

# Compile model with optimizaer and loss function.
opt = optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model.
model.fit(x_train
, y_train
, batch_size = 32
, epochs = 10
, shuffle = True
)

# Evaluate the model.
model.evaluate(x_test, y_test)

# View predictions on the test set.
#CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog'
#, 'frog', 'horse', 'ship', 'truck'])
#preds = model.predict(x_test)
#preds_single = CLASSES[np.argmax(preds, axis = -1)]
#actual_single = CLASSES[np.argmax(y_test, axis = -1)]
#n_to_show = 10
#indices = np.random.choice(range(len(x_test)), n_to_show)
#fig = plt.figure(figsize=(15, 3))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#for i, idx in enumerate(indices):
#img = x_test[idx]
#ax = fig.add_subplot(1, n_to_show, i+1)
#ax.axis('off')
#ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10
#, ha='center', transform=ax.transAxes)
#ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10
#, ha='center', transform=ax.transAxes)
#ax.imshow(img)
