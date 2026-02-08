#!/usr/bin/env python

# Load libraries.
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Load pre-trained model.
weights_file = '/home/ddopessoa/.cache/kagglehub/datasets/madmaxliu/inceptionv3/versions/1/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(weights_file)
#print(pre_trained_model.summary())

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')

#print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

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
                                                    target_size = (150, 150),
                                                    class_mode = 'binary'
                                                    )

val_generator = train_datagen.flow_from_directory(val_path,
                                                  target_size = (150, 150),
                                                  class_mode = 'binary'
                                                 )

# Build convolutional neural network model using transfer learning.
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation = 'relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation = 'sigmoid')(x)

model = models.Model(pre_trained_model.input, x)

# Compile model.
model.compile(optimizer = RMSprop(learning_rate = 0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

#history = model.fit(train_generator, epochs = 4)

history = model.fit(train_generator, validation_data = val_generator, epochs = 4)

# Predict images.
test_path = './test'
test_files = os.listdir(test_path)

for test_file in test_files:
    img = image.load_img(os.path.join(test_path, test_file), target_size = (150, 150))
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


