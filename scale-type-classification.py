# This file initializes and trains the model

import librosa
import librosa.display
from path import Path

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizers import Adam

import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc

from memory_profiler import memory_usage
import os
import pandas as pd
import glob
import numpy as np

from numpy import expand_dims
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array

#######################################################################################

# some filepath variables
filepath_train = os.path.dirname(__file__) + "/images/"
filepath_valid = os.path.dirname(__file__) + "/val-images/"

filepath_wav = os.path.dirname(__file__) + "/wav/"
scale_folders = ["major/", "nat-minor/", "har-minor/", "mel-minor/", "other/"]

# some other important variables
## scale categories here:
scale_labels = ["major", "natural minor", "harmonic minor", "melodic minor", "other"]

#######################################################################################


# Image data generator for the NN - incorporated Dhruv's version from read-wav.py now with minor changes
training_img_generator = ImageDataGenerator(
    brightness_range=[0.7, 1.3]
)

valid_img_generator = ImageDataGenerator(
    brightness_range=[0.7, 1.3]
)

# this code uses the image data generator declared above on all the x values (training images)
train_generator = training_img_generator.flow_from_directory(
        filepath_train,  # This is the source directory for training images
        target_size=(200, 200), # resizes all the images to this size
        batch_size=140, # will generate 140 images per batch
        # categorical because we will be classifying the data into 5 discrete categories
        class_mode='categorical',
        shuffle=True # shuffles the dataset into a random order
)

valid_generator = valid_img_generator.flow_from_directory(
    filepath_valid,
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)


#######################################################################################
# # code to help us visualize what a flow_from_directory object looks like
# print(train_generator.classes) # an array of numbers from 0 - 4, representing the 5 classes
# print(len(train_generator)) # 1 - idk
# print(len(train_generator[0])) # 2 - labels [1] and training data [0]
#
# ## let's go through the labels first
# print(len(train_generator[0][1])) # 74 - 74 images and 74 corresponding labels
# print(len(train_generator[0][1][0])) # 5 - This is the length of one label. So I know that the labels are are one-hot
# # encoded as opposed to being represented by 5 integers.
#
# ## by the way: one-hot encoding: instead of representing 5 labels as 0, 1, 2, 3, 4, they are represented as vectors
# ### [1 0 0 0 0]  for label 0, [0 1 0 0 0] for label 1, [0 0 1 0 0] for label 2, etc
# ### the name one-hot comes from the fact that all the values in the vector are 0 EXCEPT the one "hot" value that is 1
#
# ## now let's go through the images. Images are represented as 2D matrices
# print(len(train_generator[0][0])) # 74 - 74 images and 74 corresponding labels in train_generator[0][1]
# print(len(train_generator[0][0][0])) # 200 - the images are 200 by 200, and this is image data, so this number probably
# # represents how many rows there are
# print(len(train_generator[0][0][0][0])) # 200 - now we're in row 0 so this number represents how many columns there are
#################################################################################


# defining the keras model
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # Using 2 convolutions for now to reduce the size bc I printed summary and there were 61 million parameters
    ## and 2 layers managed to reduce it to 18 million
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), # Reduces the dimensions to feed into the layer

    # 256 neuron hidden layer
    tf.keras.layers.Dense(256, activation='relu'),
    # Output (non-hidden) layer
    tf.keras.layers.Dense(5, activation='softmax')
])

model.summary()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics=['acc'])

hist = model.fit(
      train_generator, # the training set
      steps_per_epoch=10, # how many batches are you going to split the training set into every epoch?
      epochs=12, # how many epochs are you going to train for?
      verbose=2, # verbose=0 means don't print out the epochs at all, 1 and 2 mean print out the epochs as you go through them
      validation_steps=8, # how many batches will you split the validation set into per epoch?
      validation_data = valid_generator, # at the end of each epoch, NN will test on the validation set
      )


model.save("model.h5")
