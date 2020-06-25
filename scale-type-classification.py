import librosa
import librosa.display
from path import Path

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizers import Adam
# import tensorflow.python.keras.backend as K

import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc

from memory_profiler import memory_usage
import os
import pandas as pd
import glob
import numpy as np


# example of vertical shift image augmentation
from numpy import expand_dims
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array

from tensorflow.python.keras.utils import to_categorical

#######################################################################################

# some filepath variables
filepath_img = os.path.dirname(__file__) + "/images/"
filepath_wav = os.path.dirname(__file__) + "/wav/"
scale_folders = ["major/", "nat-minor/", "har-minor/", "mel-minor/", "other/"]

# some other important variables
## scale categories here:
scale_labels = ["major", "natural minor", "harmonic minor", "melodic minor", "other"]

#######################################################################################


# some handy functions
## to create a spectrogram from any .wav file
def create_spectrogram(filename, newname, savepath):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    newname = newname + '.jpg'
    plt.savefig(savepath + newname, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, newname, clip, sample_rate, fig, ax, S


# Image data generator for the NN - get Dhruv's version soon
training_img_generator = ImageDataGenerator(
    rescale=1./255
)

# using the image data generator declared above on all the training data
train_generator = training_img_generator.flow_from_directory(
        filepath_img,  # This is the source directory for training images
        target_size=(200, 200), # resizes all the images to this size
        batch_size=128, # will generate 128 images per batch
        # categorical because we will be classifying the data into 5 discrete categories
        class_mode='categorical')

model = tf.keras.models.Sequential([
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), # Reduces the dimensions

    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(4, activation='sigmoid')
])
