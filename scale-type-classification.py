import librosa
import librosa.display
from path import Path

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

filepath_img = os.path.dirname(__file__) + "/images/"
# filepath_wav = os.path.dirname(__file__) + "/wav/"
scale_folders = ["major/", "nat-minor/", "har-minor/", "mel-minor/"]

training_img_generator = ImageDataGenerator(
    rescale=1./255
)

train_generator = training_img_generator.flow_from_directory(
        filepath_img,  # This is the source directory for training images
        target_size=(200, 200),
        batch_size=128,
        class_mode='categorical')

