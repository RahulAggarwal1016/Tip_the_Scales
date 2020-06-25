import librosa
import librosa.display
from path import Path

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizers import Adam
# import tensorflow.python.keras.backend as K
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc

# from memory_profiler import memory_usage
import os
import pandas as pd
from glob import glob
import numpy as np


def create_spectrogram(filename, name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S


create_spectrogram("wav/major/scale_c_major.wav", "image")

# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('image.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
train_datagen = ImageDataGenerator(
    width_shift_range=[-10, 30],
    brightness_range=[0.7, 1.0])

# prepare iterator
it = train_datagen.flow(samples, batch_size=1)

# displays transformed images on pyplot
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    pyplot.imshow(image)

pyplot.show()
