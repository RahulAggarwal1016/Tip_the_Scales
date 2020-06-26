
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

import librosa
import librosa.display
from path import Path

import tensorflow as tf
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc

from memory_profiler import memory_usage
import os
import pandas as pd
import glob

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

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
    plt.savefig(savepath + newname, dpi=250, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, newname, clip, sample_rate, fig, ax, S

create_spectrogram('html/audio_input.wav', 'image', 'html/img/')  # CHANGE PATH NAME OF AUDIO FILE
