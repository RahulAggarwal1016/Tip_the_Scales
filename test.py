# This file takes a .wav input and predicts which class the scale falls in

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
import numpy as np

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
    plt.savefig(savepath + newname, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, newname, clip, sample_rate, fig, ax, S

try:
    create_spectrogram('wav/har-minor/scale_c_har-minor.wav', 'image', '')
    print("Image Created")

    model = load_model('model.h5')

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['acc'])

    img = image.load_img('image.jpg', target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    print(classes)
    # print(max(classes[0][0], classes[0][1], classes[0][2], classes[0][3]))

    if classes[0][0] >= classes[0][1] and classes[0][0] >= classes[0][2] and classes[0][0] >= classes[0][3]:
        print("Major")
    elif classes[0][1] >= classes[0][0] and classes[0][1] >= classes[0][2] and classes[0][1] >= classes[0][3]:
        print("Natural Minor")
    elif classes[0][2] >= classes[0][1] and classes[0][2] >= classes[0][0] and classes[0][2] >= classes[0][3]:
        print("Harmonic Minor")
    elif classes[0][3] >= classes[0][1] and classes[0][3] >= classes[0][2] and classes[0][3] >= classes[0][0]:
        print("Melodic Minor")
    elif max(classes[0][0], classes[0][1], classes[0][2], classes[0][3]) <= 0.4:
        print("Scale is not a major, natural minor, harmonic minor, or melodic minor")

except:
    print("Did not work")


# removes the image.jpg file that was created earlier
os.remove("image.jpg")
