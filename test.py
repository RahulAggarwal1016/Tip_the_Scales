# This file takes a .wav input and predicts which class the scale falls in

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np

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

from numpy import expand_dims
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array


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

# try:
image_filenames = ['image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7']
create_spectrogram('wav/har-minor/scale_c_har-minor.wav', 'image1', '')
create_spectrogram('scale-c-major.wav', 'image2', '')
create_spectrogram('a-minor-har_r.wav', 'image3', '')
create_spectrogram('bflat-minor-mel-incomplete_r.wav', 'image4', '')
create_spectrogram('d-major_r.wav', 'image5', '')
create_spectrogram('e-minor-nat_r.wav', 'image6', '')
create_spectrogram('fsharp-major-incomplete_r.wav', 'image7', '')
# print("Images Created")

model = load_model('model.h5')

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['acc'])
# print("Model loaded + compiled")

images = []
for img in image_filenames:
    current_img = image.load_img(img + '.jpg', target_size=(200, 200))
    x = image.img_to_array(current_img)
    x = np.expand_dims(x, axis=0)

    images.append(x)
# print("Images converted to arrays")

    # stack up images list to pass for prediction
images = np.vstack(images)
classes = model.predict(images, batch_size=10)
# print(classes[0].index(max(classes[0])))
    # print(max(classes[0][0], classes[0][1], classes[0][2], classes[0][3]))

for i in range(len(classes)):
    if classes[0][0] >= classes[0][1] and classes[0][0] >= classes[0][2] and classes[0][0] >= classes[0][3]:
        print("Harmonic Minor")
    elif classes[0][1] >= classes[0][0] and classes[0][1] >= classes[0][2] and classes[0][1] >= classes[0][3]:
        print("Major")
    elif classes[0][2] >= classes[0][1] and classes[0][2] >= classes[0][0] and classes[0][2] >= classes[0][3]:
        print("Melodic Minor")
    elif classes[0][3] >= classes[0][1] and classes[0][3] >= classes[0][2] and classes[0][3] >= classes[0][0]:
        print("Natural Minor")
    elif max(classes[0][0], classes[0][1], classes[0][2], classes[0][3]) <= 0.4:
        print("Scale is not a major, natural minor, harmonic minor, or melodic minor")

# print(len(classes))
# print(len(classes[0]))
# print(index_max)


# except:
#     print("Did not work")


# removes the image.jpg file that was created earlier
# os.remove("image.jpg")
