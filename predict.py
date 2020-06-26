# Not a part of the final product. This was used for debugging purposes

#################################################################################
#
# How the code works:
#   - User writes the path to their WAV audio file on line 61 and
#     create_spectrogram() function makes a .jpg image that is saved to project
#     directory.
#   - Program uses ML model to predict scale type.
#
#################################################################################


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

# .wav file converted to jpg spectrogram
wav_path = 'wav/mel-minor/scale_f_mel-minor.wav'                                # Write the path to your WAV audio file here
create_spectrogram(wav_path, 'image', '')


# loads and compiles ML model
model = load_model('model.h5')
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['acc'])


image_path = "image.jpg"

current_img = image.load_img(image_path, target_size=(200, 200))
x = image.img_to_array(current_img)
x = np.expand_dims(x, axis=0)

# stack up images list to pass for prediction
images = np.vstack([x])
classes = model.predict(images, batch_size=10)

print(classes)

if classes[0][0] >= classes[0][1] and classes[0][0] >= classes[0][2] and classes[0][0] >= classes[0][3] and classes[0][0] >= classes[0][4] and classes[0][0] >= 0.25:
    print("Harmonic Minor")
elif classes[0][1] >= classes[0][0] and classes[0][1] >= classes[0][2] and classes[0][1] >= classes[0][3] and classes[0][1] >= classes[0][4] and classes[0][1] >= 0.25:
    print("Major")
elif classes[0][2] >= classes[0][1] and classes[0][2] >= classes[0][0] and classes[0][2] >= classes[0][3] and classes[0][2] >= classes[0][4] and classes[0][2] >= 0.25:
    print("Melodic Minor")
elif classes[0][3] >= classes[0][1] and classes[0][3] >= classes[0][2] and classes[0][3] >= classes[0][0] and classes[0][3] >= classes[0][4] and classes[0][3] >= 0.25:
    print("Natural Minor")
elif classes[0][4] >= classes[0][1] and classes[0][4] >= classes[0][2] and classes[0][4] >= classes[0][0] and classes[0][4] >= classes[0][3]:
    print("Scale is not a major, natural minor, harmonic minor, or melodic minor")
elif max(classes[0][0], classes[0][1], classes[0][2], classes[0][3]) < 0.25:
    print("Unable to identify")

# removes the image.jpg file that was created earlier
os.remove(image_path)
