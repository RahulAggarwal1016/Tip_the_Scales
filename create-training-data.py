# Not a part of the final product. This was used for training purposes

#################################################################################
#
# This .py file was used to create training data from 71 WAV audio files.
#
# How the code works:
#   - Using the create_spectrogram() function, a spectrogram .jpg file was
#     created for each WAV audio file and saved in the project directory.
#   - The for loop goes through all of the .jpg files that were created earlier
#     and calls the create_more_images() for each image.
#   - The create_more_images() method applies image transformations to the .jpg
#     files and creates 20 more images that are stored in the images directory.
#
#################################################################################

import librosa
import librosa.display
from path import Path

from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc

# from memory_profiler import memory_usage
import os
import pandas as pd
from glob import glob
import numpy as np

# used for image transformations
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot


def create_spectrogram(filename, name, image_num):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = name + str(image_num) + '.jpg'
    # creates image.jpg file in the common_western_scales directory
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename, name, clip, sample_rate, fig, ax, S

## takes wave files and creates spectrogram .jpg images
# Major scales
create_spectrogram("wav/major/scale_a_major.wav", "image", 0)
create_spectrogram("wav/major/scale_aflat_major.wav", "image", 1)
create_spectrogram("wav/major/scale_b_major.wav", "image", 2)
create_spectrogram("wav/major/scale_bflat_major.wav", "image", 3)
create_spectrogram("wav/major/scale_c_major.wav", "image", 4)
create_spectrogram("wav/major/scale_d_major.wav", "image", 5)
create_spectrogram("wav/major/scale_dflat_major.wav", "image", 6)
create_spectrogram("wav/major/scale_e_major.wav", "image", 7)
create_spectrogram("wav/major/scale_eflat_major.wav", "image", 8)
create_spectrogram("wav/major/scale_f_major.wav", "image", 9)
create_spectrogram("wav/major/scale_fsharp_major.wav", "image", 10)
create_spectrogram("wav/major/scale_g_major.wav", "image", 11)

# Natural Minor scales
create_spectrogram("wav/nat-minor/scale_a_nat-minor.wav", "image", 12)
create_spectrogram("wav/nat-minor/scale_b_nat-minor.wav", "image", 13)
create_spectrogram("wav/nat-minor/scale_bflat_nat-minor.wav", "image", 14)
create_spectrogram("wav/nat-minor/scale_c_nat-minor.wav", "image", 15)
create_spectrogram("wav/nat-minor/scale_csharp_nat-minor.wav", "image", 16)
create_spectrogram("wav/nat-minor/scale_d_nat-minor.wav", "image", 17)
create_spectrogram("wav/nat-minor/scale_dsharp_nat-minor.wav", "image", 18)
create_spectrogram("wav/nat-minor/scale_e_nat-minor.wav", "image", 19)
create_spectrogram("wav/nat-minor/scale_f_nat-minor.wav", "image", 20)
create_spectrogram("wav/nat-minor/scale_fsharp_nat-minor.wav", "image", 21)
create_spectrogram("wav/nat-minor/scale_g_nat-minor.wav", "image", 22)
create_spectrogram("wav/nat-minor/scale_gsharp_nat-minor.wav", "image", 23)

# Melodic Minor scales
create_spectrogram("wav/mel-minor/scale_a_minor.wav", "image", 24)
create_spectrogram("wav/mel-minor/scale_b_mel-minor.wav", "image", 25)
create_spectrogram("wav/mel-minor/scale_bflat_mel-minor.wav", "image", 26)
create_spectrogram("wav/mel-minor/scale_c_mel-minor.wav", "image", 27)
create_spectrogram("wav/mel-minor/scale_csharp_mel-minor.wav", "image", 28)
create_spectrogram("wav/mel-minor/scale_d_mel-minor.wav", "image", 29)
create_spectrogram("wav/mel-minor/scale_dsharp_mel-minor.wav", "image", 30)
create_spectrogram("wav/mel-minor/scale_e_mel-minor.wav", "image", 31)
create_spectrogram("wav/mel-minor/scale_f_mel-minor.wav", "image", 32)
create_spectrogram("wav/mel-minor/scale_fsharp_mel-minor.wav", "image", 33)
create_spectrogram("wav/mel-minor/scale_g_mel-minor.wav", "image", 34)
create_spectrogram("wav/mel-minor/scale_gsharp_mel-minor.wav", "image", 35)

# Harmonic Minor scales
create_spectrogram("wav/har-minor/scale_a_har-minor.wav", "image", 36)
create_spectrogram("wav/har-minor/scale_b_har-minor.wav", "image", 37)
create_spectrogram("wav/har-minor/scale_bflat_har-minor.wav", "image", 38)
create_spectrogram("wav/har-minor/scale_c_har-minor.wav", "image", 39)
create_spectrogram("wav/har-minor/scale_csharp_har-minor.wav", "image", 40)
create_spectrogram("wav/har-minor/scale_d_har-minor.wav", "image", 41)
create_spectrogram("wav/har-minor/scale_dsharp_har-minor.wav", "image", 42)
create_spectrogram("wav/har-minor/scale_e_har-minor.wav", "image", 43)
create_spectrogram("wav/har-minor/scale_f_har-minor.wav", "image", 44)
create_spectrogram("wav/har-minor/scale_fsharp_har-minor.wav", "image", 45)
create_spectrogram("wav/har-minor/scale_g_har-minor.wav", "image", 46)
create_spectrogram("wav/har-minor/scale_gsharp_har-minor.wav", "image", 47)

# Other scales
create_spectrogram("wav/other/a_minor_pentatonic.wav", "image", 48)
create_spectrogram("wav/other/b_lydian_mode.wav", "image", 49)
create_spectrogram("wav/other/c_dorian_mode.wav", "image", 50)
create_spectrogram("wav/other/c_mixolydian_mode.wav", "image", 51)
create_spectrogram("wav/other/c_pentatonic.wav", "image", 52)
create_spectrogram("wav/other/c_pentatonic_2.wav", "image", 53)
create_spectrogram("wav/other/d_chromatic_fast.wav", "image", 54)
create_spectrogram("wav/other/d_phrygian_mode.wav", "image", 55)
create_spectrogram("wav/other/dualtone_1.wav", "image", 56)
create_spectrogram("wav/other/dualtone_2.wav", "image", 57)
create_spectrogram("wav/other/g_chromatic_slow.wav", "image", 58)
create_spectrogram("wav/other/gflat_locrian_mode.wav", "image", 59)
create_spectrogram("wav/other/pulses.wav", "image", 60)
create_spectrogram("wav/other/random_chords.wav", "image", 61)
create_spectrogram("wav/other/random_chords_2.wav", "image", 62)
create_spectrogram("wav/other/random_melody_1_1.wav", "image", 63)
create_spectrogram("wav/other/random_melody_1_2.wav", "image", 64)
create_spectrogram("wav/other/random_melody_2.wav", "image", 65)
create_spectrogram("wav/other/singletone_1.wav", "image", 66)
create_spectrogram("wav/other/sweep_high.wav", "image", 67)
create_spectrogram("wav/other/sweep_low.wav", "image", 68)
create_spectrogram("wav/other/warble.wav", "image", 69)
create_spectrogram("wav/other/whitenoise_1.wav", "image", 70)
create_spectrogram("wav/other/whitenoise_2.wav", "image", 71)


def create_more_images(image_file, path, use_name, letter):
    img = load_img(image_file) # loads image we want to transform
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator (increases dataset)
    train_datagen = ImageDataGenerator(
        width_shift_range=[-10, 30],
        brightness_range=[0.7, 1.0],
        fill_mode = "constant" # should make images fill with black when translating
    )
    # prepare iterator
    it = train_datagen.flow(samples, batch_size=1)
    # displays transformed images on pyplot
    for i in range(20):
        plt.interactive(False)
        batch = it.next() # transforms image
        image = batch[0].astype('uint8') # gets the image and converts it to an unsigned integer
        fig = plt.figure(figsize=[0.72, 0.72]) # creates a figure
        plt.imshow(image) # shows the transformed image
        # does not display axes
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        filename = "image_" + letter + "_" + use_name + "_" + str(i) + ".jpg" # creates file name
        # creates image.jpg file in the common_western_scales directory
        plt.savefig(path + filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()
        fig.clf()
        plt.close(fig)
        plt.close('all')
        del filename, fig, ax


# load all the created images
for i in range(71):
    # parameters used for naming the files
    image_file = "image" + str(i) + ".jpg"

    if i <= 11 and i >= 0:
        path = "images/major/"
        use_name = "major"
    elif i <= 23 and i >= 12:
        path = "images/nat-minor/"
        use_name = "nat-minor"
    elif i <= 35 and i >= 24:
        path = "images/mel-minor/"
        use_name = "mel-minor"
    elif i <= 47 and i >= 36:
        path = "images/har-minor/"
        use_name = "har-minor"
    else:
        path = "other/"
        use_name = "other" + str(i-47) + "_"

    if i == 0 or i == 12 or i == 24 or i == 36:
        letter = "a"
    elif i == 1:
        letter = "aflat"
    elif i == 2 or i == 13 or i == 25 or i == 37:
        letter = "b"
    elif i == 3 or i == 14 or i == 26 or i == 38:
        letter = "bflat"
    elif i == 4 or i == 15 or i == 27 or i == 39:
        letter = "c"
    elif i == 16 or i == 28 or i == 40:
        letter = "csharp"
    elif i == 5 or i == 17 or i == 29 or i == 41:
        letter = "d"
    elif i == 6:
        letter = "dflat"
    elif i == 18 or i == 30 or i == 42:
        letter = "dsharp"
    elif i == 7 or i == 19 or i == 31 or i == 43:
        letter = "e"
    elif i == 8:
        letter = "eflat"
    elif i == 9 or i == 20 or i == 32 or i == 44:
        letter = "f"
    elif i == 10 or i == 21 or i == 33 or i == 45:
        letter = "fsharp"
    elif i == 11 or i == 22 or i == 34 or i == 46:
        letter = "g"
    elif i == 23 or i == 35 or i == 47:
        letter = "gsharp"

    try:
        create_more_images(image_file, path, use_name, letter) # call function to create transformations of the image
        # removes the file that was created earlier as it is not needed anymore
        os.remove(image_file)
    except: # if the function could not be called
        os.remove(image_file)
        print("Didn't work " + str(i))


print("All images are generated!")
