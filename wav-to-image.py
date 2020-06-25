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

from memory_profiler import memory_usage
import os
import pandas as pd
import glob
import numpy as np

def create_spectrogram(filename,name,savepath):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = name + '.jpg'
    plt.savefig(savepath + filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


filepath_img = os.path.dirname(__file__) + "/images/"
filepath_wav = os.path.dirname(__file__) + "/wav/"
scale_folders = ["major/", "nat-minor/", "har-minor/", "mel-minor/"]

# time to generate the graphs
for folder in scale_folders:
    wav_folder = filepath_wav + folder + "*.wav"
    img_folder = filepath_img + folder
    for scale in glob.iglob(wav_folder):
        # 1st split gets the filename of the current .wav file, 2nd split removes the .wav
        img_name = scale.split("/")[-1].split(".")[0]
        # print(img_name)
        create_spectrogram(scale, img_name, img_folder)


