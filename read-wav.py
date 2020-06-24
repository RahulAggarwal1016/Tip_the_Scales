import wave
import struct

def parse_wave(filepath):
    with wave.open(filepath, "rb") as wave_file:
        sample_rate = wave_file.getframerate()
        length_in_sec = wave_file.getnframes() / sample_rate
        # print(wave_file.readframes(1))

        first_sample = struct.unpack('<I', wave_file.readframes(1))[0]
        second_sample = struct.unpack('<I', wave_file.readframes(1))[0]

        print('''
Parsed {filename}
-----------------------------------------------
Channels: {num_channels}
Sample Rate: {sample_rate}
First Sample: {first_sample}
Second Sample: {second_sample}
Length in Seconds: {length_in_seconds}'''.format(
            filename=filepath,
            num_channels=wave_file.getnchannels(),
            sample_rate=wave_file.getframerate(),
            first_sample=first_sample,
            second_sample=second_sample,
            length_in_seconds=length_in_sec))
        return wave_file

wf = parse_wave("wav/major/scale_c_major.wav")
print(wf)

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
from glob import glob
import numpy as np

def create_spectrogram(filename,name):
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
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

create_spectrogram("wav/major/scale_c_major.wav", "C Major Scale")
