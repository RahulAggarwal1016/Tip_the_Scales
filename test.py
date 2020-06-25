import librosa
import librosa.display
import matplotlib.pyplot as plt

import os
import glob
import numpy as np

from tensorflow.python.keras.preprocessing.image import load_img
from numpy import expand_dims
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def create_spectrogram(filename,name,savepath,img_quality):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.7,0.7])
    ax = fig.add_subplot(111)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    fig.subplots_adjust(0,0,1,1)

    filename = name + '.jpg'
    plt.savefig(savepath + filename, dpi=img_quality, bbox_inches='tight',pad_inches=0)
    print("saved")
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


filepath_img = os.path.dirname(__file__) + "/images/"
filepath_wav = os.path.dirname(__file__) + "/wav/"
scale_folders = ["major/", "nat-minor/", "har-minor/", "mel-minor/", "other/"]

wav_folder = filepath_wav + "har-minor/" + "scale_c_har-minor.wav"
print(wav_folder)
print(filepath_img)

create_spectrogram(wav_folder,"scale_c_har-minor-400",filepath_img,400)
create_spectrogram(wav_folder,"scale_c_har-minor-300",filepath_img,300)
create_spectrogram(wav_folder,"scale_c_har-minor-250",filepath_img,250)
create_spectrogram(wav_folder,"scale_c_har-minor-200",filepath_img,200)
create_spectrogram(wav_folder,"scale_c_har-minor-100",filepath_img,100)
create_spectrogram(wav_folder,"scale_c_har-minor-50",filepath_img,50)
