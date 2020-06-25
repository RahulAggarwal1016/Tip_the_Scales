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

def create_spectrogram(filename, name, savepath, quality):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.7, 0.7])
    ax = fig.add_subplot(111)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

    filename = name + '.jpg'
    plt.savefig(savepath + filename, dpi=quality, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

# def create_spectrogram(filename, name, savepath, img_quality):
#     plt.interactive(False)
#     clip, sample_rate = librosa.load(filename, sr=None)
#     fig = plt.figure(figsize=[0.7,0.7])
#     ax = fig.add_subplot(111)
#
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#     ax.set_frame_on(False)
#     S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
#     librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
#     fig.subplots_adjust(0,0,1,1)
#
#     filename = name + '.jpg'
#     plt.savefig(savepath + filename, dpi=img_quality, bbox_inches=0,pad_inches=0)
#     print("saved")
#     plt.close()
#     fig.clf()
#     plt.close(fig)
#     plt.close('all')
#     del filename,name,clip,sample_rate,fig,ax,S

def create_more_images(image_file, path, use_name):
    img = load_img(image_file) # loads image we want to transform
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator (increases dataset)
    train_datagen = ImageDataGenerator(
        width_shift_range=[-10, 30],
        brightness_range=[0.7, 1.0],
        fill_mode="constant" # should make images fill with black when translating
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
        filename = "image_" + use_name + "_" + str(i) + ".jpg" # creates file name
        # creates image.jpg file in the common_western_scales directory
        plt.savefig(path + filename, dpi=250, bbox_inches='tight', pad_inches=0)
        # print("saved " + str(i))

        plt.close()
        fig.clf()
        plt.close(fig)
        plt.close('all')
        del filename, fig, ax


filepath_img = os.path.dirname(__file__) + "/images-original/"
filepath_wav = os.path.dirname(__file__) + "/wav/"
scale_folders = ["major/", "nat-minor/", "har-minor/", "mel-minor/", "other/"]

# time to generate the graphs
for folder in scale_folders:
    wav_folder = filepath_wav + folder + "*.wav"
    img_folder = filepath_img + folder
    for scale in glob.iglob(wav_folder):
        # 1st split gets the filename of the current .wav file, 2nd split removes the .wav
        img_name = scale.split("/")[-1].split(".")[0]
        # print(img_name)
        create_spectrogram(scale, img_name, img_folder, 250) # current quality is 250dpi
        # filename = "image_" + letter + "_" + use_name + "_" + str(i) + ".jpg"

# # code to expand the dataset
# for folder in scale_folders:
#     # print("hi 1")
#     save_path = os.path.dirname(__file__) + "/images/" + folder
#     img_folder = filepath_img + folder + "*.jpg"
#     for image in glob.iglob(img_folder):
#         # print("hi 2")
#
#         # 1st split gets the filename of the current .jpg file, 2nd split removes the .jpg
#         use_name = image.split("/")[-1].split(".")[0]
#
#         create_more_images(image, save_path, use_name)
