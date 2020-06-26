# This is part of the final product!!

#################################################################################
#
# How the code works:
#   - The file will record an audio clip (using your microphone) for a duration
#     that you specify
#   - The code will convert your audio file to a .jpg and then use model.h5 to
#     predict which scale you inputted
#
#################################################################################

# Imports requred packages
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import sounddevice as sd
from scipy.io.wavfile import write


#################################################################################
#
# Gets audio input from user
#
#################################################################################


valid_input = False

# Gets input from user regarding recording length
while not(valid_input):
    try:
        duration = int(input("How long (seconds) would you like your audio recording to be? (The maximum value you can input is \"20\")\n"))
        if duration > 20:
            print("Your input must be between 1 and 20 inclusive.")
        else:
            valid_input = True
    except ValueError: # if wrong datatype is inputted
        print("Invalid input. Please enter an integer.")


# Sets sample rate of recording
fs = 44100

print("Recording...")
recording = sd.rec(int(duration*fs), samplerate=fs, channels=2) # records audio from microphone
sd.wait() # Wait until recording is finished
print("Done recording.")
audio_path = "audio_input.wav"
write(audio_path, fs, recording) # Save as WAV file


#################################################################################
#
# Converts inputted WAV file to jpg
#
#################################################################################

print("Predicting your scale. This may take a while...")

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
create_spectrogram(audio_path, 'image', '')


#################################################################################
#
# Classifies scale using model.h5
# The model was created in create-model.py
#
#################################################################################

# loads and compiles ML model
model = load_model('model.h5')
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer='adam',
                metrics=['acc'])


image_path = "image.jpg"

current_img = image.load_img(image_path, target_size=(224, 224, 3)) # Resizes image
x = image.img_to_array(current_img)
x = np.expand_dims(x, axis=0)

# stack up images list to pass for prediction
images = np.vstack([x])
classes = model.predict(images, batch_size=10)

print("Prediction:")
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

# removes the image and audion files that were created earlier
os.remove(image_path)
os.remove(audio_path)
