# common_western_scales

A program that will distinguish between different types of 7-note scales being played. Specifically, between major, minor, harmonic minor, and melodic minor, though hopefully later on we will expand this to include pentatonic scales, chromatic scales, and modes.

# Deployment Instructions

There are two ways to use this program:
1. Through our web application
2. Through a Python IDE

## Using our Web Application

**hewwo**

## Using a Python IDE

1. Download this repository on your computer.
2. Make sure you have [Python3](https://www.python.org/downloads/) installed on your computer
3. Open the project folder in a Python IDE. We recommend using [Pycharm CE](https://www.jetbrains.com/pycharm/download/#section=mac) by Jet Brains which can be downloaded for free.
4. Set up your Python3 interpreter and install the following modules. Before installing these modules on your IDE, you have to download them on your computer using   "pip3 install ____" (the underlines represent the module names:
    - tensorflow
    - keras
    - librosa
    - image
    - numba==0.48 (when downloading this in the IDE, you will have to specify this version of the module specifically)
    - matplotlib
    - os
    - sounddevice
    - write
    - numpy

5. Open *"common_western_scales/predict-with-recording.py"* and run the code (make sure you read the instructions at the top of the file).

**Disclaimers and Notes:**
- Python IDE Step 4: These are the modules we had to download when making this project. We noticed that the modules we had to download differed between teammates, so you may need to download additional ones.
- Python IDE Step 4: To install the modules in Pyharm CE, go to **Preferences->Project: common_western_scales->Project Interpreter**, and then **click the "+" sign** at the bottom left of the preferences window

# Contributions

This project was made between June 24-26, 2020 for the University of Waterloo Software Engineering Class of 2025 Hackathon.

**Team Name:** The EPIC Asians

**Team members:**
- Carol Xu
- Dhruv Rawat
- Rahul Aggarwal

# Hackathon Formalities ;)

## Inspiration

When we were begining our instructions in music, we found it difficult to distinguish between major, harmonic minor, melodic minor, and natrual minor scales. Often to resolve these issues, we would ask our music teachers or older friends to identify the scales for us. Now, as more experienced musicians, we are developing **Tip the Scales** to reduce the time children spend identifying scales.

## What it does

Tip the scales is very simple to use. Users input a WAV audio file of a scale and the website will identify it as major, harmonic minor, melodic minor, and natrual minor.

## How we built it

We first created a dataset for all of the scale types by making 12 .mid audio files using MuseScore. Afterwards, we then converted all of the files to WAV audio. Using Python, we used the librosa and matplotlib modules in *"wav-to-image.py"* to read the WAV audio files and create .jpg spectrograms for each of the wave files (after this step, we had 12 .jpg for each scale type). Afterwards, we used the keras.processing.image module to transform the images and create more .jpgs. Using the algorithm we made in *"create-training-data.py"*, we expanded our dataset to over 228 images per scale type.

Then, to apply the machine learning to each of the two environments, we did this:

#### Web application

We used React JavaScript to create the web application. To create a json model for tensorflow, we used [Teachable Machine](https://teachablemachine.withgoogle.com). We then linked tensorflowjs to the React app. Since the website takes WAV audio files as an input, we used ajax to call a python function in *"convert-to-jpg.py"* to create a .jpg sepctrogram that the json model can use.

#### Python IDE

Using TensorFlow Python, we generated a model in *"create-model.py"*. We used the sounddevice and write modules to record audio from the user, and keras.models and numpy to use the model in *"predict-with-recording.py"*.

## Challenges we ran into

Using TensorFlow to process audio files was extremely difficult. It took us a while to understand how to use iamge recognition to do predictions on audio files (by making spectrograms!!!). Additionally, creating the algorithm to transform the images and create more .jpgs was challenging as the transformations would distort the images. However, after some investigation, the issue was resolved and the quality of the images (which are in the images directory) are good. Moreover, running a server and accessing python functions using React js was a challenge which we resolved through perseverance and dedication (and lots of coffee).

## Accomplishments we are proud of

We are proud that we have gotten audio processing and image recognition with tensorflowjs and tensorflowpython to work. All of this would not have been possible had it not been for our amazing teamwork!

## What we learned

Through this project, we hae learned a lot about machine learning, TensorFlow, and audio processing.

## What is next for Tip the Scales

- Bringing Tip the Scales to iOS and Android.
- Adding more scale types to the machine learning model: Scale modes (ex. Major Locrian)
- Allowing the program to identify the key of the scale (ex. E Major)
