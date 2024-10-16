from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel, QVBoxLayout, QWidget, QSlider, QComboBox, QGraphicsRectItem,QGraphicsView,QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor,QMouseEvent
from PyQt5.QtCore import Qt, QRectF,pyqtSignal,QFile,QTextStream
from PyQt5.QtCore import Qt, QRectF, QObject, pyqtSignal

import pyaudio
import torch
import torch.nn.functional as F
import wave
import librosa
import numpy as np
import os
import sys
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#########
import pandas as pd
import numpy as np
from numpy import mean, var
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from PyQt5.QtCore import QCoreApplication

classifier = pickle.load(open('classifier.pkl', 'rb'))
##########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
# print(torch.__version__)

words = [  "Open middle door","Unlock the gate","Grant me access",]
path_words = ["Open middle door", "Unlock the gate" , "Grant me access"]
personal_records_path = [
    "ashf",
    "mask",
    "morad",
    "ziad",
    "emad",
]


    
records_spectrograms = []
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

def takeUserAudio(btn,duration= 4):
   # print(btn,"ana hna")
    recorder = pyaudio.PyAudio()
    # start recording
    live_record = recorder.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                frames_per_buffer=CHUNK)
    btn.setText("Recording...")
    QCoreApplication.processEvents()  # Add this line to force GUI update

  #  print("Recording...")
    data = []
    for i in range(0, int(RATE / CHUNK * duration)):
        audio_data = live_record.read(CHUNK)
        data.append(audio_data)
   # print("Finished Recording!")
    btn.setText("Processing...")
    QCoreApplication.processEvents()  # Add this line to force GUI update

    # stop and close everything
    live_record.stop_stream()
    live_record.close()
    recorder.terminate()
    # save to file
    saveAudio(recorder,data)
  

def saveAudio(recorder,data):
    file = wave.open("UserInput.wav",'w')
    file.setnchannels(CHANNELS)
    file.setsampwidth(recorder.get_sample_size(FORMAT))
    file.setframerate(RATE)
    file.writeframes(b"".join(data))
    file.close()


def extract_features(audio_path):
    #print("rec",audio_path)
    y, sr = librosa.load(audio_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    spectrogram = np.abs(librosa.stft(y))
    features = np.concatenate([
        np.mean(chroma, axis=1),
        np.mean(spectrogram, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(zero_crossing_rate, axis=1),
        np.mean(mfccs, axis=1)
    ])
    # if(audio_path == "UserInput.wav"): print("in ", features)
    # else:  print("data",features)
    return features

def spectroToTensor(audio_path):
    y, sr = librosa.load(audio_path)
    # For example, you can compute the spectrogram using librosa
    spectrogram = np.abs(librosa.stft(y))
    # You can convert the numpy array to a PyTorch tensor if needed
    tensor_representation   = torch.from_numpy(spectrogram)
    return tensor_representation  

def standardize():
     sc = StandardScaler()
     x_data = np.array(pd.read_csv('datasets\\users_dataset_features.csv'))
     y_data = np.array(pd.read_csv('datasets\\users_dataset_target.csv'))
     x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
     sc = StandardScaler()
     x_train = sc.fit_transform(x_train)
     x_test = sc.transform(x_test)
     return sc

def accessGate(btn):
    takeUserAudio(btn)
    #TODO convert spectro to tensor
    currentSpectroGram = spectroToTensor("UserInput.wav")
    words_similarity_scores = {}
    for audio_path in  path_words:
        audios = os.listdir(audio_path)
        records_spectrogram = [spectroToTensor(os.path.join(audio_path, audio)) for audio in audios]
        currentSpectroGram = currentSpectroGram.to(device=device,dtype=torch.float32)  
        records_spectrogram = [spec.to(device=device, dtype=torch.float32) for spec in records_spectrogram]  
        similarityMatrix = []
        for spectro in records_spectrogram:
            similarityMatrix.append(torch.max(F.conv2d(
                currentSpectroGram.unsqueeze(0) , spectro.unsqueeze(0).unsqueeze(0), padding = "same"
            )))
        words_similarity_scores[os.path.basename(audio_path)] = sum(similarityMatrix) / len(audios)

    magnitudes = torch.tensor(list(words_similarity_scores.values()), dtype=torch.float32, device=device)
    probabilities = torch.nn.functional.softmax(magnitudes / torch.max(magnitudes), dim=0)
    print("taza",probabilities)
    if magnitudes.all() < .3:
        print("Can't detect the word")
        probabilities.append(1- max(probabilities))
    max_similarity_score = max(words_similarity_scores.values())
    sentence_with_max_prop = max(words_similarity_scores, key=words_similarity_scores.get)
    return sentence_with_max_prop, probabilities

def transform_audio(audio, N_FRAMES, SHIFT_WINDOW, MELS):
    audio_array, sr = librosa.load(audio, duration=2)
    log_mel_audio_list_mean = []
    log_mel_audio_list_var = []
    mfccs_audio_list_mean = []
    mfccs_audio_list_var = []
    cqt_audio_list_mean = []
    cqt_audio_list_var = []
    chromagram_audio_list_mean = []
    chromagram_audio_list_var = []
    tone_audio_list_mean = []
    tone_audio_list_var = []
    log_mel_audio = librosa.power_to_db(librosa.feature.melspectrogram(audio_array, sr=sr, n_fft=N_FRAMES, hop_length=SHIFT_WINDOW, n_mels=MELS))
    mfccs_audio = librosa.feature.mfcc(y=audio_array, n_mfcc=MELS, sr=sr, n_fft=N_FRAMES, hop_length=SHIFT_WINDOW)
    cqt_audio = np.abs(librosa.cqt(y=audio_array, sr=sr, hop_length=SHIFT_WINDOW))
    chromagram_audio = librosa.feature.chroma_stft(audio_array, sr=sr, n_fft=N_FRAMES, hop_length=SHIFT_WINDOW)
    tone_audio = librosa.feature.tonnetz(y=audio_array, sr=sr)
    for i in range(len(log_mel_audio)):
         log_mel_audio_list_mean.append(log_mel_audio[i].mean())
         log_mel_audio_list_var.append(log_mel_audio[i].var())

    for i in range(len(mfccs_audio)):
         mfccs_audio_list_mean.append(mfccs_audio[i].mean())
         mfccs_audio_list_var.append(mfccs_audio[i].var())

    for i in range(len(cqt_audio)):
         cqt_audio_list_mean.append(cqt_audio[i].mean())
         cqt_audio_list_var.append(cqt_audio[i].var())

    for i in range(len(chromagram_audio)):
         chromagram_audio_list_mean.append(chromagram_audio[i].mean())
         chromagram_audio_list_var.append(chromagram_audio[i].var())

    for i in range(len(tone_audio)):
         tone_audio_list_mean.append(tone_audio[i].mean())
         tone_audio_list_var.append(tone_audio[i].var())

    sb_audio = librosa.feature.spectral_bandwidth(y=audio_array, sr=sr, n_fft=N_FRAMES, hop_length=SHIFT_WINDOW)

    ae_audio = fancy_amplitude_envelope(audio_array, N_FRAMES, SHIFT_WINDOW)
    rms_audio = librosa.feature.rms(audio_array, frame_length=N_FRAMES, hop_length=SHIFT_WINDOW)

    return np.hstack((mean(ae_audio), var(ae_audio), mean(rms_audio), var(rms_audio), mean(sb_audio), var(sb_audio), chromagram_audio_list_mean, chromagram_audio_list_var, tone_audio_list_mean, tone_audio_list_var, cqt_audio_list_mean, cqt_audio_list_var, mfccs_audio_list_mean, mfccs_audio_list_var, log_mel_audio_list_mean, log_mel_audio_list_var))


def fancy_amplitude_envelope(signal, framesize, hoplength):
    return np.array([max(signal[i:i+framesize]) for i in range(0, len(signal), hoplength)])

def recongize():
    sc = standardize()
    audio_path = "UserInput.wav"
    # Assuming you have a function 'transform_audio' to process the audio
    x_ver = transform_audio(audio_path, 1024, 512, 13)
    # Standardize the data
    x_ver = sc.transform(x_ver.reshape(1, -1))
    # Use predict_proba to get class probabilities
    class_probabilities = classifier.predict_proba(x_ver)
    class_probabilities = np.append(class_probabilities, np.full(4, random.uniform(0.01, 0.10)))
    # Get the index of the class with the highest probability
    predicted_class_index = class_probabilities.argmax()
    # Print or use the class probabilities as needed
  #  print("Class Probabilities:", class_probabilities)
    # Print the predicted person
    predicted_person = personal_records_path[predicted_class_index]
   # print(f"Predicted person is: {predicted_person}")

    return predicted_person,class_probabilities
        
        
    
def plotSpectro(self):
        y, sr = librosa.load("UserInput.wav")
    # Compute the spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # Convert the power spectrogram to decibels
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        plotSpectrogram(
        canvas=self.spectrogram.canvas,
        signal=spectrogram_db,
        sample_rate=sr
            )   
def plotSpectrogram(canvas,signal,sample_rate):
          canvas.axes.clear()
          canvas.axes.specgram(signal,Fs=sample_rate)
          canvas.draw()
        

