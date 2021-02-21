# import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
# from pydub import AudioSegment
import librosa
import librosa.display
import sound

# extract features
path = "pirates.wav"
wav, sr = librosa.load(path, sr=44100)
# print(type(wav))
X = librosa.stft(wav)
Xdb = librosa.amplitude_to_db(abs(X))
fig, ax = plt.subplots(1, 2, figsize=(8, 6))
plt.subplot(211)
librosa.display.waveplot(wav, sr=sr)
plt.subplot(212)
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.show()