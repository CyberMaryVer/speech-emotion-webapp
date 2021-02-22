# import time
import os
import numpy as np
import matplotlib.pyplot as plt
# import streamlit as st
# import pandas as pd
import cv2
import librosa
import librosa.display
# import sound
from tensorflow.keras.models import load_model

# load models
model = load_model("model.h5")
# tmodel = load_model("tmodel_all.h5")

# costants
CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']
CAT3 = ["positive", "neutral", "negative"]

# page settings
# st.set_page_config(layout="wide")

max_width = 1000
padding_top = 0
padding_right = "20%"
padding_left = "10%"
padding_bottom = 0
COLOR = "#1f1f2e"
BACKGROUND_COLOR = "#d1d1e0"

# st.markdown(
#         f"""
# <style>
#     .reportview-container .main .block-container{{
#         max-width: {max_width}px;
#         padding-top: {padding_top}rem;
#         padding-right: {padding_right}rem;
#         padding-left: {padding_left}rem;
#         padding-bottom: {padding_bottom}rem;
#     }}
#     .reportview-container .main {{
#         color: {COLOR};
#         background-color: {BACKGROUND_COLOR};
#     }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


# @st.cache
def save_audio(file):
    with open(os.path.join("audio", file.name), "wb") as f:
        f.write(file.getbuffer())

# @st.cache
def get_melspec(audio):
  y, sr = librosa.load(audio, sr=44100)
  X = librosa.stft(y)
  Xdb = librosa.amplitude_to_db(abs(X))
  img = np.stack((Xdb,) * 3,-1)
  img = img.astype(np.uint8)
  grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  grayImage = cv2.resize(grayImage, (224, 224))
  rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
  return (rgbImage, Xdb)

# @st.cache
def get_mfccs(audio, limit):
  y, sr = librosa.load(audio)
  a = librosa.feature.mfcc(y, sr=sr, n_mfcc = 20)
  if a.shape[1] > limit:
    mfccs = a[:,:limit]
  elif a.shape[1] < limit:
    mfccs = np.zeros((a.shape[0], limit))
    mfccs[:, :a.shape[1]] = a
  return mfccs

# @st.cache
def get_title(predictions, categories=CAT6):
    title = f"Detected emotion: {categories[predictions.argmax()]} \
    - {predictions.max() * 100:.2f}%"
    return title

# @st.cache
def plot_emotions(fig, data6, data3=None, title="Detected emotion",
                  categories6=CAT6, categories3=CAT3):

  color_dict = {"neutral":"grey",
                "positive":"green",
                "happy": "green",
                "surprise":"orange",
                "fear":"purple",
                "negative":"red",
                "angry":"red",
                "sad":"lightblue"}

  if data3 is None:
      pos = data6[3] + data6[5]
      neu = data6[2]
      neg = data6[0] + data6[1] + data6[4]
      data3 = np.array([pos, neu, neg])

  ind = categories6[data6.argmax()]
  color6 = color_dict[ind]

  data6 = list(data6)
  n = len(data6)
  data6 += data6[:1]
  angles6 = [i/float(n)*2*np.pi for i in range(n)]
  angles6 += angles6[:1]

  ind = categories3[data3.argmax()]
  color3 = color_dict[ind]

  data3 = list(data3)
  n = len(data3)
  data3 += data3[:1]
  angles3 = [i/float(n)*2*np.pi for i in range(n)]
  angles3 += angles3[:1]

  # fig = plt.figure(figsize=(10, 4))
  fig.set_facecolor('#d1d1e0')
  ax = plt.subplot(122, polar="True")
  # ax.set_facecolor('#d1d1e0')
  plt.polar(angles6, data6, color=color6)
  plt.fill(angles6, data6, facecolor=color6, alpha=0.25)

  ax.spines['polar'].set_color('lightgrey')
  ax.set_theta_offset(np.pi / 3)
  ax.set_theta_direction(-1)
  plt.xticks(angles6[:-1], categories6)
  ax.set_rlabel_position(0)
  plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
  plt.title("BIG 6", color=color6)
  plt.ylim(0, 1)

  ax = plt.subplot(121, polar="True")
  # ax.set_facecolor('#d1d1e0')
  plt.polar(angles3, data3, color=color3, linewidth=2, linestyle="--", alpha=.8)
  plt.fill(angles3, data3, facecolor=color3, alpha=0.25)

  ax.spines['polar'].set_color('lightgrey')
  ax.set_theta_offset(np.pi / 6)
  ax.set_theta_direction(-1)
  plt.xticks(angles3[:-1], categories3)
  ax.set_rlabel_position(0)
  plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
  plt.title("BIG 3", color=color3)
  plt.ylim(0, 1)
  plt.suptitle(title)
  plt.subplots_adjust(top=0.75)

def main():
    # st.title("Speech Emotion Recognition")
    # st.sidebar.markdown("## Use the menu to navigate on the site")

    # menu = ["Upload audio", "Dataset analysis", "About"]
    # choice = st.sidebar.selectbox("Menu", menu)
    choice = "Upload audio" ######################################################3
    if choice == "Upload audio":

        # st.subheader("Upload audio")
        # audio_file = st.file_uploader("Upload audio file", type=['wav'])

        # if st.button('Record'):
        #     with st.spinner(f'Recording for 5 seconds ....'):
        #         st.write("Recording...")
        #         time.sleep(3)
        #     st.success("Recording completed")
        audio_file = True #######################################################
        if audio_file is not None:
            # st.title("Analyzing...")
            # file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
            # st.write(file_details)
            # st.subheader(f"File {file_details['Filename']}")

            # st.audio(audio_file, format='audio/wav', start_time=0)

            path = os.path.join("audio", "whyhere.wav")
            # save_audio(audio_file)

            # extract features
            # wav, sr = librosa.load(path, sr=44100)
            # Xdb = get_melspec(path)[1]

            # fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
            # fig.set_facecolor('#d1d1e0')
            #
            # plt.subplot(211)
            # plt.title("Wave-form")
            # librosa.display.waveplot(wav, sr=sr)
            # plt.gca().axes.get_yaxis().set_visible(False)
            # plt.gca().axes.get_xaxis().set_visible(False)
            # plt.gca().axes.spines["right"].set_visible(False)
            # plt.gca().axes.spines["left"].set_visible(False)
            # plt.gca().axes.spines["top"].set_visible(False)
            # plt.gca().axes.spines["bottom"].set_visible(False)
            # plt.gca().axes.set_facecolor('#d1d1e0')
            #
            # plt.subplot(212)
            # plt.title("Mel-log-spectrogram")
            # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            # plt.gca().axes.get_yaxis().set_visible(False)
            # plt.gca().axes.spines["right"].set_visible(False)
            # plt.gca().axes.spines["left"].set_visible(False)
            # plt.gca().axes.spines["top"].set_visible(False)
            # # st.write(fig)
            # plt.show()

            # st.title("Getting the result...")

            mfccs = get_mfccs(path, model.input_shape[-1])
            mfccs = mfccs.reshape(1, *mfccs.shape)
            pred = model.predict(mfccs)[0]
            print([f"{i:.2f}" for i in pred])
            txt = get_title(pred)
            fig = plt.figure(figsize=(10, 4))
            plot_emotions(data6=pred, fig=fig, title=txt)
            plt.show()
            # st.write(fig)

            # mel = get_melspec(path)
            # mel = mel.reshape(1, *mel.shape)
            # tpred = model.predict(mel)[0]
            # txt = get_title(tpred)
            # fig = plt.figure(figsize=(10, 4))
            # plot_emotions(data3=data3, data6=tpred, fig=fig, title=txt)
            # st.write(fig)
    #
    # elif choice == "Dataset analysis":
    #     st.subheader("Dataset analysis")
    #     # with st.echo(code_location='below'):
    #
    #
    # else:
    #     st.subheader("About")
    #     st.info("maria.s.startseva@gmail.com")
    #     st.info("talbaram3192@gmail.com")
    #     st.info("asherholder123@gmail.com")


if __name__ == '__main__':
    main()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.


# st.button("Re-run")