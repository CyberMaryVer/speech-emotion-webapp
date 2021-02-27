import time
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
import cv2
import librosa
import librosa.display
from datetime import datetime
from tensorflow.keras.models import load_model
from PIL import Image

# load models
model = load_model("model3.h5")

# constants
CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']
CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
CAT3 = ["positive", "neutral", "negative"]
COLOR_DICT = {"neutral": "grey",
              "positive": "green",
              "happy": "green",
              "surprise": "orange",
              "fear": "purple",
              "negative": "red",
              "angry": "red",
              "sad": "lightblue",
              "disgust":"brown"}
# page settings
st.set_page_config(layout="wide", page_title="Speech Emotion Recognition app", page_icon="random")

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1000}px;
        padding-top: {0}rem;
        padding-right: {"20%"}rem;
        padding-left: {"10%"}rem;
        padding-bottom: {0}rem;
    }}
    .reportview-container .main {{
        color: {"#1f1f2e"};
        background-color: {"#d1d1e0"};
    }}
</style>
""",
        unsafe_allow_html=True,
    )

def log_file(txt=None):
    with open(os.path.join("log.txt"), "a") as f:
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} - {datetoday};\n")

@st.cache
def save_audio(file):
    folder = "audio"
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    with open(os.path.join("test.txt"), "a") as f:
        f.write(f"{file.name} - {file.size};\n")

    with open(os.path.join(folder, file.name), "wb") as f:
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
  a = librosa.feature.mfcc(y, sr=sr, n_mfcc = 40)
  if a.shape[1] > limit:
    mfccs = a[:,:limit]
  elif a.shape[1] < limit:
    mfccs = np.zeros((a.shape[0], limit))
    mfccs[:, :a.shape[1]] = a
  return mfccs

@st.cache
def get_title(predictions, categories=CAT6):
    title = f"Detected emotion: {categories[predictions.argmax()]} \
    - {predictions.max() * 100:.2f}%"
    return title

@st.cache
def plot_polar(fig, predictions, categories, title,
               colors=COLOR_DICT):
    # color_sector = "grey"

    N = len(predictions)
    ind = predictions.argmax()

    COLOR = color_sector = colors[categories[ind]]
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = np.zeros_like(predictions)
    radii[predictions.argmax()] = predictions.max() * 10
    width = np.pi / 1.8 * predictions
    fig.set_facecolor("#d1d1e0")
    ax = plt.subplot(121, polar="True")
    ax.bar(theta, radii, width=width, bottom=0.0, color=color_sector, alpha=0.25)

    angles = [i / float(N) * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    data = list(predictions)
    data += data[:1]
    plt.polar(angles, data, color=COLOR, linewidth=2)
    plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
    plt.suptitle(title, color="darkblue", size=12)
    plt.title(f"BIG {N}\n", color=COLOR)
    plt.ylim(0, 1)
    ax = plt.subplot(122)
    img = Image.open("images/spectrum.png")
    plt.imshow(img)
    ################################################################################
    plt.subplots_adjust(top=0.75)
    plt.axis("off")

@st.cache
def plot_emotions(fig, data6, data3=None, title="Detected emotion",
                  categories6=CAT6, categories3=CAT3, color_dict=COLOR_DICT):

  if data3 is None:
      pos = data6[3]
      neu = data6[2] + data6[5]
      neg = data6[0] + data6[1] + data6[4]
      data3 = np.array([pos, neu, neg])

  ind = categories6[data6.argmax()]
  color6 = color_dict[ind]

  # parameters for sector highlighting #6
  theta6 = np.linspace(0.0, 2 * np.pi, data6.shape[0], endpoint=False)
  radii6 = np.zeros_like(data6)
  radii6[data6.argmax()] = data6.max() * 10
  width6 = np.pi / 1.8 * data6

  data6 = list(data6)
  n = len(data6)
  data6 += data6[:1]
  angles6 = [i/float(n)*2*np.pi for i in range(n)]
  angles6 += angles6[:1]

  ind = categories3[data3.argmax()]
  color3 = color_dict[ind]

  # parameters for sector highlighting #3
  theta3 = np.linspace(0.0, 2 * np.pi, data3.shape[0], endpoint=False)
  radii3 = np.zeros_like(data3)
  radii3[data3.argmax()] = data3.max() * 10
  width3 = np.pi / 1.8 * data3

  data3 = list(data3)
  n = len(data3)
  data3 += data3[:1]
  angles3 = [i/float(n)*2*np.pi for i in range(n)]
  angles3 += angles3[:1]

  # fig = plt.figure(figsize=(10, 4))
  fig.set_facecolor('#d1d1e0')

  ax = plt.subplot(122, polar="True")
  plt.polar(angles6, data6, color=color6)
  plt.fill(angles6, data6, facecolor=color6, alpha=0.25)
  ax.bar(theta6, radii6, width=width6, bottom=0.0, color=color6, alpha=0.25)
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
  ax.bar(theta3, radii3, width=width3, bottom=0.0, color=color3, alpha=0.25)
  ax.spines['polar'].set_color('lightgrey')
  ax.set_theta_offset(np.pi / 6)
  ax.set_theta_direction(-1)
  plt.xticks(angles3[:-1], categories3)
  ax.set_rlabel_position(0)
  plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
  plt.title("BIG 3", color=color3)
  plt.ylim(0, 1)
  plt.suptitle(title, color="darkblue", size=12)
  plt.subplots_adjust(top=0.75)

def main():
    st.title("Speech Emotion Recognition")
    st.sidebar.markdown("### Use the menu to navigate on the site")
    img = Image.open("images/emotion3.jpg")
    with st.sidebar:
        st.image(img, width=300)

    menu = ["Emotion recognition", "Dataset description", "Our team", "Leave feedback"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Emotion recognition":
        audio_file = st.file_uploader("Upload audio file", type=['wav'])
        if st.button('Record'):
            with st.sidebar.spinner(f'Recording for 5 seconds ....'):
                st.sidebar.write("Recording...")
                time.sleep(3)
            st.sidebar.success("Recording completed")



        if audio_file is not None:
            st.markdown("## Analyzing...")
            st.sidebar.subheader("Audio file")
            file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
            st.sidebar.write(file_details)
            st.audio(audio_file, format='audio/wav', start_time=0)

            st.sidebar.markdown("### Settings:")
            show_more_labels = st.sidebar.checkbox("Show prediction for 7 emotions")
            show_mel = st.sidebar.checkbox("Show Mel-spec model prediction")
            show_gender = st.sidebar.checkbox("Show gender prediction")

            path = os.path.join("audio", audio_file.name)
            save_audio(audio_file)

            # extract features
            wav, sr = librosa.load(path, sr=44100)
            Xdb = get_melspec(path)[1]

            fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
            fig.set_facecolor('#d1d1e0')

            plt.subplot(211)
            plt.title("Wave-form")
            librosa.display.waveplot(wav, sr=sr)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.spines["right"].set_visible(False)
            plt.gca().axes.spines["left"].set_visible(False)
            plt.gca().axes.spines["top"].set_visible(False)
            plt.gca().axes.spines["bottom"].set_visible(False)
            plt.gca().axes.set_facecolor('#d1d1e0')

            plt.subplot(212)
            plt.title("Mel-log-spectrogram")
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.spines["right"].set_visible(False)
            plt.gca().axes.spines["left"].set_visible(False)
            plt.gca().axes.spines["top"].set_visible(False)
            st.write(fig)

            st.markdown("## Getting the result...")

            # mfccs model results
            with st.spinner('Wait for it...'):
                mfccs = get_mfccs(path, model.input_shape[-1])
                mfccs = mfccs.reshape(1, *mfccs.shape)
                pred = model.predict(mfccs)[0]
                txt = "MFCCs\n" + get_title(pred)
                fig = plt.figure(figsize=(10, 4))
                plot_emotions(data6=pred, fig=fig, title=txt)
                st.write(fig)

            if show_more_labels:
                with st.spinner('Wait for it...'):
                    model_ = load_model("model4.h5")
                    mfccs_ = get_mfccs(path, model_.input_shape[-2])
                    mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
                    pred = model_.predict(mfccs_)[0]
                    txt = "MFCCs\n" + get_title(pred, CAT7)
                    fig = plt.figure(figsize=(10, 4))
                    plot_polar(fig, predictions=pred, categories=CAT7, title=txt)
                    st.write(fig)

            if show_gender:
                with st.spinner('Wait for it...'):
                    gmodel = load_model("model_mw.h5")
                    gmfccs = get_mfccs(path, gmodel.input_shape[-1])
                    gmfccs = gmfccs.reshape(1, *gmfccs.shape)
                    gpred = gmodel.predict(gmfccs)[0]
                    gdict = [["female","woman.png"], ["male","man.png"]]
                    ind = gpred.argmax()
                    txt = "Predicted gender: " + gdict[ind][0]
                    st.subheader(txt)
                    img = Image.open("images/"+ gdict[ind][1])
                    st.image(img, width=300)

            if show_mel:
                #################################################################################
                st.subheader("This section was disabled")
                st.write("Since we are currently using a free tier instance of AWS, "
                         "we are not going to deploy this model.\n\n"
                         "If you want to try it we recommend to clone our GitHub repo")
                link = '[GitHub](https://github.com/CyberMaryVer/speech-emotion-webapp)'
                st.markdown(link, unsafe_allow_html=True)

                st.write("After that, just uncomment this section in the main file "
                         "to use the mel-spectrograms model:")
                code = '''
                # tmodel = load_model("tmodel_all.h5")
                #
                # # mel-spec model results
                # mel = get_melspec(path)[0]
                # mel = mel.reshape(1, *mel.shape)
                # tpred = tmodel.predict(mel)[0]
                # txt = "Mel-spectrograms" + get_title(tpred)
                # fig = plt.figure(figsize=(10, 4))
                # plot_emotions(data6=tpred, fig=fig, title=txt)
                # st.write(fig)'''
                st.code(code, language='python')
                #################################################################################

                ############## Uncomment this section below to enable the model #################
                # tmodel = load_model("tmodel_all.h5")
                #
                # # mel-spec model results
                # mel = get_melspec(path)[0]
                # mel = mel.reshape(1, *mel.shape)
                # tpred = tmodel.predict(mel)[0]
                # txt = "Mel-spectrograms\n" + get_title(tpred)
                # fig = plt.figure(figsize=(10, 4))
                # plot_emotions(data6=tpred, fig=fig, title=txt)
                # st.write(fig)
                #################################################################################

    elif choice == "Dataset description":
        st.subheader("Dataset analysis")
        link = '[GitHub](https://github.com/talbaram3192/Emotion_Recognition)'
        st.markdown(link, unsafe_allow_html=True)

        df = pd.read_csv("df_audio.csv")
        fig = px.violin(df, y="source", x="emotion4", color="actors", box=True, points="all", hover_data=df.columns)
        st.plotly_chart(fig, use_container_width=True)
        # st.write(df.source.value_counts())
        # st.write(df.actors.value_counts())
        # st.write(df.emotion4.value_counts())

    elif choice == "Our team":
        st.subheader("Our team")
        st.info("maria.s.startseva@gmail.com")
        st.info("talbaram3192@gmail.com")
        st.info("asherholder123@gmail.com")
        st.balloons()

    else:
        st.subheader("Leave feedback")
        user_input = st.text_area("Your feedback is greatly appreciated")
        user_name = st.selectbox("Choose your personality", ["checker1","checker2","checker3","checker4"])
        if st.button("Submit"):
            log_file(user_name + " " + user_input)
            st.success(f"Message\n\"\"\"{user_input}\"\"\"\nwas sent")
            thankimg = Image.open("images/sticky.png")
            st.image(thankimg)

if __name__ == '__main__':
    main()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.


st.button("Re-run")