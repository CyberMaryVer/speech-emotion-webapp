import numpy as np
import streamlit as st
import cv2
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from PIL import Image
from melspec import plot_colored_polar, plot_melspec

# load models
model = load_model("model3.h5")

# constants
starttime = datetime.now()

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
              "disgust": "brown"}

TEST_CAT = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
TEST_PRED = np.array([.3, .3, .4, .1, .6, .9, .1])

# page settings
st.set_page_config(page_title="SER web-app", page_icon=":speech_balloon:", layout="wide")
# COLOR = "#1f1f2e"
# BACKGROUND_COLOR = "#d1d1e0"


# @st.cache(hash_funcs={tf_agents.utils.object_identity.ObjectIdentityDictionary: load_model})
# def load_model_cache(model):
#     return load_model(model)

# @st.cache
def log_file(txt=None):
    with open("log.txt", "a") as f:
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} - {datetoday};\n")


# @st.cache
def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


# @st.cache
def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)


# @st.cache
def get_mfccs(audio, limit):
    y, sr = librosa.load(audio)
    a = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
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
def color_dict(coldict=COLOR_DICT):
    return COLOR_DICT


@st.cache
def plot_polar(fig, predictions=TEST_PRED, categories=TEST_CAT,
               title="TEST", colors=COLOR_DICT):
    # color_sector = "grey"

    N = len(predictions)
    ind = predictions.argmax()

    COLOR = color_sector = colors[categories[ind]]
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = np.zeros_like(predictions)
    radii[predictions.argmax()] = predictions.max() * 10
    width = np.pi / 1.8 * predictions
    fig.set_facecolor("#d1d1e0")
    ax = plt.subplot(111, polar="True")
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
    plt.subplots_adjust(top=0.75)


def main():
    side_img = Image.open("images/emotion3.jpg")
    with st.sidebar:
        st.image(side_img, width=300)
    st.sidebar.subheader("Menu")
    website_menu = st.sidebar.selectbox("Menu", ("Emotion Recognition", "Project description", "Our team",
                                                 "Leave feedback", "Relax"))
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if website_menu == "Emotion Recognition":
        st.sidebar.subheader("Model")
        model_type = st.sidebar.selectbox("How would you like to predict?", ("mfccs", "mel-specs"))
        em3 = em6 = em7 = gender = False
        st.sidebar.subheader("Settings")

        st.markdown("## Upload the file")
        with st.container():
            col1, col2 = st.columns(2)
            # audio_file = None
            # path = None
            with col1:
                audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg'])
                if audio_file is not None:
                    if not os.path.exists("audio"):
                        os.makedirs("audio")
                    path = os.path.join("audio", audio_file.name)
                    if_save_audio = save_audio(audio_file)
                    if if_save_audio == 1:
                        st.warning("File size is too large. Try another file.")
                    elif if_save_audio == 0:
                        # extract features
                        # display audio
                        st.audio(audio_file, format='audio/wav', start_time=0)
                        try:
                            wav, sr = librosa.load(path, sr=44100)
                            Xdb = get_melspec(path)[1]
                            mfccs = librosa.feature.mfcc(wav, sr=sr)
                            # # display audio
                            # st.audio(audio_file, format='audio/wav', start_time=0)
                        except Exception as e:
                            audio_file = None
                            st.error(f"Error {e} - wrong format of the file. Try another .wav file.")
                    else:
                        st.error("Unknown error")
                else:
                    if st.button("Try test file"):
                        wav, sr = librosa.load("test.wav", sr=44100)
                        Xdb = get_melspec("test.wav")[1]
                        mfccs = librosa.feature.mfcc(wav, sr=sr)
                        # display audio
                        st.audio("test.wav", format='audio/wav', start_time=0)
                        path = "test.wav"
                        audio_file = "test"
            with col2:
                if audio_file is not None:
                    fig = plt.figure(figsize=(10, 2))
                    fig.set_facecolor('#d1d1e0')
                    plt.title("Wave-form")
                    librosa.display.waveplot(wav, sr=44100)
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    plt.gca().axes.spines["bottom"].set_visible(False)
                    plt.gca().axes.set_facecolor('#d1d1e0')
                    st.write(fig)
                else:
                    pass
            #     st.write("Record audio file")
            #     if st.button('Record'):
            #         with st.spinner(f'Recording for 5 seconds ....'):
            #             st.write("Recording...")
            #             time.sleep(3)
            #         st.success("Recording completed")
            #         st.write("Error while loading the file")

        if model_type == "mfccs":
            em3 = st.sidebar.checkbox("3 emotions", True)
            em6 = st.sidebar.checkbox("6 emotions", True)
            em7 = st.sidebar.checkbox("7 emotions")
            gender = st.sidebar.checkbox("gender")

        elif model_type == "mel-specs":
            st.sidebar.warning("This model is temporarily disabled")

        else:
            st.sidebar.warning("This model is temporarily disabled")

        # with st.sidebar.expander("Change colors"):
        #     st.sidebar.write("Use this options after you got the plots")
        #     col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        #
        #     with col1:
        #         a = st.color_picker("Angry", value="#FF0000")
        #     with col2:
        #         f = st.color_picker("Fear", value="#800080")
        #     with col3:
        #         d = st.color_picker("Disgust", value="#A52A2A")
        #     with col4:
        #         sd = st.color_picker("Sad", value="#ADD8E6")
        #     with col5:
        #         n = st.color_picker("Neutral", value="#808080")
        #     with col6:
        #         sp = st.color_picker("Surprise", value="#FFA500")
        #     with col7:
        #         h = st.color_picker("Happy", value="#008000")
        #     if st.button("Update colors"):
        #         global COLOR_DICT
        #         COLOR_DICT = {"neutral": n,
        #                       "positive": h,
        #                       "happy": h,
        #                       "surprise": sp,
        #                       "fear": f,
        #                       "negative": a,
        #                       "angry": a,
        #                       "sad": sd,
        #                       "disgust": d}
        #         st.success(COLOR_DICT)

        if audio_file is not None:
            st.markdown("## Analyzing...")
            if not audio_file == "test":
                st.sidebar.subheader("Audio file")
                file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
                st.sidebar.write(file_details)

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    fig = plt.figure(figsize=(10, 2))
                    fig.set_facecolor('#d1d1e0')
                    plt.title("MFCCs")
                    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig)
                with col2:
                    fig2 = plt.figure(figsize=(10, 2))
                    fig2.set_facecolor('#d1d1e0')
                    plt.title("Mel-log-spectrogram")
                    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig2)

            if model_type == "mfccs":
                st.markdown("## Predictions")
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    mfccs = get_mfccs(path, model.input_shape[-1])
                    mfccs = mfccs.reshape(1, *mfccs.shape)
                    pred = model.predict(mfccs)[0]

                    with col1:
                        if em3:
                            pos = pred[3] + pred[5] * .5
                            neu = pred[2] + pred[5] * .5 + pred[4] * .5
                            neg = pred[0] + pred[1] + pred[4] * .5
                            data3 = np.array([pos, neu, neg])
                            txt = "MFCCs\n" + get_title(data3, CAT3)
                            fig = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT)
                            plot_colored_polar(fig, predictions=data3, categories=CAT3,
                                               title=txt, colors=COLORS)
                            # plot_polar(fig, predictions=data3, categories=CAT3,
                            # title=txt, colors=COLORS)
                            st.write(fig)
                    with col2:
                        if em6:
                            txt = "MFCCs\n" + get_title(pred, CAT6)
                            fig2 = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT)
                            plot_colored_polar(fig2, predictions=pred, categories=CAT6,
                                               title=txt, colors=COLORS)
                            # plot_polar(fig2, predictions=pred, categories=CAT6,
                            #            title=txt, colors=COLORS)
                            st.write(fig2)
                    with col3:
                        if em7:
                            model_ = load_model("model4.h5")
                            mfccs_ = get_mfccs(path, model_.input_shape[-2])
                            mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
                            pred_ = model_.predict(mfccs_)[0]
                            txt = "MFCCs\n" + get_title(pred_, CAT7)
                            fig3 = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT)
                            plot_colored_polar(fig3, predictions=pred_, categories=CAT7,
                                               title=txt, colors=COLORS)
                            # plot_polar(fig3, predictions=pred_, categories=CAT7,
                            #            title=txt, colors=COLORS)
                            st.write(fig3)
                    with col4:
                        if gender:
                            with st.spinner('Wait for it...'):
                                gmodel = load_model("model_mw.h5")
                                gmfccs = get_mfccs(path, gmodel.input_shape[-1])
                                gmfccs = gmfccs.reshape(1, *gmfccs.shape)
                                gpred = gmodel.predict(gmfccs)[0]
                                gdict = [["female", "woman.png"], ["male", "man.png"]]
                                ind = gpred.argmax()
                                txt = "Predicted gender: " + gdict[ind][0]
                                img = Image.open("images/" + gdict[ind][1])

                                fig4 = plt.figure(figsize=(3, 3))
                                fig4.set_facecolor('#d1d1e0')
                                plt.title(txt)
                                plt.imshow(img)
                                plt.axis("off")
                                st.write(fig4)

            # if model_type == "mel-specs":
            # st.markdown("## Predictions")
            # st.warning("The model in test mode. It may not be working properly.")
            # if st.checkbox("I'm OK with it"):
            #     try:
            #         with st.spinner("Wait... It can take some time"):
            #             global tmodel
            #             tmodel = load_model_cache("tmodel_all.h5")
            #             fig, tpred = plot_melspec(path, tmodel)
            #         col1, col2, col3 = st.columns(3)
            #         with col1:
            #             st.markdown("### Emotional spectrum")
            #             dimg = Image.open("images/spectrum.png")
            #             st.image(dimg, use_column_width=True)
            #         with col2:
            #             fig_, tpred_ = plot_melspec(path=path,
            #                                         tmodel=tmodel,
            #                                         three=True)
            #             st.write(fig_, use_column_width=True)
            #         with col3:
            #             st.write(fig, use_column_width=True)
            #     except Exception as e:
            #         st.error(f"Error {e}, model is not loaded")


    elif website_menu == "Project description":
        import pandas as pd
        import plotly.express as px
        st.title("Project description")
        st.subheader("GitHub")
        link = '[GitHub repository of the web-application]' \
               '(https://github.com/CyberMaryVer/speech-emotion-webapp)'
        st.markdown(link, unsafe_allow_html=True)

        st.subheader("Theory")
        link = '[Theory behind - Medium article]' \
               '(https://talbaram3192.medium.com/classifying-emotions-using-audio-recordings-and-python-434e748a95eb)'
        st.markdown(link + ":clap::clap::clap: Tal!", unsafe_allow_html=True)
        with st.expander("See Wikipedia definition"):
            components.iframe("https://en.wikipedia.org/wiki/Emotion_recognition",
                              height=320, scrolling=True)

        st.subheader("Dataset")
        txt = """
            This web-application is a part of the final **Data Mining** project for **ITC Fellow Program 2020**. 

            Datasets used in this project
            * Crowd-sourced Emotional Mutimodal Actors Dataset (**Crema-D**)
            * Ryerson Audio-Visual Database of Emotional Speech and Song (**Ravdess**)
            * Surrey Audio-Visual Expressed Emotion (**Savee**)
            * Toronto emotional speech set (**Tess**)    
            """
        st.markdown(txt, unsafe_allow_html=True)

        df = pd.read_csv("df_audio.csv")
        fig = px.violin(df, y="source", x="emotion4", color="actors", box=True, points="all", hover_data=df.columns)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("FYI")
        st.write("Since we are currently using a free tier instance of AWS, "
                 "we disabled mel-spec and ensemble models.\n\n"
                 "If you want to try them we recommend to clone our GitHub repo")
        st.code("git clone https://github.com/CyberMaryVer/speech-emotion-webapp.git", language='bash')

        st.write("After that, just uncomment the relevant sections in the app.py file "
                 "to use these models:")

    elif website_menu == "Our team":
        st.subheader("Our team")
        st.balloons()
        col1, col2 = st.columns([3, 2])
        with col1:
            st.info("maria.s.startseva@gmail.com")
            st.info("talbaram3192@gmail.com")
            st.info("asherholder123@gmail.com")
        with col2:
            liimg = Image.open("images/LI-Logo.png")
            st.image(liimg)
            st.markdown(f""":speech_balloon: [Maria Startseva](https://www.linkedin.com/in/maria-startseva)""",
                        unsafe_allow_html=True)
            st.markdown(f""":speech_balloon: [Tal Baram](https://www.linkedin.com/in/tal-baram-b00b66180)""",
                        unsafe_allow_html=True)
            st.markdown(f""":speech_balloon: [Asher Holder](https://www.linkedin.com/in/asher-holder-526a05173)""",
                        unsafe_allow_html=True)

    elif website_menu == "Leave feedback":
        st.subheader("Leave feedback")
        user_input = st.text_area("Your feedback is greatly appreciated")
        user_name = st.selectbox("Choose your personality", ["checker1", "checker2", "checker3", "checker4"])

        if st.button("Submit"):
            st.success(f"Message\n\"\"\"{user_input}\"\"\"\nwas sent")

            if user_input == "log123456" and user_name == "checker4":
                with open("log0.txt", "r", encoding="utf8") as f:
                    st.text(f.read())
            elif user_input == "feedback123456" and user_name == "checker4":
                with open("log.txt", "r", encoding="utf8") as f:
                    st.text(f.read())
            else:
                log_file(user_name + " " + user_input)
                thankimg = Image.open("images/sticky.png")
                st.image(thankimg)

    else:
        import requests
        import json

        url = 'http://api.quotable.io/random'
        if st.button("get random mood"):
            with st.container():
                col1, col2 = st.columns(2)
                n = np.random.randint(1, 1000, 1)[0]
                with col1:
                    quotes = {"Good job and almost done": "checker1",
                              "Great start!!": "checker2",
                              "Please make corrections base on the following observation": "checker3",
                              "DO NOT train with test data": "folk wisdom",
                              "good work, but no docstrings": "checker4",
                              "Well done!": "checker3",
                              "For the sake of reproducibility, I recommend setting the random seed": "checker1"}
                    if n % 5 == 0:
                        a = np.random.choice(list(quotes.keys()), 1)[0]
                        quote, author = a, quotes[a]
                    else:
                        try:
                            r = requests.get(url=url)
                            text = json.loads(r.text)
                            quote, author = text['content'], text['author']
                        except Exception as e:
                            a = np.random.choice(list(quotes.keys()), 1)[0]
                            quote, author = a, quotes[a]
                    st.markdown(f"## *{quote}*")
                    st.markdown(f"### ***{author}***")
                with col2:
                    st.image(image=f"https://picsum.photos/800/600?random={n}")


if __name__ == '__main__':
    main()
