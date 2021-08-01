import numpy as np
import cv2
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt

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
                  "disgust":"brown"}

TEST_CAT = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
TEST_PRED = np.array([.3,.3,.4,.1,.6,.9,.1])

# page settings
# st.set_page_config(page_title="SER web-app", page_icon=":speech_balloon:", layout="wide")

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


def get_title(predictions, categories, first_line=''):
    txt = f"{first_line}\nDetected emotion: \
  {categories[predictions.argmax()]} - {predictions.max() * 100:.2f}%"
    return txt


def plot_colored_polar(fig, predictions, categories,
                        title="", colors=COLOR_DICT):
    N = len(predictions)
    ind = predictions.argmax()

    COLOR = color_sector = colors[categories[ind]]
    sector_colors = [colors[i] for i in categories]

    fig.set_facecolor("#d1d1e0")
    ax = plt.subplot(111, polar="True")

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    for sector in range(predictions.shape[0]):
        radii = np.zeros_like(predictions)
        radii[sector] = predictions[sector] * 10
        width = np.pi / 1.8 * predictions
        c = sector_colors[sector]
        ax.bar(theta, radii, width=width, bottom=0.0, color=c, alpha=0.25)

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

    plt.suptitle(title, color="darkblue", size=10)
    plt.title(f"BIG {N}\n", color=COLOR)
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.75)

def plot_melspec(path, tmodel=None, three=False,
                 CAT3=CAT3, CAT6=CAT6):
    # load model if it is not loaded
    if tmodel is None:
        tmodel = load_model("tmodel_all.h5")
    # mel-spec model results
    mel = get_melspec(path)[0]
    mel = mel.reshape(1, *mel.shape)
    tpred = tmodel.predict(mel)[0]
    cat = CAT6

    if three:
        pos = tpred[3] + tpred[5] * .5
        neu = tpred[2] + tpred[5] * .5 + tpred[4] * .5
        neg = tpred[0] + tpred[1] + tpred[4] * .5
        tpred = np.array([pos, neu, neg])
        cat = CAT3

    txt = get_title(tpred, cat)
    fig = plt.figure(figsize=(6, 4))
    plot_colored_polar(fig, predictions=tpred, categories=cat, title=txt)
    return (fig, tpred)

if __name__ == "__main__":
    plot_melspec("test.wav")