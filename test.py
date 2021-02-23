import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# load models
try:
    model = load_model("model2.h5")
except:
    model=None

try:
    tmodel = load_model("tmodel_all.h5")
except:
    model=None

# page settings
st.set_page_config(layout="wide")

max_width = 1000
padding_top = 0
padding_right = "20%"
padding_left = "10%"
padding_bottom = 0
COLOR = "#1f1f2e"
BACKGROUND_COLOR = "#d1d1e0"

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    np.random.seed(0)
    st.title("Test")
    if model == None:
        st.write("Error: model #1 is not loaded")
    if tmodel == None:
        st.write("Error: model #2 is not loaded")
    st.write("Hello world")