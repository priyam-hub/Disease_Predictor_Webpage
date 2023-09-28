import streamlit as st
from PIL import Image
from multiapp import MultiApp
from Apps import Hypertension_App, Stroke_App, Heart_Disease # import your app modules here
import base64
from pathlib import Path


def load_bootstrap():
    return st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

app = MultiApp()
st.set_page_config(
    page_title="Disease Predictor App",
    page_icon=Image.open("images/medical-team.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


load_bootstrap()

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

st.markdown(
    """
    <style>
    .markdown-section {
        margin-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1,1],gap="small")
with col1:
    st.markdown(img_to_html('images/medical-team.png'), unsafe_allow_html=True)
    col1.empty()
with col2:
    col2.empty()
    st.markdown("""
    # Disease Detector App

    **In the realm of healthcare, predicting diseases before they manifest can be a game-changer. 
    It can lead to early interventions, better management of health conditions, and improved patient outcomes. 
    To this end, we propose the development of a Disease Prediction Model using Machine Learning (ML) techniques.**

    This model will analyze various health parameters of an individual and predict the likelihood of them developing a specific disease.

    _The parameters could include_ `age, gender, lifestyle habits, genetic factors, and existing health conditions` _, among others._
    """)
st.write()

# Add all your application here
app.add_app("Heart Disease Detector", Heart_Disease.app)
app.add_app("Hypertension Detector", Hypertension_App.app)
app.add_app("Stroke Detector", Stroke_App.app)
# The main app
app.run()