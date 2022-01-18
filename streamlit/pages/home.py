import streamlit as st
from PIL import Image
import os


def app():

    header = st.container()

    with header:
        st.title("Projet de segmentation de régions nuageuses")
        
        img = Image.open(os.environ['STREAMLIT_HOME'] + "\pages\image_intro.jpg")
        st.image(img)
    