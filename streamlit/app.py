import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# Custom imports 
from multipage import MultiPage
# import des pages
from pages import home, dataviz, exploration_images, classification, segmentation

HOME_STREAMLIT = 'F:\Travail\DataScientest\Git_sources\streamlit'

# Create an instance of the app 
app = MultiPage()

# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("DataFrame + dataviz", dataviz.app)
app.add_page("Exploration images", exploration_images.app)     
app.add_page("Classification multilabels", classification.app)
app.add_page("Segmentation sémantique", segmentation.app)

# The main app
app.run()