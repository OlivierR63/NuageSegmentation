import streamlit as st
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam

from clouds_utilities_functions import get_labels


def app():

    st.title("Classification images multilabels")

    ######params généraux #########

    HEIGHT = 224
    WIDTH = 224
    CHANNELS = 3 
    NB_CLASSES = 4

    weights_file = os.environ['STREAMLIT_HOME'] + "\pages\model_h5\model_vgg16_224_224.h5"

    def VGG16_classification_model(weights_file):

        base_model = VGG16(include_top=False,
                       weights="imagenet",
                       input_shape=(HEIGHT, WIDTH, CHANNELS))

        # entrainement des couches du modele
        for layer in base_model.layers:
            layer.trainable = False

        # Construction du modele
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(units = NB_CLASSES, activation = "sigmoid"))
        # compilation du modele
        model.compile(loss="binary_crossentropy", optimizer= "adam")

        if weights_file is not None:
            model.load_weights(weights_file)

        return model


    def machine_classification(model, img):
    
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, HEIGHT, WIDTH, CHANNELS), dtype=np.float32)
        image = img
        #image sizing
        size = (HEIGHT, WIDTH)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = image_array.astype(np.float32) / 255

        # Load the image into the array
        data[0] = normalized_image_array

        # prediction of model
        prediction = model.predict(data)

        return prediction

    # DataFrame
    train_df = pd.read_csv(os.environ['STREAMLIT_HOME'] + "/data/train.csv")
    
    # Reorganisation du DataFrame
    split_df = train_df["Image_Label"].str.split("_", n = 1, expand = True)
    train_df['ImageId'] = split_df[0]
    train_df['Label'] = split_df[1]
    train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
        
    train_df = train_df.fillna('-1')

    uploaded_file = st.file_uploader("Choisir une image", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        file_name = str(uploaded_file.name)
        try:
            labels = get_labels(train_df, file_name)
            st.image(image, caption=labels, width=500)
        except:
            st.image(image, caption="Uploaded image", width=500)
        
        st.write("")
        st.write("Résultat de classification avec le modèle VGG16:")
        model = VGG16_classification_model(weights_file)
        prediction_labels = machine_classification(model, image)[0].round(2)

        st.text("[Fish, Flower, Sugar, Gravel]")
        st.text(prediction_labels)
