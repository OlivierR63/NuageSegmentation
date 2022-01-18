import streamlit as st
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
import os

sm.set_framework('tf.keras')
sm.framework()

from clouds_graph_functions import visualize_image_mask_prediction, plot_image_with_masks, plot_images_and_masks
from clouds_utilities_functions import np_resize, build_masks
from clouds_utilities_functions import dice_coef, dice_loss, bce_dice_loss, dice_coef_class
from clouds_utilities_functions import rle_to_mask, get_labels, get_mask_by_image_id, create_segmap, draw_labels, draw_segmentation_maps


def app():

    st.title("Segmentation sémantique images")

    ######params généraux #########

    HEIGHT = 320
    WIDTH = 480
    CHANNELS = 3 
    NB_CLASSES = 4

    weights_UNET_resnet50 = os.environ['STREAMLIT_HOME'] + "/pages/model_h5/UNET_resnet50.h5"

    weights_FPN_seresnext50 = os.environ['STREAMLIT_HOME'] + "\\pages\\model_h5\\FPN_seresnext50.h5"

    weights = {"UNET_resnet50": weights_UNET_resnet50,
               "FPN_seresnext50": weights_FPN_seresnext50,}


    def build_segmentation_model(choix_model, backbone, weights_file):
        ## création du modèle de segmentation

        MODELS = {"UNET": sm.Unet,
                    "FPN": sm.FPN}

        model = MODELS[choix_model](backbone, 
                                        classes=NB_CLASSES,
                                        input_shape=(HEIGHT, WIDTH, CHANNELS),
                                        encoder_weights='imagenet',
                                        activation='sigmoid',
                                        encoder_freeze=False)

        try:
            model.load_weights(weights_file)
        except ValueError:
            print('weights_file = ',weights_file)
            st.error("erreur chargement fichier .h5")

        return model


    def machine_segmentation(model, img):
    
        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, HEIGHT, WIDTH, CHANNELS), dtype=np.float32)
        image = img
        #image sizing
        size = (WIDTH, HEIGHT)
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


    def visualize_image_mask_prediction(image, mask_prediction, threshold):
        """ Fonction pour visualiser l'image original, le mask original et le mask predit"""

        mask_pred = np.zeros(mask_prediction.shape).astype(np.float32)
        mask_pred[mask_prediction > threshold] = 1

        class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
        cols = st.columns(4) 
        for i in range(4):
            title='class  ' + class_dict[i]
            cols[i].image(mask_pred[0, :, :, i], caption=title, width=150)


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
        train_images_path = os.environ['STREAMLIT_HOME'] + '/data/images/train_images\\'
 
        try:
            # draw segmentation maps and labels on image
            full_image_path = train_images_path + file_name
            image_in = draw_segmentation_maps(train_df, full_image_path)

            labels = get_labels(train_df, file_name)
            st.image(image_in, caption=labels, width=500)

        except:
            st.image(image, caption="Uploaded image", width=500)

        st.write("")
        MODELS = ["UNET_resnet50", "FPN_seresnext50"]

        MODEL_SELECTED = st.multiselect('Choix du modele', MODELS)
        seuil = st.sidebar.slider("Seuil", value= 0.5, min_value=0.0, max_value=1.0)

        for i in range(len(MODEL_SELECTED)):
            model, backbone = MODEL_SELECTED[i].split("_")

            st.write("Résultat de segmentation modèle : ", model, " et backbone : ", backbone)
            model = build_segmentation_model(model, backbone, weights[MODEL_SELECTED[i]])
            prediction = machine_segmentation(model, image)
 
            visualize_image_mask_prediction(image, prediction, seuil)



