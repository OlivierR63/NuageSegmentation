import streamlit as st
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from PIL import Image
import pandas as pd
import os

from clouds_graph_functions import plot_image_with_masks, plot_images_and_masks
from clouds_utilities_functions import rle_to_mask, get_labels, get_mask_by_image_id, create_segmap, draw_labels, draw_segmentation_maps


def app():

    header = st.container()
    images_explore = st.container()
    images_masks = st.container()

    with header:
        st.title("Exploration images")


    train_df = pd.read_csv(os.environ['STREAMLIT_HOME'] + "/data/train.csv")
    
    # Reorganisation du DataFrame
    split_df = train_df["Image_Label"].str.split("_", n = 1, expand = True)
    train_df['ImageId'] = split_df[0]
    train_df['Label'] = split_df[1]
    train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
        
    train_df = train_df.fillna('-1')

    # images path
    images_path = os.environ['STREAMLIT_HOME'] + '/data/images/train_images/'
        
    # get a list of images from training set
    images = sorted(glob(images_path + '*.jpg'))
        
    with images_explore:

        st.header("Echantillon d'images du jeu d'entrainement")

        width = 3
        height = 2
        
        # create a list of random indices 
        rnd_indices = [np.random.choice(range(0, len(images))) for i in range(height * width)]
		
        fig, axs = plt.subplots(height, width, figsize=(width * 6, height * 5))
        
        for im in range(0, height * width):
            # open image with a random index
            image = Image.open(images[rnd_indices[im]])
            
            i = im // width
            j = im % width
            
            # plot the image
            axs[i,j].imshow(image, aspect='auto')
            axs[i,j].axis('off')
            axs[i,j].set_title(get_labels(train_df, images[rnd_indices[im]].split('\\')[-1]), fontsize=20)

        st.pyplot(fig)
        
    with images_masks:

        st.header("Echantillon d'images avec leurs masques")

        width = 2
        height = 2
        
        fig2, axs2 = plt.subplots(height, width, figsize=(width * 6, height * 5))
        
        for im in range(0, height * width):
            # open image with a random index
            image = Image.open(images[im])
            # draw segmentation maps and labels on image
            image = draw_segmentation_maps(train_df, images[im])
           
            i = im // width
            j = im % width
           
            # plot the image
            axs2[i,j].imshow(image, aspect='auto') #plot the data
            axs2[i,j].axis('off')
            axs2[i,j].set_title(get_labels(train_df, images[im].split('\\')[-1]), fontsize=20)

        st.pyplot(fig2)



