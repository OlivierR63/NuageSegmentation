import streamlit as st
from PIL import Image
import pandas as pd
from glob import glob

from clouds_graph_functions import plot_image_with_masks, plot_images_and_masks
from clouds_utilities_functions import rle_to_mask, get_labels, get_mask_by_image_id, create_segmap, draw_labels, draw_segmentation_maps


train_df = pd.read_csv("data/train.csv")
    
# Reorganisation du DataFrame
split_df = train_df["Image_Label"].str.split("_", n = 1, expand = True)
train_df['ImageId'] = split_df[0]
train_df['Label'] = split_df[1]
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
        
train_df = train_df.fillna('-1')

# images path
images_path = 'data/images/train_images/'
        
# get a list of images from training set
images = sorted(glob(images_path + '*.jpg'))

img = images[0].split('\\')[-1]
labels = get_labels(train_df, img)

print(img, type(img))

print(labels)
    