import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os


def app():

    header = st.container()
    dataset = st.container()

    with header:
        st.title("Données et dataviz")
    
    with dataset:
        st.header("DataFrame initial")
    
        train_df = pd.read_csv(os.environ['STREAMLIT_HOME'] + "/data/train.csv")
        st.write(train_df.head())
    
        # Reorganisation du DataFrame
        split_df = train_df["Image_Label"].str.split("_", n = 1, expand = True)
        train_df['ImageId'] = split_df[0]
        train_df['Label'] = split_df[1]
        train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
    
        ##### Répartition des différents types de nuages #######
        fish = train_df[train_df['Label'] == 'Fish'].EncodedPixels.count()
        flower = train_df[train_df['Label'] == 'Flower'].EncodedPixels.count()
        gravel = train_df[train_df['Label'] == 'Gravel'].EncodedPixels.count()
        sugar = train_df[train_df['Label'] == 'Sugar'].EncodedPixels.count()
    
        labels_size = {'Fish': fish, 'Flower': flower, 'Gravel': gravel, 'Sugar': sugar}
    
        st.subheader("Répartition des différents types de nuages")
        fig, ax = plt.subplots()
        ax.pie(labels_size.values(), labels=labels_size.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
    
        st.pyplot(fig)
        
        ##### Répartition des labels par image #######
        labels_per_image = train_df.groupby('ImageId')['EncodedPixels'].count()
        
        st.subheader("Histogramme de répartition du nombre de labels par image")
        
        fig1, ax1 = plt.subplots()
        ax1.hist(labels_per_image)
        
        st.pyplot(fig1)