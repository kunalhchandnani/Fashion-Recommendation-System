import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os

# Load feature and image lists
try:
    features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
    img_files_list = pickle.load(open("img_files.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading feature or image files: {e}")

# Define the model
try:
    model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = Sequential([model, GlobalMaxPooling2D()])
except Exception as e:
    st.error(f"Error setting up the model: {e}")

st.title('Fashion Recommendation System')

def save_file(uploaded_file):
    try:
        os.makedirs("uploader", exist_ok=True)
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def extract_img_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expand_img = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expand_img)
        result_to_resnet = model.predict(preprocessed_img)
        flatten_result = result_to_resnet.flatten()
        result_normalized = flatten_result / norm(flatten_result)
        return result_normalized
    except Exception as e:
        st.error(f"Error extracting image features: {e}")
        return None

def recommend(features, features_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(features_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return None

uploaded_file = st.file_uploader("Choose your image")
if uploaded_file is not None:
    if save_file(uploaded_file):
        try:
            # Display the uploaded image
            show_images = Image.open(uploaded_file)
            size = (400, 400)
            resized_im = show_images.resize(size)
            st.image(resized_im)

            # Extract features of uploaded image
            features = extract_img_features(os.path.join("uploader", uploaded_file.name), model)
            if features is not None:
                img_indices = recommend(features, features_list)
                if img_indices is not None and len(img_indices) > 0:
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.header("I")
                        st.image(img_files_list[img_indices[0][0]])

                    with col2:
                        st.header("II")
                        st.image(img_files_list[img_indices[0][1]])

                    with col3:
                        st.header("III")
                        st.image(img_files_list[img_indices[0][2]])

                    with col4:
                        st.header("IV")
                        st.image(img_files_list[img_indices[0][3]])

                    with col5:
                        st.header("V")
                        st.image(img_files_list[img_indices[0][4]])
                else:
                    st.error("Could not retrieve recommendations. No indices returned.")
            else:
                st.error("Could not extract features from the image.")
        except Exception as e:
            st.error(f"An error occurred while processing the uploaded file: {e}")
    else:
        st.error("Some error occurred while saving the file.")
