import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image as image_fn

# Load model architecture
from tensorflow.keras.models import model_from_json

# Load the model architecture from JSON
with open('alzheimer_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights('alzheimer_model_weights.h5')

# Define function to preprocess image
def preprocess_image(uploaded_file):
    img = image_fn.load_img(uploaded_file, target_size=(176, 176))
    img_array = image_fn.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Define function to predict disease
def predict_disease(image):
    prediction = loaded_model.predict(image)
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    result = class_names[np.argmax(prediction)]
    return result

# Streamlit UI
st.title('Alzheimer Disease Detection')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classified as...")

    result = predict_disease(image)
    st.success(f'The image is classified as: {result}')
