import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

# Load model and class names
MODEL_PATH = 'D:\Elsayed\F_AI\Deep learning\project\DL_project4.keras'
CLASS_NAMES_PATH = 'D:\Elsayed\F_AI\Deep learning\project\class_names4.npy'

# Load the trained model
model = load_model(MODEL_PATH)

# Load class names
class_names = np.load(CLASS_NAMES_PATH)

# Title and team members
st.title("Animal Detection Project")
st.subheader("Team Members")
team_members = [
    "Ahmed Soudy Tawfik Ahmed",
    "Mustafa Gaser Mekhemar",
    "Elsayed Osama Elsayed",
    "Mahmoud Foad Sleem",
    "Islam Ragab Ahmed",
    "Ahmed Reda Farag"
]
for member in team_members:
    st.text(member)

# File upload section
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence_score = predictions[0][predicted_class_idx] * 100

    # Display prediction result
    st.success(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence Score: {confidence_score:.2f}%")