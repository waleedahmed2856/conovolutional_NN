import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("1.keras")

st.title("Plant Disease Detection 🌿")

# Upload image
uploaded_file = st.file_uploader("Upload Plant Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image.resize((128,128)))  
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    # Prediction
    prediction = model.predict(img_array)
    class_names = ['Early Blight', 'Late Blight', 'Healthy']  
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")