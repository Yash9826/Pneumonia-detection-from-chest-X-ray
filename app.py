
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load trained model
model = load_model("pneumonia_cnn_model.h5")

st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to predict if it shows pneumonia.")

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        img_array = np.array(img)
        
        # ======= X-ray Validation =======
        is_valid = True
        warning_message = ""
        
        # Check if image has 2D or 3D array
        if len(img_array.shape) < 2:
            is_valid = False
            warning_message = "Invalid image: cannot detect dimensions."
        
        # Check if image is very small
        elif img_array.size < 50000:
            is_valid = False
            warning_message = "Image too small to be a chest X-ray."
        
        # Check aspect ratio (X-rays are usually taller)
        elif len(img_array.shape) == 3:
            height, width, channels = img_array.shape
            if channels != 3:
                is_valid = False
                warning_message = "Image is not standard RGB. Check the file."
            elif height/width < 0.8:
                is_valid = False
                warning_message = "Image aspect ratio not typical for chest X-ray."
        else:
            height, width = img_array.shape
            if height/width < 0.8:
                is_valid = False
                warning_message = "Image aspect ratio not typical for chest X-ray."
        
        if not is_valid:
            st.warning(f"Uploaded image may not be a chest X-ray. {warning_message}")
        else:
            # ======= Preprocess and Predict =======
            img_resized = img.resize((150,150))
            img_array = np.array(img_resized)/255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            pred = model.predict(img_array)[0][0]
            if pred < 0.5:
                st.success(f"Prediction: Normal ({pred:.2f})")
            else:
                st.error(f"Prediction: Pneumonia ({pred:.2f})")
    
    except Exception as e:
        st.error(f"Error processing the image: {e}")



