import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

# Hide deprecation warnings
import warnings
warnings.filterwarnings("ignore")

# Function to load and cache the model for faster performance
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.keras')
    return model

def predict_image(image_path, model):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(48, 48), color_mode="grayscale")
    
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale the image
    img_array /= 255.0  # Assuming the model was trained with rescaled images
    
    # Make a prediction
    prediction = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    predicted_class_label = class_labels[predicted_class_index]
    predicted_probability = prediction[0][predicted_class_index]
    
    return predicted_class_label, predicted_probability

# Set page configuration
st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'Emotion Detection'

def set_tab(tab_name):
    st.session_state.selected_tab = tab_name

st.title("😉 Face Emotion Recognition")

# Load the model once
model = load_model()

tab_options = ["Emotion Detection", "Camera Input", "About"]
selected_tab = st.session_state.selected_tab

with st.sidebar:
    selected_tab = st.radio("Navigation", tab_options, index=tab_options.index(selected_tab), key='selected_tab', on_change=None)


if selected_tab == "Emotion Detection":
    st.header("Detect Emotions from Images")
    st.markdown("Upload an image to detect the emotion expressed on the face.")
    file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if file:
        # Display the uploaded image and prediction results side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(file, use_container_width =True)

        with col2:
            st.subheader("Prediction")
            cls, prob = predict_image(file, model)
            st.write(f"**Detected Emotion:** {cls}")
            st.write(f"**Confidence:** {round(prob * 100, 2)}%")
            st.progress(int(prob * 100))

    else:
        st.info("Please upload an image to get started.")

# Content for "Camera Input" tab
elif selected_tab == "Camera Input":
    st.header("Capture Image from Camera")
    st.markdown("Use your device's camera to capture an image.")

    # Use camera input
    image_file = st.camera_input("")

    if image_file:
        # Display the captured image and prediction results side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Captured Image")
            st.image(image_file, use_container_width =True)

        with col2:
            st.subheader("Prediction")
            cls, prob = predict_image(image_file, model)
            st.write(f"**Detected Emotion:** {cls}")
            st.write(f"**Confidence:** {round(prob * 100, 2)}%")
            st.progress(int(prob * 100))

    else:
        st.info("Please use the camera above to capture an image.")

# Content for "About" tab
elif selected_tab == "About":
    st.header("About This App")
    st.write("""
        This application uses a deep learning model to recognize emotions from facial images.
        The model predicts one of the following emotions:
        - Angry
        - Disgust
        - Fear
        - Happy
        - Sad
        - Surprise
        - Neutral
        """)
    st.write("""
        **How to use this app:**
        1. Go to the **Emotion Detection** tab.
            - Upload a facial image using the file uploader.
        2. Or go to the **Camera Input** tab.
            - Capture a facial image using your device's camera.
        3. Wait for the model to predict the emotion.
        4. View the predicted emotion and confidence level.
        """)

# Footer
st.markdown("<br><hr><center>© 2025 Rewan Abdulkariem❤️</center>", unsafe_allow_html=True)