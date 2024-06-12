import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
# Load the pre-trained model
model_path = "D:\SKAY\Projects\KEEP\VScode\Brain\model.h5"
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image):
    image = tf.image.resize(image, (150, 150))  # Resize the image
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Function to make predictions
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    # Assuming model output is binary
    if prediction[0][0] > 0.5:
        return "Tumor Detected"
    else:
        return "Healthy Brain"

# Create interface
gr.Interface(predict_image,'image' ,"label", title="Brain Tumor Detection").launch(share=True)
