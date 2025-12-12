import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Load the Model (We do this once when the app starts)
model = load_model('Age_Gender_Model.h5')

# 2. Define the Logic Function
def predict_chronovis(image):
    # Gradio passes us an image as a numpy array (RGB)
    if image is None:
        return "Please upload an image or use the webcam."

    # --- Preprocessing (Must match training exactly) ---
    # Resize to 128x128
    processed_img = cv2.resize(image, (128, 128))
    
    # Normalize (0 to 1)
    processed_img = processed_img / 255.0
    
    # Expand dimensions (The model expects a batch of images)
    # Shape becomes (1, 128, 128, 3)
    processed_img = np.expand_dims(processed_img, axis=0)

    # --- Prediction ---
    predictions = model.predict(processed_img)
    
    # Extract results
    pred_gender = predictions[0][0][0] # 0 to 1
    pred_age = predictions[1][0][0]    # Number
    
    # Interpret Gender
    if pred_gender < 0.5:
        gender_result = "Male"
        confidence = (1 - pred_gender) * 100
    else:
        gender_result = "Female"
        confidence = pred_gender * 100
        
    # Interpret Age (Round it)
    age_result = int(round(pred_age))

    return f"Gender: {gender_result} ({confidence:.1f}%)\nEstimated Age: {age_result} years old"

# 3. Create the UI (User Interface)
interface = gr.Interface(
    fn=predict_chronovis,                 # The function above
    inputs=gr.Image(sources=["webcam", "upload"]), # Input method
    outputs="text",                       # Output method
    title="ChronoVis",
    description="Real-Time Age & Gender Recognition powered by Deep Learning (CNN). Upload a photo or use your webcam!"
)

# 4. Launch
interface.launch()
