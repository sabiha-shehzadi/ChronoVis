
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("Age_Gender_Model.h5", compile=False)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_chronovis(image):
    if image is None:
        return "Please upload an image or use the webcam."

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        margin = int(w * 0.1)
        face_img = image[
            max(0, y - margin): y + h + margin,
            max(0, x - margin): x + w + margin
        ]
        status = "✅ Face Detected"
    else:
        face_img = image
        status = "⚠️ No face detected (using full image)"

    # Resize to model input size (128x128)
    try:
        face_img = cv2.resize(face_img, (128, 128))
    except:
        return "Error processing image."

    # Normalize
    face_img = face_img.astype("float32")
    face_img = (face_img / 127.5) - 1.0
    face_img = np.expand_dims(face_img, axis=0)

    # Predict
    preds = model.predict(face_img)
    gender_score = preds[0][0][0]
    age = int(round(preds[1][0][0]))

    if gender_score < 0.5:
        gender = "Male"
        confidence = (1 - gender_score) * 100
    else:
        gender = "Female"
        confidence = gender_score * 100

    return (
        f"{status}\n"
        f"Gender: {gender} ({confidence:.1f}%)\n"
        f"Estimated Age: {age} years"
    )

# Gradio Interface
interface = gr.Interface(
    fn=predict_chronovis,
    inputs=gr.Image(sources=["webcam", "upload"], type="numpy"),
    outputs="text",
    title="ChronoVis",
    description="Real-Time Age & Gender Prediction using Deep Learning"
)

interface.launch()
