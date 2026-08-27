import os
import cv2
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Face & Emotion Recognition", layout="wide")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
COLOR_MAP = {
    "angry": (255, 50, 50),
    "disgust": (140, 70, 20),
    "fear": (138, 43, 226),
    "happy": (0, 220, 0),
    "neutral": (0, 200, 255),
    "sad": (30, 144, 255),
    "surprise": (255, 165, 0)
}

@st.cache_resource(show_spinner=False)
def load_models():
    face_model = YOLO("best.pt")
    emotion_model = load_model(
        "best_emotion_model_scratch_enhanced.keras",
        custom_objects={"preprocess_input": preprocess_input}
    )
    return face_model, emotion_model

face_model, emotion_model = load_models()

def process_frame(rgb_frame, conf_thresh):
    h, w, _ = rgb_frame.shape
    results = face_model.predict(rgb_frame, conf=conf_thresh, imgsz=640, verbose=False)

    for person_id, box in enumerate(results[0].boxes.xyxy, start=1):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = rgb_frame[y1:y2, x1:x2]
        if face.size == 0 or face.shape[0] < 12 or face.shape[1] < 12:
            continue

        face_input = cv2.resize(face, (48, 48)).astype("float32") / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        preds = emotion_model(face_input, training=False).numpy()[0]
        emotion_idx = int(np.argmax(preds))
        confidence = float(preds[emotion_idx])
        emotion_name = EMOTIONS[emotion_idx]

        display_label = f"Person {person_id}: {emotion_name.upper()} ({confidence * 100:.1f}%)"
        box_color = COLOR_MAP.get(emotion_name, (0, 255, 0))

        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(rgb_frame, (x1, max(0, y1 - 32)), (x1 + len(display_label) * 12, max(30, y1)), box_color, -1)
        cv2.putText(rgb_frame, display_label, (x1 + 6, max(22, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return rgb_frame

st.title("🎭 AI Face & Emotion Video Detector")

st.sidebar.title("Settings")
conf_thresh = st.sidebar.slider("Face Detection Confidence", 0.1, 1.0, 0.40, 0.05)
mode = st.sidebar.radio("Select Mode", ["Upload Video", "Live Local Webcam"])

if mode == "Upload Video":
    video_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
    
    if video_file is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        # UI Elements for video playback
        start_processing = st.button("▶️ Process Video")
        stop_processing = st.button("⏹️ Stop")
        frame_window = st.empty()  # Placeholder for the video stream
        
        if start_processing:
            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened() and not stop_processing:
                ret, frame = cap.read()
                if not ret:
                    st.success("Video processing complete!")
                    break
                
                # Convert BGR to RGB for processing and displaying
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run the AI models on the frame
                processed_frame = process_frame(rgb, conf_thresh)
                
                # Render the processed frame to the Streamlit UI
                frame_window.image(processed_frame)
                
            cap.release()

elif mode == "Live Local Webcam":
    start_cam = st.toggle("Start Camera", value=False)
    frame_window = st.empty()
    if start_cam:
        cap = cv2.VideoCapture(0)
        while start_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access camera.")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = process_frame(rgb, conf_thresh)
            frame_window.image(processed)
        cap.release()
