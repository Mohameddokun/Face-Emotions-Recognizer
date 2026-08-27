import cv2
import numpy as np
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Live Face & Emotion Recognition", layout="wide")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
COLOR_MAP = {
    "angry": (50, 50, 255),       # BGR formats for OpenCV drawing
    "disgust": (20, 70, 140),
    "fear": (226, 43, 138),
    "happy": (0, 220, 0),
    "neutral": (255, 200, 0),
    "sad": (255, 144, 30),
    "surprise": (0, 165, 255)
}

# 1. Load Models (Cached so they only load once)
@st.cache_resource(show_spinner=False)
def load_models():
    face_model = YOLO("best.pt")
    emotion_model = load_model(
        "best_emotion_model_scratch_enhanced.keras",
        custom_objects={"preprocess_input": preprocess_input}
    )
    return face_model, emotion_model

face_model, emotion_model = load_models()

st.title("🎭 Live AI Face & Emotion Detector")
st.markdown("Ensure your browser allows camera access. The video processes in real-time on the cloud.")

# 2. Define the Frame Processing Callback
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert WebRTC frame to OpenCV BGR format
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape

    # Face Detection (Hardcoded conf=0.40 for thread safety)
    results = face_model.predict(img, conf=0.40, imgsz=640, verbose=False)

    for person_id, box in enumerate(results[0].boxes.xyxy, start=1):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = img[y1:y2, x1:x2]
        if face.size == 0 or face.shape[0] < 12 or face.shape[1] < 12:
            continue

        # Convert face crop from BGR to RGB for the Keras model
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_input = cv2.resize(face_rgb, (48, 48)).astype("float32") / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        # Emotion Prediction
        preds = emotion_model(face_input, training=False).numpy()[0]
        emotion_idx = int(np.argmax(preds))
        confidence = float(preds[emotion_idx])
        emotion_name = EMOTIONS[emotion_idx]

        display_label = f"Person {person_id}: {emotion_name.upper()} ({confidence * 100:.1f}%)"
        box_color = COLOR_MAP.get(emotion_name, (0, 255, 0))

        # Draw Bounding Box and Label directly onto the image
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(img, (x1, max(0, y1 - 32)), (x1 + len(display_label) * 12, max(30, y1)), box_color, -1)
        cv2.putText(img, display_label, (x1 + 6, max(22, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Return the processed frame to the browser
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Setup STUN server to guarantee peer-to-peer connection over strict firewalls
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 4. Mount the WebRTC Streamer UI Component
webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
