from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)
CORS(app)

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')

# Load the face detector
face_classifier = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_classifier.empty():
    raise FileNotFoundError(f"Could not load face cascade from {FACE_CASCADE_PATH}")

# Load the trained emotion detection model
try:
    classifier = load_model(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(f"Error loading model.h5: {e}")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def home():
    return jsonify({"message": "Emotion Detection API is running!"}), 200

@app.route('/detect', methods=['POST'])
def detect_emotion():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame provided"}), 400

    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Convert image to grayscale & detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    emotion = "Neutral"  # Default emotion
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        try:
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        except Exception as e:
            continue

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            emotion = emotion_labels[prediction.argmax()]
            break  # Process only one face

    return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
