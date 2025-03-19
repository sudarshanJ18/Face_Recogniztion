from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
from utils import get_embedding
from sklearn.preprocessing import LabelEncoder
import face_recognition

app = Flask(__name__)

# Load pre-trained SVM model
svm_model = pickle.load(open("models/svm_model.pkl", "rb"))

@app.route('/')
def home():
    return "Face Recognition API is Running!"

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Recognize faces from an uploaded image."""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    face_locations = face_recognition.face_locations(image)
    recognized_faces = []

    for (top, right, bottom, left) in face_locations:
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (160, 160))

        embedding = get_embedding(face)
        prediction = svm_model.predict([embedding])[0]
        name = "Unknown" if prediction is None else str(prediction)

        recognized_faces.append({"name": name, "bounding_box": [top, right, bottom, left]})

    return jsonify({
        "faces_detected": len(face_locations),
        "recognized_faces": recognized_faces
    })


@app.route('/webcam', methods=['GET'])
def webcam_recognition():
    """Start real-time webcam-based face recognition."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))

            embedding = get_embedding(face)
            prediction = svm_model.predict([embedding])[0]

            name = "Unknown" if prediction is None else str(prediction)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Webcam recognition session closed."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
