import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

try:
    embedding_model = hub.load('https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1')
except:
    try:
        embedding_model = hub.load('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4')
    except:
        embedding_model = None

def preprocess_face(face_pixels):
    face = face_pixels.astype('float32')
    face = cv2.resize(face, (224, 224))
    face = face / 255.0
    return np.expand_dims(face, axis=0)

def get_embedding(face_pixels):
    try:
        face = preprocess_face(face_pixels)
        if embedding_model is None:
            return np.zeros(1280)
        embedding = embedding_model(face).numpy().flatten()
        return embedding / np.linalg.norm(embedding)
    except:
        return np.zeros(1280)
