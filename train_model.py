import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from utils import get_embedding
import pickle

dataset_path = "dataset"
X, y = [], []

total_files = len([f for f in os.listdir(dataset_path) if f.endswith(".jpg")])
processed_files = 0

for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        try:
            image_path = os.path.join(dataset_path, file)
            face = cv2.imread(image_path)
            if face is None:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            embedding = get_embedding(face)
            name = file.split("_")[0]

            X.append(embedding)
            y.append(name)

            processed_files += 1
            if processed_files % 10 == 0:
                print(f"Processed {processed_files}/{total_files} images")

        except Exception as e:
            print(f"Error processing {file}: {e}")

if len(X) == 0:
    exit()

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y = le.fit_transform(y)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X, y)

with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("Training complete!")
