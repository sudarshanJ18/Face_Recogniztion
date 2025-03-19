# Face Recognition API

## Overview
This project is a **Face Recognition API** built with **Flask** and **OpenCV**. It detects and recognizes faces in images using deep learning models. The API is deployed on **Render** for easy accessibility.

## Features
- Face detection and recognition
- RESTful API for image processing
- Supports multiple image formats (JPG, PNG, etc.)
- Easy deployment on **Render**

## Tech Stack
- **Backend**: Flask (Python)
- **Machine Learning**: OpenCV, dlib, face-recognition
- **Deployment**: Render (Cloud Hosting)

---

## Setup and Deployment on Render

### 1. Clone the Repository
```sh
git clone https://github.com/sudarshanJ18/Face_Recogniztion.git
cd Face_Recogniztion.git
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run Locally (Optional)
To test locally before deployment:
```sh
python detect_faces.py  # Face detection
python collect_faces.py  # Collect and store faces
python train_model.py  # Train the recognition model
python recognize.py  # Recognize faces from images
```
Access API locally at: `http://127.0.0.1:5000`

---

## Deploy on Render

### 1. Create a New Web Service on Render
- Go to **[Render.com](https://render.com)**
- Click **New Web Service**
- Connect your GitHub repository

### 2. Set Environment Variables
Navigate to **Environment Settings** in Render and add:
```sh
PYTHON_VERSION=3.9
PORT=10000  # Render assigns its own port dynamically
```

### 3. Build and Deploy
- Set **Start Command**: `gunicorn app:app`
- Select the correct Python runtime (Python 3.9+ recommended)
- Click **Deploy**

### 4. Get Public URL
Once deployed, Render provides a **public URL** where your API will be accessible.

---

## API Endpoints

### 1. Face Recognition (POST)
#### Endpoint:
```sh
POST /detect-face
```
#### Request (Form-Data):
- `image` (File): Upload an image containing faces

#### Response (JSON):
```json
{
  "faces_detected": 2,
  "bounding_boxes": [[x1, y1, x2, y2], [...]]
}
```

---

## Additional Notes
- Make sure your **Render Web Service** is **publicly accessible**.
- If using **GPU acceleration**, ensure Render supports it (optional).

## License
This project is licensed under the **MIT License**.

## Author
[Sudarshan](https://github.com/sudarshanJ18)

