import cv2
import os

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
person_name = "Sudarshan"

while count < 50:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        cv2.imwrite(f"{dataset_path}/{person_name}_{count}.jpg", face)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
