import cv2
import os
import numpy as np
import pickle

data_path = "dataset"
faces = []
labels = []
label_map = {}
label_id = 0

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for person in sorted(os.listdir(data_path)):
    person_path = os.path.join(data_path, person)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person
    print(f"Assigning label {label_id} to {person}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces_rect = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces_rect:
            faces.append(img[y:y+h, x:x+w])
            labels.append(label_id)

    label_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")

with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Training completed ✅")
print("Label map:", label_map)

