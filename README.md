Face Recognition System using LBPH (OpenCV)
📌 Project Overview

This project implements a real-time Face Recognition System using OpenCV’s LBPH (Local Binary Patterns Histogram) algorithm.
It detects and recognizes human faces from a live webcam feed and displays the identified person’s name along with confidence percentage.

🎯 Objectives

Detect human faces using Haar Cascade

Recognize known individuals using LBPH algorithm

Display name and confidence score

Handle unknown faces

Simple, fast, and Windows-compatible implementation

🛠️ Technologies Used

Python 3.10

OpenCV (opencv-contrib-python)

NumPy

Haar Cascade Classifier

LBPH Face Recognizer

📂 Project Structure
face/
│
├── dataset/
│   ├── Akhil/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   ├── Vijay/
│       ├── 1.jpg
│       ├── 2.jpg
│
├── train_faces.py
├── recognize_faces.py
├── face_model.yml
├── labels.pkl
└── README.md

🖼️ Dataset Guidelines

Each person must have a separate folder

Folder name = Person name

Use 5–10 clear front-face images

Avoid blurry or side-angle images

⚙️ Installation
Step 1: Install Dependencies
python -m pip install opencv-contrib-python numpy

▶️ How to Run the Project
Step 2: Train the Model (Run Once)
python train_faces.py


Expected output:

Assigning label 0 to Akhil
Assigning label 1 to Vijay
Training completed ✅


This creates:

face_model.yml

labels.pkl

Step 3: Run Face Recognition
python recognize_faces.py


🎥 Webcam opens and shows:

Face bounding box

Name of recognized person

Confidence percentage

Press ESC to exit.

📊 Output Example
Akhil (92%)
Vijay (88%)
Unknown (0%)

🧠 Algorithm Used – LBPH

LBPH (Local Binary Patterns Histogram) analyzes local texture patterns of the face and compares them with trained images.

Why LBPH?

Works well in low light

Fast and lightweight

No GPU required

Ideal for academic projects

⚠️ Limitations

Accuracy decreases with face angles

Not suitable for large-scale systems

Sensitive to lighting variations

🚀 Future Enhancements

Attendance system integration

GUI using Tkinter

Deep learning (CNN / FaceNet)

Mask detection

Database storage
