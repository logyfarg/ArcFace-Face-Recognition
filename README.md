ArcFace-Face-Recognition
Description

This project implements a state-of-the-art face recognition system using ArcFace, a high-performance deep learning model for learning discriminative face embeddings. The system is designed for robust identity recognition, verification, and inference, suitable for applications such as authentication, monitoring, and access control.

Features

ResNet50 backbone for strong feature extraction.

ArcFace loss function to enforce angular margin and improve discriminative power.

Supports folder-structured datasets such as MS1MV2, CASIA-WebFace, or small subsets for testing.

Data augmentation: Horizontal flipping, normalization, resizing.

GPU-accelerated training using PyTorch.

Inference pipeline: Generate embeddings for new faces and perform recognition using cosine similarity.

Modular and professional: Easily extendable for live webcam recognition or large-scale face databases.

Dataset Structure
```bash
datasets/face_recognition/
├── person_1/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── person_2/
│   └── ...
└── person_3/
    └── ...

```

Each folder corresponds to a person and contains face images of that individual. Images should be cropped and aligned (112x112 recommended).

Recommended Datasets:

MS1M-ArcFace / MS1MV2: https://github.com/deepinsight/insightface/wiki/Dataset-Zoo

CASIA-WebFace: http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html

Requirements

Python 3.8+

PyTorch 2.0+

torchvision

Pillow

OpenCV

CUDA-enabled GPU (optional but recommended for training)

Install dependencies:
```bash
pip install -r requirements.txt
```
Usage
1. Train the model

Update data_dir in src/ArcFace_FaceRecognition.py to point to your dataset folder, then run:
```bash
python src/ArcFace_FaceRecognition.py
```

The model will be saved as arcface_model.pth.

2. Inference
```python
from src.ArcFace_FaceRecognition import ArcFaceModel, recognize_face

embedding = recognize_face(model, label_map, "./datasets/face_recognition/test_face.jpg")
print(embedding)
```
3. Evaluation

Use cosine similarity between embeddings for verification or identification.

Extendable to threshold-based recognition or large-scale face matching.

Next Steps / Extensions

Live webcam recognition and real-time identification.

Save embeddings of all known faces and implement a matching database.

Fine-tune the model on custom datasets for specialized recognition tasks.

Author

Logina Mahmoud (Logy) — Aspiring Advanced AI Engineer | Portfolio-ready project demonstrating state-of-the-art face recognition using ArcFace.
