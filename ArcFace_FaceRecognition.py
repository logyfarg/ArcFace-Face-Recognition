# ArcFace_FaceRecognition.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import os
from PIL import Image
import numpy as np
import cv2
import math

# -------------------------------
# 1. Dataset Class
# -------------------------------
class FaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        label_id = 0
        for person_name in os.listdir(self.image_dir):
            person_path = os.path.join(self.image_dir, person_name)
            if not os.path.isdir(person_path):
                continue
            self.label_map[label_id] = person_name
            for img_name in os.listdir(person_path):
                self.image_paths.append(os.path.join(person_path, img_name))
                self.labels.append(label_id)
            label_id += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------
# 2. ArcFace Loss
# -------------------------------
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        # Normalize
        embeddings = nn.functional.normalize(embeddings)
        W = nn.functional.normalize(self.weight)
        cosine = torch.matmul(embeddings, W.t())
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return self.ce(output, labels)

# -------------------------------
# 3. Model Backbone (ResNet)
# -------------------------------
class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(ArcFaceModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove classification layer
        self.backbone = nn.Sequential(*modules)
        self.embedding = nn.Linear(resnet.fc.in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

# -------------------------------
# 4. Training Pipeline
# -------------------------------
def train_model(data_dir, num_classes, batch_size=32, epochs=10, lr=1e-4, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    dataset = FaceDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = ArcFaceModel(num_classes).to(device)
    criterion = ArcFaceLoss(in_features=512, out_features=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(imgs)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "arcface_model.pth")
    print("Training complete. Model saved as arcface_model.pth")
    return model, dataset.label_map

# -------------------------------
# 5. Inference / Recognition
# -------------------------------
def recognize_face(model, label_map, img_path, device='cuda', threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    model.eval()
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_tensor)
        emb = nn.functional.normalize(emb)
    
    # Compare with known embeddings (in a real scenario save & load embeddings)
    # Here we just return embedding for simplicity
    return emb.cpu().numpy()

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    data_dir = "./VGGFace2_subset"  # Update path to your dataset
    num_classes = len(os.listdir(data_dir))
    model, label_map = train_model(data_dir, num_classes, epochs=5)  # quick test

    # Inference
    test_img = "./test_face.jpg"
    embedding = recognize_face(model, label_map, test_img)
    print("Face embedding:", embedding)
