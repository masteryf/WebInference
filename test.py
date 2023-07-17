import time
import torch
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from PIL import Image
from torchvision import transforms
import socket
import urllib.request

DEVICE = "cpu"


class GoogleLeNet(nn.Module):

    def __init__(self):
        super(GoogleLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.gle = torchvision.models.GoogLeNet()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.gle(x)
        return x


def ImageTransform(Img):
    Img = cv2.resize(Img, (96, 96))
    Img = Image.fromarray(Img)
    Img = transform(Img)
    Img = Img.unsqueeze(1)
    return Img


model = GoogleLeNet().to(DEVICE)
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(
    "utils/haarcascade_frontalface_default.xml")
model.load_state_dict(torch.load("model/GoogleNet/GoogleLeNet.pth", map_location=DEVICE))
transform = transforms.Compose([
    transforms.Resize([96, 96]),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
model.eval()
emo = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprised', 6: 'normal'}


def Show():
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in face:
            inputs = gray[y:y + h, x:x + w]
            inputs = ImageTransform(inputs)
            label = model(inputs)
            label = F.log_softmax(label, dim=1)
            pred = label.max(1, keepdim=True)[1]
            pred = int(pred)
            emotion = emo[pred]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 255, 155), 3)
            cv2.putText(frame, 'emotion_%s' % emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
            cv2.imshow('img', frame)
            cv2.waitKey(50)


def GetEmotion(Img):
    emotion = ''
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in face:
        inputs = gray[y:y + h, x:x + w]
        inputs = ImageTransform(inputs)
        label = model(inputs)
        label = F.log_softmax(label, dim=1)
        pred = label.max(1, keepdim=True)[1]
        pred = int(pred)
        emotion = emo[pred]
    return emotion


def GetFaceEmotion(Img):
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    inputs = ImageTransform(Img)
    label = model(inputs)
    label = F.log_softmax(label, dim=1)
    pred = label.max(1, keepdim=True)[1]
    pred = int(pred)
    print(pred)
    emotion = emo[pred]
    return emotion




Show()