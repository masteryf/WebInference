import cv2
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model.GoogleNet.GoogleNet import model, labels

DEVICE = "cpu"


def ImageTransform(Img):
    transform = transforms.Compose([
        transforms.Resize([96, 96]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    Img = cv2.resize(Img, (96, 96))
    Img = Image.fromarray(Img)
    Img = transform(Img)
    Img = Img.unsqueeze(1)
    return Img


def GetLabel(Img):
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    inputs = ImageTransform(Img)
    label = model(inputs)
    label = F.log_softmax(label, dim=1)
    pred = label.max(1, keepdim=True)[1]
    pred = int(pred)
    out = labels[pred]
    return out
