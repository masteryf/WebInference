import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

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



model = GoogleLeNet().to(DEVICE)
model.load_state_dict(torch.load("GoogleLeNet.pth", map_location=DEVICE))

model.eval()


labels = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprised', 6: 'normal'}
