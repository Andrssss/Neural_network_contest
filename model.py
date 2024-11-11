import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# MobileNetV2 inicializálása és testreszabása az osztályok számával
class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2Custom, self).__init__()
        #self.model = mobilenet_v2(pretrained=True)  # Betölt egy előre betanított MobileNetV2 modellt
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)  # Kimeneti réteg módosítása

    def forward(self, x):
        return self.model(x)
