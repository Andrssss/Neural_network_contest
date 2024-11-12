import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, resnet34, efficientnet_b0



#  DenseNet
#  Xception --> nagyon pontos és komplex
#  ResNet





# MobileNetV2 inicializálása és testreszabása az osztályok számával
class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes=21):
        super(MobileNetV2Custom, self).__init__()
        #self.model = mobilenet_v2(pretrained=True)  # Betölt egy előre betanított MobileNetV2 modellt
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)  # Kimeneti réteg módosítása

    def forward(self, x):
        return self.model(x)

class ResNet34Custom(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34Custom, self).__init__()
        # Betölt egy előre betanított ResNet34 modellt
        self.model = resnet34(pretrained=True)
        # A teljesen kapcsolt réteget (fc) módosítja, hogy az osztályok számával működjön
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB0Custom(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0Custom, self).__init__()
        # Betölt egy előre betanított EfficientNet-B0 modellt
        self.model = efficientnet_b0(pretrained=True)
        # A kimeneti réteget módosítja az osztályok számával
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


