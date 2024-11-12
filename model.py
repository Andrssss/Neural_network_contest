import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, resnet34, efficientnet_b0
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from timm import create_model




# MobileNetV2 inicializálása és testreszabása az osztályok számával
class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes=21):
        super(MobileNetV2Custom, self).__init__()
        #self.model = mobilenet_v2(pretrained=True)  # Betölt egy előre betanított MobileNetV2 modellt
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)  # Kimeneti réteg módosítása

    def forward(self, x):
        return self.model(x)

# ResNet34 testreszabása regresszióhoz
class ResNet34Custom(nn.Module):
    def __init__(self, num_classes=21):
        super(ResNet34Custom, self).__init__()
        self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB0Custom(nn.Module):
    def __init__(self, num_classes=21):
        super(EfficientNetB0Custom, self).__init__()
        # Betölt egy előre betanított EfficientNet-B0 modellt
        self.model = efficientnet_b0(pretrained=True)
        # A kimeneti réteget módosítja az osztályok számával
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Swin Transformer testreszabása regresszióhoz
class SwinTransformerCustom(nn.Module):
    def __init__(self, num_classes=21):
        super(SwinTransformerCustom, self).__init__()
        self.model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# ConvNeXt testreszabása regresszióhoz
class ConvNeXtCustom(nn.Module):
    def __init__(self, num_classes=21):
        super(ConvNeXtCustom, self).__init__()
        self.model = create_model('convnext_base', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)