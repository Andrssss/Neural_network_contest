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




class AlexNet(nn.Module):
    def __init__(self, num_classes=21):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 64 szűrő, 11x11 méretű, stride=4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # Az adaptív pooling kimeneti méretet ad meg
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)  # ez az utolsó layer
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
