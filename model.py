import torch
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import efficientnet_b0
from timm import create_model


import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Custom, self).__init__()

        # MobileNetV2 inicializálása előre betanított súlyokkal
        self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Utolsó osztályozó réteg lecserélése az osztályok számára
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class AdvancedMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(AdvancedMobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # Configurations for each stage
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # First stage
            [6, 24, 2, 2],  # Second stage
            [6, 32, 3, 2],  # Third stage
            [6, 64, 4, 2],  # Fourth stage
            [6, 96, 3, 1],  # Fifth stage
            [6, 160, 3, 2],  # Sixth stage
            [6, 320, 1, 1],  # Seventh stage
        ]

        # Building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last layer
        features.append(nn.Conv2d(input_channel, self.last_channel, kernel_size=1, bias=False))
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))

        # Combine features
        self.features = nn.Sequential(*features)

        # Classifier layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.classifier(x)
        return x
