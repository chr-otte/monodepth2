from torch import nn
from torchvision import models, transforms


class PoseNet18Encoder(nn.Module):
    def __init__(self):
        super(PoseNet18Encoder, self).__init__()
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Break down ResNet into blocks for skip connections
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        self.layer1 = resnet18.layer1  # 64 channels
        self.layer2 = resnet18.layer2  # 128 channels
        self.layer3 = resnet18.layer3  # 256 channels
        self.layer4 = resnet18.layer4  # 512 channels

    def forward(self, x):
        # Store intermediate features
        skip_connections = {}

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_connections['conv1'] = x  # 64 channels

        x = self.maxpool(x)
        x = self.layer1(x)
        skip_connections['layer1'] = x  # 64 channels

        x = self.layer2(x)
        skip_connections['layer2'] = x  # 128 channels

        x = self.layer3(x)
        skip_connections['layer3'] = x  # 256 channels

        x = self.layer4(x)

        return x, skip_connections

class PoseNet18Encoder_LEGACY(nn.Module):
    def __init__(self):
        super(PoseNet18Encoder_LEGACY, self).__init__()
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet18.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x
