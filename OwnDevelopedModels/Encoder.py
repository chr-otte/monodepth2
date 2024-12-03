from torch import nn
from torchvision import models, transforms


class PoseNet18Encoder(nn.Module):
    def __init__(self):
         # Initialize parent class (nn.Module)
        super(PoseNet18Encoder, self).__init__()
        
         # Load pre-trained ResNet-18 model
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Separate the ResNet-18 layers for access to intermediate outputs (for skip connections)
        self.conv1 = resnet18.conv1       # Initial convolutional layer (64 channels)
        self.bn1 = resnet18.bn1           # Initial batch normalization layer
        self.relu = resnet18.relu         # ReLU activation
        self.maxpool = resnet18.maxpool   # Max-pooling layer after conv1

        self.layer1 = resnet18.layer1  # 64 channels
        self.layer2 = resnet18.layer2  # 128 channels
        self.layer3 = resnet18.layer3  # 256 channels
        self.layer4 = resnet18.layer4  # 512 channels

    def forward(self, x):
        # Dictionary to store intermediate features for skip connection
        skip_connections = {}

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_connections['conv1'] = x  # Save features after initial convolution (64 channels)

        x = self.maxpool(x)
        x = self.layer1(x)
        skip_connections['layer1'] = x  # Save features after layer1 (64 channels)

        x = self.layer2(x)
        skip_connections['layer2'] = x  # Save features after layer2 (128 channels)

        x = self.layer3(x)
        skip_connections['layer3'] = x  # Save features after layer3 (256 channels)
        
        # Final layer without skip connection (output features are 512 channels)
        x = self.layer4(x)
        
        # Return final output features and skip connections
        return x, skip_connections


class PoseNet50Encoder(nn.Module):
    def __init__(self):
        # Initialize parent class (nn.Module)
        super(PoseNet50Encoder, self).__init__()
        
        # Load pre-trained ResNet-50 model
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Separate the ResNet-50 layers for access to intermediate outputs (for skip connections)
        self.conv1 = resnet50.conv1       # Initial convolutional layer (64 channels)
        self.bn1 = resnet50.bn1           # Initial batch normalization layer
        self.relu = resnet50.relu         # ReLU activation
        self.maxpool = resnet50.maxpool   # Max-pooling layer after conv1

        self.layer1 = resnet50.layer1  # 256 channels
        self.layer2 = resnet50.layer2  # 512 channels
        self.layer3 = resnet50.layer3  # 1024 channels
        self.layer4 = resnet50.layer4  # 2048 channels

    def forward(self, x):
        # Dictionary to store intermediate features for skip connection
        skip_connections = {}

        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_connections['conv1'] = x  # Save features after initial convolution (64 channels)

        x = self.maxpool(x)
        x = self.layer1(x)
        skip_connections['layer1'] = x  # Save features after layer1 (256 channels)

        x = self.layer2(x)
        skip_connections['layer2'] = x  # Save features after layer2 (512 channels)

        x = self.layer3(x)
        skip_connections['layer3'] = x  # Save features after layer3 (1024 channels)
        
        # Final layer without skip connection (output features are 2048 channels)
        x = self.layer4(x)
        
        # Return final output features and skip connections
        return x, skip_connections
