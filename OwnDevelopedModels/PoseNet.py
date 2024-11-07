from torch import nn
from torchvision import models, transforms
class PoseNet(nn.Module):
    def __init__(self, num_input_images=2):
        super(PoseNet, self).__init__()

        self.num_input_images = num_input_images
        resnet18 = models.resnet18(pretrained=True)

        # 3 input images as we are using frame t-1, t and t+1
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Copy pretrained weights as we are using resnet18 nets.
        self.conv1.weight.data = resnet18.conv1.weight.data.repeat(1, num_input_images, 1, 1) / num_input_images

        # Feature extraction
        self.features = nn.Sequential(
            self.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
        )

        # Converts to single vector
        self.pool = nn.AdaptiveAvgPool2d(1)

        # final feature dimension, 6 -> 6 degrees of freedom -> x,y,z + rotational x,y,z
        self.fc = nn.Linear(512, 6 * (num_input_images - 1))

    def forward(self, x):
        # x shape: [batch_size, 6, height, width]
        x = self.features(x)  # Extract features
        x = self.pool(x)  # Global pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Predict poses
        # Reshape to [batch_size, num_source_frames, 6]
        x = x.view(-1, self.num_input_images - 1, 6)
        return x