from torch import nn
from torchvision import models, transforms
class PoseNet(nn.Module):
    def __init__(self, num_input_images=2):
        # Initialize the parent class (nn.Module) to enable its features
        super(PoseNet, self).__init__()

        # Set the number of input images (e.g., 2 images means frame pairs for pose estimation)
        self.num_input_images = num_input_images
        
        # Load a pre-trained ResNet-18 model from torchvision
        resnet18 = models.resnet18(pretrained=True)

        # Replace the first convolutional layer to accept multiple input images (e.g., 3 channels * num_input_images)
        # 3 input images as we are using frame t-1, t and t+1
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        
         # Copy the pretrained weights from ResNet-18’s first convolutional layer and adjust for multiple images
        self.conv1.weight.data = resnet18.conv1.weight.data.repeat(1, num_input_images, 1, 1) / num_input_images

        # Feature extraction from resnet
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

        # Fully connected layer to predict poses (6-DOF for each frame pair)
        # 6 * (num_input_images - 1) represents the 6 pose parameters (x, y, z, rotation x, y, z) per frame pair
        self.fc = nn.Linear(512, 6 * (num_input_images - 1))

    def forward(self, x):
        # x shape: [batch_size, 6, height, width]
        # Pass input through feature extraction layers
        # It captures both low and high level features.
        x = self.features(x)  # Extract features from the input 
        
    
        x = self.pool(x)  # Global pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Predict poses
        # Reshape to [batch_size, num_source_frames, 6]
        x = x.view(-1, self.num_input_images - 1, 6)
        return x
