from torch import nn
from torch.nn import functional as F
import torch




class Decoder50(nn.Module):
    def __init__(self, num_output_channels=1):
        super(Decoder50, self).__init__()

        # Decoder blocks with skip connections
        # Each block includes: upsampling, concatenation, and convolution

          # Adjust the first upconv layer to match the encoder's output channels
        self.upconv4 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)  # 2048 = 1024 (upsampled) + 1024 (skip)

        self.upconv3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)  # 1024 = 512 (upsampled) + 512 (skip)

        self.upconv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 512 = 256 (upsampled) + 256 (skip)

        self.upconv1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128 = 64 (upsampled) + 64 (skip)

        self.final_conv = nn.Conv2d(64, num_output_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(0.1)  # Add dropout for regularization
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip_connections):
        # Decoder with skip connections
        # Up-conv 4
        x = self.up(x)
        x = F.relu(self.bn4(self.upconv4(x)))
        x = torch.cat([x, skip_connections['layer3']], dim=1)  # Concatenate with skip connection
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)  # Dropout for regularization

        # Up-conv 3
        x = self.up(x)
        x = F.relu(self.bn3(self.upconv3(x)))
        x = torch.cat([x, skip_connections['layer2']], dim=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        # Up-conv 2
        x = self.up(x)
        x = F.relu(self.bn2(self.upconv2(x)))
        x = torch.cat([x, skip_connections['layer1']], dim=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        # Up-conv 1
        x = self.up(x)
        x = F.relu(self.bn1(self.upconv1(x)))
        x = torch.cat([x, skip_connections['conv1']], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))

        # Final convolution to get the desired output channels
        x = self.final_conv(x)

        return x




class Decoder18(nn.Module):
    def __init__(self, num_output_channels=1):
        super(Decoder18, self).__init__()

        # Decoder blocks with skip connections
        # Each block includes: upsampling, concatenation, and convolution

        # Starting from 512 channels from encoder
        self.upconv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 512 = 256 (upsampled) + 256 (skip)

        self.upconv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 256 = 128 (upsampled) + 128 (skip)

        self.upconv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128 = 64 (upsampled) + 64 (skip)

        self.upconv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(96, 32, kernel_size=3, padding=1)  # 96 = 32 (upsampled) + 64 (skip)

        self.upconv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, num_output_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(0.1)  # Add dropout for regularization
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip_connections):
        # Decoder with skip connections
        # Up-conv 5
        x = self.up(x)
        x = F.relu(self.bn5(self.upconv5(x)))
        x = torch.cat([x, skip_connections['layer3']], dim=1) # Concatenate with skip connection
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x) # Dropout for regularization

        # Up-conv 4
        x = self.up(x)
        x = F.relu(self.bn4(self.upconv4(x)))
        x = torch.cat([x, skip_connections['layer2']], dim=1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        # Up-conv 3
        x = self.up(x)
        x = F.relu(self.bn3(self.upconv3(x)))
        x = torch.cat([x, skip_connections['layer1']], dim=1)
        x = F.relu(self.bn3(self.conv3(x)))

        # Up-conv 2
        x = self.up(x)
        x = F.relu(self.bn2(self.upconv2(x)))
        x = torch.cat([x, skip_connections['conv1']], dim=1)
        x = F.relu(self.bn2(self.conv2(x)))

        # Up-conv 1
        x = self.up(x)
        x = F.relu(self.bn1(self.upconv1(x)))
        x = self.conv1(x)

        return x




class Decoder_LEGACY(nn.Module):
    def __init__(self, num_output_channels=1):
        super(Decoder_LEGACY, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

        # Final layer without batch norm
        self.conv6 = nn.Conv2d(16, num_output_channels, kernel_size=3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.up(F.relu(self.bn1(self.conv1(x))))
        x = self.up(F.relu(self.bn2(self.conv2(x))))
        x = self.up(F.relu(self.bn3(self.conv3(x))))
        x = self.up(F.relu(self.bn4(self.conv4(x))))
        x = self.up(F.relu(self.bn5(self.conv5(x))))
        x = self.conv6(x)
        return x
