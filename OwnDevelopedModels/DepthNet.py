import torch
from torch import nn
from torch.nn import functional as F
from Encoder import PoseNet18Encoder
from Encoder import PoseNet50Encoder
from Decoder import Decoder18
from Decoder import Decoder50




class DepthNet50(nn.Module):
    def __init__(self, min_depth=0.05, max_depth=150.0):
        super(DepthNet50, self).__init__()
        # Initialize encoder (ResNet18-based) and decoder
        self.encoder = PoseNet50Encoder()
        self.decoder = Decoder50()
        
        # Set minimum and maximum depth constraints
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Target mean depth value for normalization during training
        self.target_mean_depth = 10.0


    def forward(self, x):
        # Step 1: Pass input through the encoder to extract features and intermediate skip connections
        features, skip_connections = self.encoder(x)
        

        # Step 2: Decode features into a raw disparity map, utilizing skip connections for finer details
        disp_raw = self.decoder(features, skip_connections)
        
        # Step 3: Scale disparity with sigmoid to produce values between 0.01 and 0.31
        disp = 0.3 * F.sigmoid(disp_raw) + 0.01

        # Step 4: Convert disparity to depth by taking the inverse clamp to instue no divition with 0.
        depth = 1.0 / torch.clamp(disp, min=1e-3)

        # Step 5: clamp depth. 
        depth = torch.clamp(depth, min=self.min_depth, max=self.max_depth)

        return disp,depth




class DepthNet18(nn.Module):
    def __init__(self, min_depth=0.05, max_depth=150.0):
        super(DepthNet18, self).__init__()
        # Initialize encoder (ResNet18-based) and decoder
        self.encoder = PoseNet18Encoder()
        self.decoder = Decoder18()
        
        # Set minimum and maximum depth constraints
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Target mean depth value for normalization during training
        self.target_mean_depth = 10.0

         # Additional layers for complexity
        self.extra_layers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # skip connection
        self.skip1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.skip2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.skip3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    


    def forward(self, x):
        # Step 1: Pass input through the encoder to extract features and intermediate skip connections
        features, skip_connections = self.encoder(x)
        
        # Pass through additional convolutional layers
        #features = self.extra_layers(features)

        # Step 2: Decode features into a raw disparity map, utilizing skip connections for finer details
        disp_raw = self.decoder(features, skip_connections)
        
        # Step 3: Scale disparity with sigmoid to produce values between 0.01 and 0.31
        disp = 0.3 * F.sigmoid(disp_raw) + 0.01

        # Step 4: Convert disparity to depth by taking the inverse clamp to instue no divition with 0.
        depth = 1.0 / torch.clamp(disp, min=1e-3)

        # Step 5: clamp depth. 
        depth = torch.clamp(depth, min=self.min_depth, max=self.max_depth)

        return disp,depth


class DepthNetbase(nn.Module):
    def __init__(self, min_depth=0.05, max_depth=150.0):
        super(DepthNetbase, self).__init__()
        # Initialize encoder (ResNet18-based) and decoder
        self.encoder = PoseNet18Encoder()
        self.decoder = Decoder18()
        
        # Set minimum and maximum depth constraints
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Target mean depth value for normalization during training
        self.target_mean_depth = 10.0

         # Additional layers for complexity
        self.extra_layers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
    


    def forward(self, x):
        # Step 1: Pass input through the encoder to extract features and intermediate skip connections
        features, skip_connections = self.encoder(x)
        
        # Pass through additional convolutional layers
        #features = self.extra_layers(features)

        # Step 2: Decode features into a raw disparity map, utilizing skip connections for finer details
        disp_raw = self.decoder(features, skip_connections)
        
        # Step 3: Scale disparity with sigmoid to produce values between 0.01 and 0.31
        disp = 0.3 * F.sigmoid(disp_raw) + 0.01

        # Step 4: Convert disparity to depth by taking the inverse clamp to instue no divition with 0.
        depth = depth = 1.0 / (disp + 1e-6)

        # Step 5: tanh depth. 
        depth = self.min_depth + (self.max_depth - self.min_depth) * (torch.tanh(depth / self.max_depth) + 1.0) / 2.0
       
        return disp,depth
