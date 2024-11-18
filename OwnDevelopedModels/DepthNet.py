import torch
from torch import nn
from torch.nn import functional as F
from Encoder import PoseNet18Encoder
from Decoder import Decoder


class DepthNet(nn.Module):
    def __init__(self, min_depth=0.05, max_depth=150.0):
        super(DepthNet, self).__init__()
        # Initialize encoder (ResNet18-based) and decoder
        self.encoder = PoseNet18Encoder()
        self.decoder = Decoder()
        
        # Set minimum and maximum depth constraints
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Target mean depth value for normalization during training
        self.target_mean_depth = 10.0

        
        self.skip1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.skip2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.skip3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    


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

        

class DepthNet_LEGACY(nn.Module):
    def __init__(self, min_depth=0.1, max_depth=100.0):
        super(DepthNet_LEGACY, self).__init__()
        self.encoder = PoseNet18Encoder()
        self.decoder = Decoder()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_mean_depth = 10.0

    def forward(self, x):
        # 1. Feature Extraction
        # Input image passes through encoder
        # Produces high-level feature representation
        # Shape: [B, 512, H/32, W/32]
        features = self.encoder(x)

        # 2. Raw Disparity Prediction
        # Decoder processes features
        # Outputs raw disparity values
        #   (disparity; difference in position of an object when viewed from different positions)
        #   Mathematical Properties of disparity;
        #       Disparity has a linear relationship with inverse depth
        #       Depth = baseline * focal_length / disparity
        # No activation function in final layer
        # Shape: [B, 1, H, W]
        disp_raw = self.decoder(features)

        # 3. Disparity Processing
        # Apply softplus to ensure positive disparity values
        # 0.05 scaling factor controls the disparity range
        # 0.001 offset prevents division by zero in depth conversion
        disp = 0.05 * F.softplus(disp_raw) + 0.001

        # 4. Depth Conversion
        # Convert disparity to depth using inverse relationship
        # Add small epsilon (1e-8) for numerical stability
        # Higher disparity = closer objects (smaller depth)
        # Lower disparity = farther objects (larger depth)
        depth_raw = 1.0 / (disp + 1e-8)

        # 5. Depth Scaling
        # Calculate mean depth of the scene
        # Scale depth to maintain target mean of 10.0 meters
        # This helps prevent scale drift during training
        # Add small epsilon (1e-8) to prevent division by zero
        mean_depth = depth_raw.mean()
        scale = self.target_mean_depth / (mean_depth + 1e-8)
        depth = depth_raw * scale

        # 6. Calculate scaled disparity
        # Convert scaled depth back to disparity
        # This disparity corresponds to the scaled depth values
        # Used for loss calculations and evaluation
        # Add small epsilon (1e-8) for numerical stability
        disp_scaled = 1.0 / (depth + 1e-8)

        # 7. Final Depth Constraints
        # Clamp depth values to physically plausible range
        # min_depth: typically 0.1 meters (very close objects)
        # max_depth: typically 100.0 meters (far objects)
        # Prevents unrealistic predictions
        depth_final = torch.clamp(depth, self.min_depth, self.max_depth)

        # Return both disparity and depth representations
        # disp_scaled: used for certain loss calculations
        # depth_final: final depth prediction with physical constraints
        return disp_scaled, depth_final
