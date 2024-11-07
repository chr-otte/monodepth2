import torch
from torch import nn
from torch.nn import functional as F
from Encoder import PoseNet18Encoder
from Decoder import Decoder


class DepthNet(nn.Module):
    def __init__(self, min_depth=0.05, max_depth=150.0):
        super(DepthNet, self).__init__()
        self.encoder = PoseNet18Encoder()
        self.decoder = Decoder()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_mean_depth = 10.0

        # Add skip connections
        self.skip1 = nn.Conv2d(256, 256, 1)
        self.skip2 = nn.Conv2d(128, 128, 1)
        self.skip3 = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        features, skip_connections = self.encoder(x)

        # Pass both to decoder
        disp_raw = self.decoder(features, skip_connections)
        # Better sigmoid scaling
        disp = 0.3 * F.sigmoid(disp_raw) + 0.01

        # Convert to depth (avoid extreme values)
        depth = 1.0 / (disp + 1e-6)

        # Soft depth constraints (avoid hard clipping)
        depth = self.min_depth + (self.max_depth - self.min_depth) * (
                torch.tanh(depth / self.max_depth) + 1.0) / 2.0
        return disp,depth

        # Improved disparity activation with gradient-friendly sigmoid
        disp = 0.3 * F.sigmoid(disp_raw) + 0.01

        # Convert to depth with better numerical stability
        depth = torch.where(
            disp > 1e-3,
            1.0 / disp,
            torch.ones_like(disp) * self.max_depth
        )

        # Scale depth with gradient-friendly normalization
        mean_depth = torch.mean(depth, dim=[1, 2, 3], keepdim=True)
        scale = self.target_mean_depth / (mean_depth + 1e-8)
        depth = depth * scale
        depth = torch.exp(torch.log(depth) * 1.2)  # Add exponential scaling

        # Improved soft constraints with better gradient flow
        normalized_depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        depth = self.min_depth + (self.max_depth - self.min_depth) * torch.sigmoid(
            4.0 * (normalized_depth - 0.5)  # Steeper sigmoid for better range utilization
        )

        # Recompute disparity with the same numerical stability as above
        disp_final = torch.where(
            depth > 1e-3,
            1.0 / depth,
            torch.ones_like(depth) * (1.0 / self.max_depth)
        )

        return disp_final, depth

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
