import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from KittiDataset import KITTIRawDataset
from DepthNet import DepthNet
from PoseNet import PoseNet
import cv2
import numpy as np


# POTENTIAL EXPERIMENTS;
# currently using image_02 (left color camera) - taking the other camera and black/white?
# currently we have one target image, and +/- 1 consecutive image.
#   What if it was increased?
#   What if it was from different cameras - not monolithic anymore, but is it required?
# currently resizing images to 192,640 from 375,1242 (HxW) to lower processing time, this leaves out details
#   What if we did not resize the image to test if we loose anything significant based on this?
# currently are we adding the loss for disparity and depth
#   What if we changed we balance this, e.g. (0,7/0,3), (0,5/0,5), (0,3/0,7)


def compute_mean_depth_loss(depth, target_mean=10.0):
    return 0.1 * torch.abs(depth.mean() - target_mean) / target_mean



def get_default_intrinsics():
    """
    Get KITTI camera intrinsics matrix scaled to our image size.
    Original KITTI image size: 1242 x 375
    Our size: 640 x 192
    """
    # Original KITTI intrinsics
    fx_original = 721.5377
    fy_original = 721.5377
    cx_original = 609.5593
    cy_original = 172.854

    # Scale factors
    width_scale = 640 / 1242
    height_scale = 192 / 375

    # Scale intrinsics
    fx = fx_original * width_scale  # Scale focal length by width ratio
    fy = fy_original * height_scale  # Scale focal length by height ratio
    cx = cx_original * width_scale  # Scale center x by width ratio
    cy = cy_original * height_scale  # Scale center y by height ratio

    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=torch.float32)

    return K

def backproject(depth, inv_K):
    batch_size, _, height, width = depth.shape
    device = depth.device

    # Create mesh grid of pixel coordinates
    i_range = torch.arange(0, height, device=device).reshape(1, height, 1).repeat(1, 1, width)
    j_range = torch.arange(0, width, device=device).reshape(1, 1, width).repeat(1, height, 1)
    ones = torch.ones(1, height, width, device=device)
    pix_coords = torch.stack((j_range, i_range, ones), dim=1).float()  # Shape: [1, 3, H, W]

    pix_coords = pix_coords.expand(batch_size, -1, -1, -1)  # Shape: [B, 3, H, W]

    # Flatten pixel coordinates
    pix_coords_flat = pix_coords.view(batch_size, 3, -1)  # Shape: [B, 3, H*W]

    # Apply inverse intrinsic matrix
    cam_points = torch.bmm(inv_K, pix_coords_flat)  # Shape: [B, 3, H*W]

    # Multiply by depth
    depth_flat = depth.view(batch_size, 1, -1)
    cam_points = cam_points * depth_flat  # Shape: [B, 3, H*W]

    # Reshape to [B, 3, H, W]
    cam_points = cam_points.view(batch_size, 3, height, width)
    return cam_points

def project(cam_points, K, T):
    batch_size, _, height, width = cam_points.shape
    device = cam_points.device

    # Flatten camera points
    cam_points_flat = cam_points.view(batch_size, 3, -1)  # [B, 3, H*W]
    ones = torch.ones(batch_size, 1, cam_points_flat.shape[2], device=device)
    cam_points_hom = torch.cat([cam_points_flat, ones], dim=1)  # [B, 4, H*W]

    # Apply transformation matrix
    P = torch.bmm(K, T[:, :3, :])  # [B, 3, 4]
    proj_points = torch.bmm(P, cam_points_hom)  # [B, 3, H*W]

    # Normalize by third coordinate
    pix_coords = proj_points[:, :2, :] / (proj_points[:, 2:3, :] + 1e-7)  # [B, 2, H*W]

    # Reshape to [B, 2, H, W]
    pix_coords = pix_coords.view(batch_size, 2, height, width)
    return pix_coords


def transformation_from_parameters(axisangle, translation, invert=False):
    R = rot_from_axisangle(axisangle)  # [B, 3, 3]
    t = translation.view(-1, 3, 1)     # [B, 3, 1]

    # Combine rotation and translation
    T = torch.cat([R, t], dim=2)  # [B, 3, 4]

    # Add bottom row [0, 0, 0, 1]
    bottom_row = torch.tensor([0, 0, 0, 1], device=T.device).unsqueeze(0).repeat(T.shape[0], 1).unsqueeze(1)  # [B, 1, 4]
    T = torch.cat([T, bottom_row], dim=1)  # [B, 4, 4]

    if invert:
        T = torch.inverse(T)
    return T

def rot_from_axisangle(axisangle):
    angle = torch.norm(axisangle, dim=2, keepdim=True)  # [B, 1, 1]
    axis = axisangle / (angle + 1e-7)  # [B, 1, 3]

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]  # [B, 1]

    zero = torch.zeros_like(x)
    K = torch.stack([
        zero, -z, y,
        z, zero, -x,
        -y, x, zero
    ], dim=-1).view(-1, 3, 3)  # [B, 3, 3]

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    I = torch.eye(3, device=axisangle.device).unsqueeze(0)
    R = I + sin * K + (1 - cos) * torch.bmm(K, K)
    return R


def reconstruct_image(src_image, depth, T, K, inv_K):
    batch_size, _, height, width = depth.shape

    # Backproject pixels to 3D points in camera coordinates
    cam_points = backproject(depth, inv_K)  # [B, 3, H, W]

    # Flatten cam_points to [B, 3, H*W]
    cam_points_flat = cam_points.view(batch_size, 3, -1)  # [B, 3, H*W]

    # Apply transformation
    ones = torch.ones(batch_size, 1, cam_points_flat.shape[2], device=cam_points.device)
    cam_points_hom = torch.cat([cam_points_flat, ones], dim=1)  # [B, 4, H*W]
    cam_points_src = torch.bmm(T, cam_points_hom)  # [B, 4, H*W]
    cam_points_src = cam_points_src[:, :3, :]  # [B, 3, H*W]

    # Project back to 2D pixel coordinates
    pix_coords = project(cam_points_src.view(batch_size, 3, height, width), K, torch.eye(4, device=depth.device)[:3, :4].unsqueeze(0).expand(batch_size, -1, -1))

    # Normalize coordinates to [-1, 1]
    pix_coords[:, 0, :, :] = (pix_coords[:, 0, :, :] / (width - 1) - 0.5) * 2
    pix_coords[:, 1, :, :] = (pix_coords[:, 1, :, :] / (height - 1) - 0.5) * 2
    grid = pix_coords.permute(0, 2, 3, 1)  # [B, H, W, 2]

    # Sample source image
    reconstructed = F.grid_sample(src_image, grid, padding_mode='border', align_corners=True)
    return reconstructed


def ssim(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = ssim_n / ssim_d
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


def compute_photometric_loss(target_image, reconstructed_image):
    """Modified photometric loss with auto-masking"""
    # Standard photometric loss
    l1_loss = torch.abs(target_image - reconstructed_image)
    ssim_loss = ssim(target_image, reconstructed_image)

    # Combine losses
    photometric_loss = 0.85 * ssim_loss.mean(1, True) + 0.15 * l1_loss.mean(1, True)

    # Auto-masking: ignore static pixels
    mean_target = target_image.mean(1, True).mean(2, True).mean(3, True)
    mask = (torch.abs(target_image.mean(1, True) - mean_target) > 0.1).float()

    return (photometric_loss * mask).mean()


def analyze_depth_values(depth_tensor):
    """Analyzes depth tensor to identify potential issues"""
    # Convert to numpy for analysis
    depth_np = depth_tensor.detach().cpu().numpy()

    stats = {
        'min': float(depth_np.min()),
        'max': float(depth_np.max()),
        'mean': float(depth_np.mean()),
        'std': float(depth_np.std()),
        'percentiles': np.percentile(depth_np, [0, 25, 50, 75, 100])
    }

    return stats


def save_visualization_improved(curr_image, depth, epoch, batch_idx):
    """Improved visualization function with better depth map rendering"""
    import numpy as np
    import cv2

    # Convert input image to numpy and denormalize
    input_image = curr_image.detach().cpu().numpy()
    input_image = np.transpose(input_image, (1, 2, 0))

    # Denormalize input image
    mean = np.array([0.3706, 0.3861, 0.3680])  # KITTI means
    std = np.array([0.3059, 0.3125, 0.3136])  # KITTI stds
    input_image = input_image * std + mean
    input_image = np.clip(input_image * 255, 0, 255).astype(np.uint8)

    # Process depth map
    depth_np = depth.squeeze().detach().cpu().numpy()

    # 1. Raw depth map (just normalized to 0-255 for visualization)
    depth_raw_vis = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
    depth_raw_colored = cv2.applyColorMap(depth_raw_vis, cv2.COLORMAP_TURBO)

    # 2. Log normalization
    depth_log = np.log(depth_np + 1e-8)
    depth_log_norm = (depth_log - depth_log.min()) / (depth_log.max() - depth_log.min() + 1e-8)
    depth_log_colored = cv2.applyColorMap((depth_log_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    # 3. Histogram equalization
    depth_norm = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
    depth_eq = cv2.equalizeHist(depth_norm)
    depth_eq_colored = cv2.applyColorMap(depth_eq, cv2.COLORMAP_TURBO)

    # Create side-by-side visualization
    h, w = input_image.shape[:2]
    combined_img = np.zeros((h, w * 4, 3), dtype=np.uint8)  # Space for 4 images

    # Convert input_image to BGR for consistent visualization
    input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Place images side by side
    combined_img[:, :w] = input_image_bgr
    combined_img[:, w:2 * w] = depth_raw_colored
    combined_img[:, 2 * w:3 * w] = depth_log_colored
    combined_img[:, 3 * w:] = depth_eq_colored

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_img, 'Input Image', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined_img, 'Raw Depth', (w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined_img, 'Depth (Log)', (2 * w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined_img, 'Depth (Eq)', (3 * w + 10, 30), font, 1, (255, 255, 255), 2)

    # Save the visualization
    cv2.imwrite(f'depth_vis_epoch_{epoch}_batch_{batch_idx}.png', combined_img)
    # Save depth values in case we need them for further analysis
    np.save(f'depth_raw_epoch_{epoch}_batch_{batch_idx}.npy', depth_np)

# Modified compute_mean_depth_loss with more detailed constraints
def compute_mean_depth_loss_improved(depth, target_mean=10.0):
    """Improved depth loss with better regularization"""
    batch_mean = depth.mean()
    mean_loss = 0.1 * torch.abs(batch_mean - target_mean) / target_mean

    # Add variance loss to encourage depth variation
    variance = torch.var(depth)
    variance_loss = 0.1 * torch.exp(-variance)

    # Add gradient loss to encourage sharp depth transitions
    dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :]).mean()
    dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1]).mean()
    gradient_loss = 0.1 * torch.exp(-(dx + dy))

    return mean_loss + variance_loss + gradient_loss
def compute_disparity_smoothness_loss(disp, img):
    # calculates smoothness for disparity

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    # no normalization of gradients as it is already fixed, compared to depth

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


# Add a stronger depth regularization loss
def compute_depth_regularization_loss(depth):
    """Additional loss to encourage greater depth variation"""
    # Encourage larger depth range
    depth_range = depth.max() - depth.min()
    range_loss = torch.exp(-depth_range / 10.0)

    # Encourage depth variation
    depth_std = torch.std(depth)
    std_loss = torch.exp(-depth_std)

    return range_loss + std_loss
def compute_smoothness_loss(depth, image):
    # calculates smoothness for depth

    # Get gradients
    # Calculates how much depth changes between adjacent pixels
    # Does this for both horizontal (x) and vertical (y) directions
    # Large gradients indicate sudden depth changes
    depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    # Normalize depth gradients by local depth values, to avoid the relativity depth changes in the real world to punish more than it should
    depth_dx = depth_dx / (depth[:, :, :, :-1] + 1e-6)
    depth_dy = depth_dy / (depth[:, :, :-1, :] + 1e-6)


    # Get image gradients
    # finding edges, high means indicate edge, low means indicate smooth region
    image_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
    image_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)


    # Converts gradients to weights using exponential function
    #The negative exponential means:
    #   High gradients (edges) → Low weights (close to 0)
    #   Low gradients (smooth areas) → High weights (close to 1)
    #
    #The 150.0 factor controls how sharply weights fall off at edges
    #   Higher value = sharper transition at edges
    #   Lower value = more gradual transition
    weights_x = torch.exp(-150.0 * image_dx)  # Increased from 100.0
    weights_y = torch.exp(-150.0 * image_dy)

    # Apply weights
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y

    return smoothness_x.mean() + smoothness_y.mean()




def save_visualization(curr_image, depth, epoch, batch_idx):
    import numpy as np
    from PIL import Image
    import matplotlib as mpl

    # Convert input image to numpy and denormalize
    input_image = curr_image.detach().cpu().numpy()
    input_image = np.transpose(input_image, (1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_image = input_image * std + mean
    input_image = np.clip(input_image * 255, 0, 255).astype(np.uint8)

    # Process depth map with better range visualization
    depth_np = depth.squeeze().detach().cpu().numpy()

    # Use log scale with adjusted range
    depth_np = np.clip(depth_np, 0.1, 100.0)
    depth_vis = np.log(depth_np)

    # Normalize with focus on typical KITTI ranges
    depth_min = np.log(0.1)
    depth_max = np.log(80.0)
    depth_vis = (depth_vis - depth_min) / (depth_max - depth_min)

    # Create custom colormap (purple-green-yellow for better depth perception)
    colors = ['purple', 'blue', 'cyan', 'green', 'yellow']
    n_bins = 256
    cmap = mpl.colors.LinearSegmentedColormap.from_list("depth", colors, N=n_bins)

    # Apply colormap
    depth_colored = cmap(depth_vis)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # Create side-by-side visualization
    input_pil = Image.fromarray(input_image)
    depth_pil = Image.fromarray(depth_colored)

    # Combine images
    total_width = input_pil.width * 2
    max_height = max(input_pil.height, depth_pil.height)
    combined_img = Image.new('RGB', (total_width, max_height))

    combined_img.paste(input_pil, (0, 0))
    combined_img.paste(depth_pil, (input_pil.width, 0))

    combined_img.save(f'combined_epoch_{epoch + 1}_batch_{batch_idx + 1}.png')

def main():


    # Sequences for training and validation
    train_sequences = ['2011_09_26', '2011_09_28', '2011_09_29']
    val_sequences = ['2011_09_30', '2011_10_03']

    # Root directory
    root_dir = 'C:/Github/monodepth2/kitti_data'

    # Create datasets
    train_dataset = KITTIRawDataset(root_dir=root_dir, sequences=train_sequences)
    val_dataset = KITTIRawDataset(root_dir=root_dir, sequences=val_sequences)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_net = DepthNet().to(device)
    pose_net = PoseNet(num_input_images=2).to(device)

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                if m.out_channels == 1:  # Final layer
                    # Initialize to predict middle-range depths
                    nn.init.constant_(m.bias, 0.0)  # Start at neutral point
                else:
                    nn.init.zeros_(m.bias)

    depth_net.decoder.apply(initialize_weights)

    # Define optimizer and scheduler
    params = list(depth_net.parameters()) + list(pose_net.parameters())
    optimizer = torch.optim.Adam([
        {'params': depth_net.encoder.parameters(), 'lr': 5e-5},  # Slower learning rate
        {'params': depth_net.decoder.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-7)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=500,
        verbose=True
    )

    # Constants
    num_epochs = 10
    lambda_smooth = 0.000005    # Further reduce smoothness for better detail
    K = get_default_intrinsics().to(device)
    inv_K = torch.inverse(K).to(device)

    for epoch in range(num_epochs):



        depth_net.train()
        pose_net.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            curr_image = batch['curr_image'].to(device)
            src_images = batch['src_images'].to(device)

            if src_images.nelement() == 0:
                continue

            batch_size, num_src, _, _, _ = src_images.shape

            # Initialize total loss for this batch
            total_loss = torch.tensor(0.0, device=device)

            # Get depth and disparity predictions
            disp, depth = depth_net(curr_image)

            # Compute smoothness losses
            # Having both losses helps the model learn better depth representations:
            #   Depth smoothness handles far-range smoothness well
            #   Disparity smoothness handles near-range smoothness well

            depth_smoothness = compute_smoothness_loss(depth, curr_image)
            disp_smoothness = compute_disparity_smoothness_loss(disp, curr_image)
            smoothness_loss = depth_smoothness + disp_smoothness
            reg_loss = compute_depth_regularization_loss(depth)
            total_loss += 0.1 * reg_loss  # Adjust weight as needed

            # Process each source image (frame t-1 and t+1)
            for i in range(num_src):
                src_image = src_images[:, i]

                is_forward = i == 1 # determines the order of the input for the pose network

                # Prepare input for pose network, predicts the movement between the images
                if is_forward:
                    pose_input = torch.cat([curr_image, src_image], dim=1)
                else:
                    pose_input = torch.cat([src_image, curr_image], dim=1)

                pose = pose_net(pose_input)[:, 0]

                # Get transformation
                axisangle = pose[:, :3].unsqueeze(1)    # rotational x,y,z
                translation = pose[:, 3:].unsqueeze(1)  # x,y,z
                T = transformation_from_parameters(axisangle, translation, invert=is_forward)

                # Reconstruct image
                reconstructed_image = reconstruct_image(
                    src_image,
                    depth,
                    T,
                    K.expand(batch_size, -1, -1),
                    inv_K.expand(batch_size, -1, -1)
                )

                # Compute photometric loss
                photometric_loss = compute_photometric_loss(curr_image, reconstructed_image)
                total_loss += photometric_loss

            # Add smoothness loss
            total_loss += lambda_smooth * smoothness_loss
            total_loss += compute_mean_depth_loss(depth)  # Add mean depth constraint
            # Logging
            if batch_idx % 100 == 0:
                # Save visualization
                save_visualization_improved(
                    curr_image[0],
                    depth[0, 0],
                    epoch,
                    batch_idx
                )

                print(f"\nBatch {batch_idx}")
                print(
                    f"Disparity min={disp.min().item():.4f}, max={disp.max().item():.4f}, mean={disp.mean().item():.4f}")
                print(
                    f"Depth min={depth.min().item():.4f}, max={depth.max().item():.4f}, mean={depth.mean().item():.4f}")
                print(f"Photometric Loss: {photometric_loss.item():.4f}")
                print(f"Smoothness Loss: {smoothness_loss.item():.4f}")
                print(f"Total Loss: {total_loss.item():.4f}")
                print("-" * 20)

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

        # Step the scheduler with the average epoch loss
        scheduler.step(avg_epoch_loss)  # Note: ReduceLROnPlateau needs a loss value

        # Optional: Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'depth_net_state_dict': depth_net.state_dict(),
                'pose_net_state_dict': pose_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, f'checkpoint_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    main()
