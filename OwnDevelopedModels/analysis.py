import numpy as np
import matplotlib.pyplot as plt
import cv2


def analyze_depth_file(npy_filename):
    # Load the raw depth values
    depth_array = np.load(npy_filename)

    # Print basic statistics
    print(f"Depth Statistics:")
    print(f"Min depth: {depth_array.min():.4f}")
    print(f"Max depth: {depth_array.max():.4f}")
    print(f"Mean depth: {depth_array.mean():.4f}")
    print(f"Std dev: {depth_array.std():.4f}")

    # Create multiple visualizations
    plt.figure(figsize=(15, 10))

    # 1. Histogram of depth values
    plt.subplot(221)
    plt.hist(depth_array.flatten(), bins=50)
    plt.title('Depth Distribution')
    plt.xlabel('Depth Value')
    plt.ylabel('Count')

    # 2. Original depth map
    plt.subplot(222)
    plt.imshow(depth_array, cmap='viridis')
    plt.colorbar()
    plt.title('Original Depth Map')

    # 3. Log-scaled depth map
    plt.subplot(223)
    log_depth = np.log(depth_array + 1e-8)
    plt.imshow(log_depth, cmap='viridis')
    plt.colorbar()
    plt.title('Log-Scaled Depth Map')

    # 4. Histogram-equalized depth map
    plt.subplot(224)
    depth_normalized = ((depth_array - depth_array.min()) /
                        (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
    hist_eq = cv2.equalizeHist(depth_normalized)
    plt.imshow(hist_eq, cmap='viridis')
    plt.colorbar()
    plt.title('Histogram-Equalized Depth Map')

    plt.tight_layout()
    plt.savefig(npy_filename.replace('.npy', '_analysis.png'))
    plt.close()


# Example usage:
analyze_depth_file('depth_raw_epoch_0_batch_700.npy')