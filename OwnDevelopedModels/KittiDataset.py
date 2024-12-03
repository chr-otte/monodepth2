import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def compute_kitti_mean_std(dataset, batch_size=16, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # First pass: compute mean
    mean = torch.zeros(3)
    total_samples = 0

    for data in loader:
        curr_images = data['curr_image']  # Shape: [B, C, H, W]
        batch_samples = curr_images.size(0)
        curr_images = curr_images.view(batch_samples, curr_images.size(1), -1)
        mean += curr_images.mean(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples

    # Second pass: compute std
    var = torch.zeros(3)

    for data in loader:
        curr_images = data['curr_image']  # Shape: [B, C, H, W]
        batch_samples = curr_images.size(0)
        curr_images = curr_images.view(batch_samples, curr_images.size(1), -1)
        var += ((curr_images - mean.view(1, -1, 1)) ** 2).sum([0, 2])

    std = torch.sqrt(var / (total_samples * curr_images.size(2)))  # Divide by N*H*W

    return mean, std
class KITTIRawDataset(Dataset):
    def __init__(self, root_dir, sequences, camera='image_02'):

        self.root_dir = root_dir
        self.sequences = sequences  # sequences of directories for data
        self.transform = transforms.Compose([
            transforms.Resize((192, 640)),  # Resizes image to 192, 640 (H,W) to lower processing time.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3706, 0.3861, 0.3680],  # KITTI calculated mean
            std=[0.3059, 0.3125, 0.3136])                        # KITTI calculated std
        ])
        self.samples = []

        # Loop over sequences to collect file paths
        for seq in sequences:
            #seq_dir = os.path.join(root_dir, seq)
            seq_dir = root_dir
            # List all drives within the date
            drives = [d for d in os.listdir(seq_dir) if 'sync' in d]
            for drive in drives:
                drive_dir = os.path.join(seq_dir, drive)
                image_dir = os.path.join(drive_dir, camera, 'data')

                if not os.path.exists(image_dir):
                    continue  # Skip if images are not available

                # Get sorted list of image files
                image_files = sorted([
                    os.path.join(image_dir, f) for f in os.listdir(image_dir)
                    if f.endswith('.PNG') or f.endswith('.png') # just ensuring it is images only
                ])

                # Collect samples as dictionaries containing current and source images

                # Two images to simulate multiple cameras, needs to be consecutive images.
                # avoiding the use of first and last image, as no consecutive images exists.
                for i in range(1, len(image_files) - 1):
                    frame = {}
                    frame['curr_img'] = image_files[i]
                    frame['src_imgs'] = [
                        image_files[i - 1],
                        image_files[i + 1]
                    ]
                    self.samples.append(frame)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # getting sample
        sample = self.samples[idx]

        # Load current target image
        # sample['curr_img'] is a datapath
        # 'RGB' Should be changed id we experiment with black/white images
        curr_image = Image.open(sample['curr_img']).convert('RGB')

        # Loading source images
        src_images = [Image.open(img_path).convert('RGB') for img_path in sample['src_imgs']]


        # Apply transformations defined during initializations.
        if self.transform:
            curr_image = self.transform(curr_image)
            src_images = [self.transform(img) for img in src_images]

        # Stack source images along a new dimension
        # Shape: [2, C, H, W] - 2 for +/- 1 image of the src images, C for the number of color channels (3 - RGB), H for the height of the image, W for the width
        src_images = torch.stack(src_images)

        # Return current image and source images
        return {
            'curr_image': curr_image,
            'src_images': src_images
        }


class KITTIRawDatasetWithAugmentation(Dataset):
    def __init__(self, root_dir, sequences, camera='image_02'):
        self.root_dir = root_dir
        self.sequences = sequences
        self.transform = transforms.Compose([
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3706, 0.3861, 0.3680],
                                 std=[0.3059, 0.3125, 0.3136])
        ])
        self.samples = []

        # Data augmentation transforms
        self.augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
            transforms.RandomHorizontalFlip()  # Random horizontal flip
        ])

        for seq in sequences:
            seq_dir = root_dir
            drives = [d for d in os.listdir(seq_dir) if 'sync' in d]
            for drive in drives:
                drive_dir = os.path.join(seq_dir, drive)
                image_dir = os.path.join(drive_dir, camera, 'data')

                if not os.path.exists(image_dir):
                    continue

                image_files = sorted([
                    os.path.join(image_dir, f) for f in os.listdir(image_dir)
                    if f.endswith('.PNG') or f.endswith('.png')
                ])

                for i in range(1, len(image_files) - 1):
                    frame = {}
                    frame['curr_img'] = image_files[i]
                    frame['src_imgs'] = [
                        image_files[i - 1],
                        image_files[i + 1]
                    ]
                    self.samples.append(frame)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        curr_image = Image.open(sample['curr_img']).convert('RGB')
        src_images = [Image.open(img_path).convert('RGB') for img_path in sample['src_imgs']]

        # Apply data augmentation
        if random.random() < 0.5:  # 50% chance to apply augmentation
            curr_image = self.augmentation(curr_image)
            src_images = [self.augmentation(img) for img in src_images]

        if self.transform:
            curr_image = self.transform(curr_image)
            src_images = [self.transform(img) for img in src_images]

        src_images = torch.stack(src_images)

        return {
            'curr_image': curr_image,
            'src_images': src_images
        }
