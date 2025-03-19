import torch
from torch.utils.data import DataLoader, Subset
from monai.data import CacheDataset, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityd, ToTensord
import os
import glob
from utils.data_preprocessing import BraTS21Dataset
from utils.data_preprocessing import load_npy_data

def get_dataloader(data_dir, batch_size=16, num_workers=4):
    """Loads MRI dataset from .npy files and returns DataLoader."""
    print(f"Loading dataset from: {data_dir}")

    try:
        # Load .npy files
        image_paths = [
            f"{data_dir}/BraTS2021_flair_images_300.npy",
            f"{data_dir}/BraTS2021_t1_images_300.npy",
            f"{data_dir}/BraTS2021_t2_images_300.npy",
            f"{data_dir}/BraTS2021_t1ce_images_300.npy"
        ]
        mask_path = f"{data_dir}/BraTS2021_mask_binary_300.npy"

        images_tensors, masks_tensor = load_npy_data(image_paths, mask_path)

        # Create dataset
        full_dataset = BraTS21Dataset(images_tensors, masks_tensor)

        print(f"Loaded dataset with {len(full_dataset)} samples.")
        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty! Check the .npy files.")

        return full_dataset  # Return full dataset to be split later

    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")

def split_dataset(full_dataset, train_ratio=0.7, val_ratio=0.15):
    """Splits the dataset into train, validation, and test subsets."""
    train_size = int(train_ratio * len(full_dataset))
    val_size = int(val_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset = Subset(full_dataset, list(range(0, train_size)))
    val_dataset = Subset(full_dataset, list(range(train_size, train_size + val_size)))
    test_dataset = Subset(full_dataset, list(range(train_size + val_size, len(full_dataset))))

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    full_dataset = BraTS21Dataset()
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
