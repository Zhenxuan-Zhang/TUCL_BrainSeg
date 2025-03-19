import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def load_npy_data(image_paths, mask_path, device="cpu"):
    """Loads multi-contrast MRI images and masks from .npy files with a progress bar."""
    
    try:
        print("Loading MRI images...")
        images = []
        for path in tqdm(image_paths, desc="Loading Images"):  
            img = np.load(path)
            # print(f"Loaded {path}, shape: {img.shape}") 
            images.append(img)
        
        print("Loading Masks...")
        masks = np.load(mask_path)  
        # print(f"Loaded {mask_path}, shape: {masks.shape}")
        
    except Exception as e:
        raise RuntimeError(f"Error loading .npy files: {e}")

    print("Converting to PyTorch tensors...")
    images_tensors = [torch.tensor(img, dtype=torch.float32).to(device) for img in tqdm(images, desc="Converting Images")]
    masks_tensor = torch.tensor(masks, dtype=torch.float32).to(device)

    # Process masks
    if masks_tensor.shape[1] >= 3:
        masks_tensor[:, 1] = masks_tensor[:, 0] + masks_tensor[:, 1] + masks_tensor[:, 2]
        masks_tensor[:, 2] = masks_tensor[:, 0] + masks_tensor[:, 2]

    print("Final Image Tensor Shape:", images_tensors[0].shape)
    print("Final Mask Tensor Shape:", masks_tensor.shape)

    return images_tensors, masks_tensor

class BraTS21Dataset(Dataset):
    def __init__(self, image_tensors, mask_tensor):
        """Creates a dataset with stacked multi-contrast images."""
        self.images = torch.stack(image_tensors, dim=1)  # [B, 4, H, W, D]
        self.masks = mask_tensor
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.masks[index]

