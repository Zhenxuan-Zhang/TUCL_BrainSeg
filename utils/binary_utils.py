import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_slices_as_png(volume, save_path, prefix="slice"):
    """Save 3D volume as PNG slices."""
    os.makedirs(save_path, exist_ok=True)
    volume = volume.cpu().numpy()

    for i in range(volume.shape[0]):  # Iterate over all slices
        slice_img = volume[i]  # Extract slice
        
        # Normalize to [0, 255]
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        slice_img = (slice_img * 255).astype(np.uint8)
        
        plt.imsave(os.path.join(save_path, f"{prefix}_{i:03d}.png"), slice_img, cmap='gray')

def binary(pred1, pred2, threshold=0.5):
    """Computes a binary difference mask between two predictions."""
    probs1 = torch.sigmoid(pred1)
    preds_bin1 = (probs1 > threshold).float()
    
    probs2 = torch.sigmoid(pred2)
    preds_bin2 = (probs2 > threshold).float()
    
    return preds_bin1 - preds_bin2
