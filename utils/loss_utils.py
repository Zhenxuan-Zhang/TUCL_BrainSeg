import torch
import torch.nn as nn

def softmax_loss(y_pred, y_true):
    """Computes softmax cross-entropy loss."""
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(y_pred, y_true)

def seg_loss(y_pred, y_true, smooth=1e-6):
    """Computes a combined segmentation loss (Dice + Focal)."""
    y_pred = torch.sigmoid(y_pred)

    # Dice Loss
    dice_numerator = 2 * torch.sum(y_true * y_pred) + smooth
    dice_denominator = torch.sum(y_true) + torch.sum(y_pred) + smooth
    dice_loss = 1 - dice_numerator / dice_denominator

    # Focal Loss
    gamma = 2.0
    focal_loss = -torch.mean((1 - y_pred) ** gamma * y_true * torch.log(y_pred + smooth) + 
                              y_pred ** gamma * (1 - y_true) * torch.log(1 - y_pred + smooth))
    
    return dice_loss + focal_loss
