import torch
import numpy as np

def compute_brats_dice(y_pred, y_true, smooth=1e-6):
    """Computes Dice scores for BraTS dataset segmentation regions."""
    y_pred = torch.argmax(y_pred, dim=1)  # Convert to binary mask
    
    # Whole Tumor (WT) - labels 1, 2, 4
    wt_pred = (y_pred > 0).float()
    wt_true = (y_true > 0).float()
    
    # Tumor Core (TC) - labels 1, 4
    tc_pred = ((y_pred == 2) | (y_pred == 3)).float()
    tc_true = ((y_true == 2) | (y_true == 3)).float()
    
    # Enhancing Tumor (ET) - label 4
    et_pred = (y_pred == 2).float()
    et_true = (y_true == 2).float()
    
    return dice(wt_pred, wt_true, smooth), dice(tc_pred, tc_true, smooth), dice(et_pred, et_true, smooth)

def cal_dice(output, target):
    """Calculates Dice scores for three segmentation regions."""
    dice1 = dice(output[:, 2, 40:80], target[:, 2, 40:80])
    dice2 = dice(output[:, 1, 40:80], target[:, 1, 40:80])
    dice3 = dice(output[:, 2, 40:80], target[:, 2, 40:80])
    return dice1, dice2, dice3

def dice(preds, targets, threshold=0.5, epsilon=1e-6):
    """Computes Dice coefficient for binary segmentation."""
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds_bin * targets).sum()
    return (2.0 * intersection + epsilon) / (preds_bin.sum() + targets.sum() + epsilon)

def iou_score(preds, targets, threshold=0.5, epsilon=1e-6):
    """Computes Intersection over Union (IoU) for binary segmentation."""
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)
