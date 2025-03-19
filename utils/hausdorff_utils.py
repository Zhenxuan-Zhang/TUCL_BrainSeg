import torch
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from monai.metrics import HausdorffDistanceMetric

def cal_hd95(output, target, device='cuda'):
    """Calculates the 95th percentile Hausdorff Distance for segmentation outputs."""
    output1 = output[:, :, 40:80].clone().contiguous()
    target1 = target[:, :, 40:80].clone().contiguous()

    output1 = torch.sigmoid(output1) > 0.5
    target1 = target1 > 0.5

    output1 = output1.to(device)
    target1 = target1.to(device)

    hd95_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
    
    try:
        hd_score = hd95_metric(output1, target1)
        hd1, hd2, hd3 = hd_score[0, 0], hd_score[0, 1], hd_score[0, 2]
    except Exception as e:
        print(f"Hausdorff Distance computation failed: {e}")
        return None

    hd95_metric.reset()
    return hd1, hd2, hd3

def compute_hd95_single(pred, gt, voxelspacing=None, connectivity=1):
    """Computes the 95th percentile Hausdorff Distance for a single 3D volume."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    if not np.any(pred) and not np.any(gt):
        return 0.0
    
    if not np.any(pred):
        dt = distance_transform_edt(~gt, sampling=voxelspacing)
        return np.percentile(dt[gt == 0], 95)
    
    if not np.any(gt):
        dt = distance_transform_edt(~pred, sampling=voxelspacing)
        return np.percentile(dt[pred == 0], 95)
    
    footprint = generate_binary_structure(pred.ndim, connectivity)
    pred_border = pred ^ binary_erosion(pred, structure=footprint)
    gt_border = gt ^ binary_erosion(gt, structure=footprint)
    
    dt_pred = distance_transform_edt(~pred, sampling=voxelspacing)
    dt_gt = distance_transform_edt(~gt, sampling=voxelspacing)
    
    distances_pred_to_gt = dt_gt[pred_border]
    distances_gt_to_pred = dt_pred[gt_border]
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    return np.percentile(all_distances, 95)

def hd95_score(preds, targets, threshold=0.5, voxelspacing=None, connectivity=1):
    """Computes the average 95th percentile Hausdorff Distance over a batch."""
    preds = torch.sigmoid(preds) > threshold
    preds = preds.cpu().numpy().astype(bool)
    targets = (targets.cpu().numpy() > 0.5).astype(bool)
    
    batch_hd95 = []
    for b in range(preds.shape[0]):
        hd95_val = compute_hd95_single(preds[b, 0], targets[b, 0], voxelspacing, connectivity)
        batch_hd95.append(hd95_val)
    
    return np.mean(batch_hd95)
