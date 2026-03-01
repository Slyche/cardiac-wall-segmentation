"""
Utility functions for evaluation, post-processing, and visualization.

Used by the evaluation notebook for metrics computation,
mask post-processing, and result visualization.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


# ══════════════════════════════════════════════════════════════
# Segmentation Metrics
# ══════════════════════════════════════════════════════════════

def dice_score(pred_c, gt_c):
    """Dice similarity coefficient for a single binary class."""
    inter = (pred_c * gt_c).sum()
    union = pred_c.sum() + gt_c.sum()
    return (2.0 * inter / union) if union > 0 else np.nan


def iou_score(pred_c, gt_c):
    """Intersection over Union for a single binary class."""
    inter = (pred_c * gt_c).sum()
    union = pred_c.sum() + gt_c.sum() - inter
    return (inter / union) if union > 0 else np.nan


def hausdorff_dist(pred_c, gt_c):
    """Symmetric Hausdorff distance between two binary masks."""
    pred_pts = np.argwhere(pred_c)
    gt_pts = np.argwhere(gt_c)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    return max(directed_hausdorff(pred_pts, gt_pts)[0],
               directed_hausdorff(gt_pts, pred_pts)[0])


def mean_absolute_dist(pred_c, gt_c):
    """Mean symmetric contour-to-contour distance."""
    pred_contour = pred_c.astype(bool) & ~binary_erosion(pred_c.astype(bool))
    gt_contour = gt_c.astype(bool) & ~binary_erosion(gt_c.astype(bool))
    pred_pts = np.argwhere(pred_contour)
    gt_pts = np.argwhere(gt_contour)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)
    return (d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0


# ══════════════════════════════════════════════════════════════
# Post-Processing
# ══════════════════════════════════════════════════════════════

def post_process(pred_mask, num_classes=4):
    """Keep only the largest connected component per foreground class."""
    result = np.zeros_like(pred_mask)
    for c in range(1, num_classes):
        binary = (pred_mask == c).astype(np.uint8)
        if binary.sum() == 0:
            continue
        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            continue
        sizes = np.bincount(labeled.flat)[1:]
        largest = sizes.argmax() + 1
        result[labeled == largest] = c
    return result


# ══════════════════════════════════════════════════════════════
# Visualization Helpers
# ══════════════════════════════════════════════════════════════

SEG_COLORS = {
    0: [0, 0, 0],       # Background
    1: [0, 255, 255],    # LV Cavity (cyan)
    2: [255, 255, 0],    # Myocardium (yellow)
    3: [139, 0, 0],      # Left Atrium (dark red)
}


def mask_to_rgb(mask):
    """Convert a class-index mask to an RGB image for visualization."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in SEG_COLORS.items():
        rgb[mask == c] = color
    return rgb


def parse_filename(filepath):
    """Extract view (2CH/4CH) and phase (ED/ES/seq) from CAMUS filename."""
    import os
    fname = os.path.basename(filepath)
    view = '2CH' if '2CH' in fname else '4CH' if '4CH' in fname else 'unknown'
    if 'half_sequence' in fname:
        phase = 'seq'
    elif '_ED' in fname:
        phase = 'ED'
    elif '_ES' in fname:
        phase = 'ES'
    else:
        phase = 'unknown'
    return view, phase
