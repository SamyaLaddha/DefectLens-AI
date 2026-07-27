"""
Heatmap and anomaly map construction module for DefectLens AI.

Consolidates the calculations of pixel-level anomaly maps and colormap overlays.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_anomaly_map(teacher_features, student_features, image_size: int) -> np.ndarray:
    """
    Computes the pixel-level anomaly map from teacher and student feature pyramids.

    Normalizes features, calculates the squared L2 discrepancy maps for each layer,
    interpolates them to the target image_size, and multiplies them element-wise.
    """
    anomaly_maps = []
    for ft, fs in zip(teacher_features, student_features):
        # L2 normalize channel-wise
        ft_norm = F.normalize(ft, p=2, dim=1)
        fs_norm = F.normalize(fs, p=2, dim=1)
        
        # Compute squared L2 distance at each spatial location
        sq_diff = torch.sum((ft_norm - fs_norm) ** 2, dim=1, keepdim=True)
        
        # Interpolate discrepancy map to the output image size
        interpolated = F.interpolate(
            sq_diff, 
            size=(image_size, image_size), 
            mode="bilinear", 
            align_corners=False
        )
        anomaly_maps.append(interpolated)

    # Element-wise product of all layers' anomaly maps
    combined_map = anomaly_maps[0] * anomaly_maps[1] * anomaly_maps[2]
    
    # Return as a squeezed NumPy array of shape [H, W]
    return combined_map.squeeze().cpu().numpy()

def generate_heatmap_overlay(raw_image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
    """
    Overlays a color-mapped anomaly heatmap onto a raw RGB input image.

    - raw_image: numpy float array of shape [H, W, C], scaled [0, 1]
    - anomaly_map: numpy float array of shape [H, W]
    Returns a blended RGB image of shape [H, W, C].
    """
    # Normalize anomaly map to [0, 1] range for visualization
    am_min, am_max = anomaly_map.min(), anomaly_map.max()
    if am_max > am_min:
        anomaly_map_norm = (anomaly_map - am_min) / (am_max - am_min)
    else:
        anomaly_map_norm = np.zeros_like(anomaly_map)

    # Apply JET colormap to convert anomaly intensities to RGB
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(anomaly_map_norm)[:, :, :3]

    # Blend raw image and colorized heatmap map (50/50 mix)
    overlay = (raw_image * 0.5 + heatmap_rgb * 0.5)
    return np.clip(overlay, 0.0, 1.0)
