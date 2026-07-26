"""
Evaluation pipeline execution script for DefectLens AI.

Computes AUROCs, AUPRO, fits threshold selection configurations,
and generates visual error analysis outputs.
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from scipy.ndimage import label
from scipy.integrate import trapezoid

from app.config import settings
from app.utils.seed import set_seed
from app.utils.logging_utils import get_logger
from app.data.dataset import MVTecDataset
from app.data.transforms import get_transforms
from app.models.stfpm import STFPM, load_checkpoint

logger = get_logger("evaluation")

def calculate_pro_score(y_true_masks: np.ndarray, predicted_anomaly_maps: np.ndarray, fpr_limit: float = 0.3) -> float:
    """
    Computes pixel-level Per-Region Overlap (PRO) score up to the given FPR limit (default 0.3).
    """
    # Flatten the arrays to compute the FPR curve
    y_true_flat = y_true_masks.flatten()
    y_scores_flat = predicted_anomaly_maps.flatten()

    # Subsample if pixel count is too large to prevent memory/execution slowdowns
    if len(y_true_flat) > 5000000:
        y_true_flat = y_true_flat[::10]
        y_scores_flat = y_scores_flat[::10]

    fpr, tpr, thresholds = roc_curve(y_true_flat, y_scores_flat)

    # Find the indices corresponding to FPR <= fpr_limit
    valid_idx = np.where(fpr <= fpr_limit)[0]
    if len(valid_idx) == 0:
        return 0.0

    # Select at most 100 thresholds to compute PRO for performance
    thresholds_to_test = thresholds[valid_idx]
    fpr_to_test = fpr[valid_idx]
    if len(thresholds_to_test) > 100:
        indices = np.linspace(0, len(thresholds_to_test) - 1, 100, dtype=int)
        thresholds_to_test = thresholds_to_test[indices]
        fpr_to_test = fpr_to_test[indices]

    # Pre-calculate connected components (regions) for anomalous ground truth masks
    labeled_regions = []
    num_regions = 0
    anomalous_indices = []

    for idx, mask in enumerate(y_true_masks):
        if mask.sum() > 0:
            lbl, num_feats = label(mask)
            labeled_regions.append((lbl, num_feats))
            num_regions += num_feats
            anomalous_indices.append(idx)

    if num_regions == 0:
        return 1.0  # Safe default if no anomalies exist

    pro_scores = []
    for th in thresholds_to_test:
        # Binary prediction map at threshold th
        pred_binary = (predicted_anomaly_maps > th).astype(int)
        
        region_overlaps = []
        for i, idx in enumerate(anomalous_indices):
            lbl, num_feats = labeled_regions[i]
            for region_idx in range(1, num_feats + 1):
                region_mask = (lbl == region_idx)
                overlap_count = (pred_binary[idx] & region_mask).sum()
                region_size = region_mask.sum()
                region_overlaps.append(overlap_count / region_size)
        
        pro_scores.append(np.mean(region_overlaps))

    pro_scores = np.array(pro_scores)
    fpr_to_test = np.array(fpr_to_test)

    # Sort values by FPR to ensure ascending order for integration
    sort_idx = np.argsort(fpr_to_test)
    fpr_sorted = fpr_to_test[sort_idx]
    pro_sorted = pro_scores[sort_idx]

    # Integrate the PRO curve up to the fpr_limit and normalize by dividing by the limit
    aupro = trapezoid(pro_sorted, fpr_sorted) / fpr_limit
    return float(aupro)

def select_threshold(y_true: np.ndarray, y_scores: np.ndarray, method: str = "youden"):
    """
    Fits classification thresholds on the continuous anomaly scores.

    - youden: maximizes Youden's J statistic (Sensitivity + Specificity - 1)
    - f1_max: maximizes the F1 score over the precision-recall curve
    """
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    elif method == "f1_max":
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresholds):
            return float(thresholds[best_idx])
        else:
            return float(thresholds[-1])
    else:
        raise ValueError(f"Unknown threshold selection method: {method}")

def save_error_analysis_image(image_path: str, raw_image: np.ndarray, mask: np.ndarray, anomaly_map: np.ndarray, output_path: str):
    """
    Generates and saves a three-column visualization figure:
    [Raw Image, Ground Truth Mask, Heatmap Overlay]
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Normalize anomaly map to [0, 1] for visualization
    am_min, am_max = anomaly_map.min(), anomaly_map.max()
    if am_max > am_min:
        anomaly_map_norm = (anomaly_map - am_min) / (am_max - am_min)
    else:
        anomaly_map_norm = np.zeros_like(anomaly_map)

    # Apply JET colormap
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(anomaly_map_norm)[:, :, :3]

    # Create overlay
    overlay = (raw_image * 0.5 + heatmap_rgb * 0.5)

    # Set up matplotlib figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw_image)
    axes[0].set_title("Raw Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def evaluate_model():
    """
    Executes the main model evaluation logic on the test dataset split.
    """
    set_seed(settings.SEED, settings.DETERMINISTIC)

    # Determine execution device
    if settings.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(settings.DEVICE)
    logger.info(f"Evaluation running on device: {device}")

    # Set up directories
    checkpoint_dir = os.path.join(settings.CHECKPOINT_DIR, settings.CATEGORY, settings.RUN_NAME)
    best_path = os.path.join(checkpoint_dir, "best.pth")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best model checkpoint not found at: {best_path}")

    # Load transforms and dataset
    _, test_transform = get_transforms(
        image_size=settings.IMAGE_SIZE,
        horizontal_flip=settings.HORIZONTAL_FLIP
    )

    test_dataset = MVTecDataset(
        dataset_path=settings.DATASET_PATH,
        category=settings.CATEGORY,
        split="test",
        transform=test_transform,
        image_size=settings.IMAGE_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=settings.NUM_WORKERS
    )

    # Load model
    model = STFPM(backbone_name=settings.BACKBONE)
    load_checkpoint(model, best_path)
    model.to(device)
    model.eval()

    logger.info("Running inference on test dataset split...")

    # Data collection lists
    y_true_img = []
    y_scores_img = []
    y_true_pix = []
    y_scores_pix = []
    raw_images = []
    image_paths = []

    with torch.no_grad():
        for idx, (images, labels, masks) in enumerate(test_loader):
            images = images.to(device)
            
            # Forward pass: extract features
            t_feats, s_feats = model(images)
            
            # Compute anomaly map from discrepancy normalized L2 distances
            anomaly_maps = []
            for ft, fs in zip(t_feats, s_feats):
                ft_norm = F.normalize(ft, p=2, dim=1)
                fs_norm = F.normalize(fs, p=2, dim=1)
                sq_diff = torch.sum((ft_norm - fs_norm) ** 2, dim=1, keepdim=True)
                # Interpolate to input dimensions
                interpolated = F.interpolate(
                    sq_diff, 
                    size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE), 
                    mode="bilinear", 
                    align_corners=False
                )
                anomaly_maps.append(interpolated)

            # Element-wise product of discrepancy maps across layers (standard STFPM)
            combined_map = anomaly_maps[0] * anomaly_maps[1] * anomaly_maps[2]
            combined_map = combined_map.squeeze().cpu().numpy()  # shape [H, W]

            # Image score is the maximum value of the anomaly map
            image_score = float(combined_map.max())
            
            # True image label: 1 if anomaly pixels present, 0 otherwise
            mask_np = masks.squeeze().numpy()  # shape [H, W]
            has_anomaly = int(labels.item())

            # Store results
            y_true_img.append(has_anomaly)
            y_scores_img.append(image_score)
            y_true_pix.append(mask_np)
            y_scores_pix.append(combined_map)
            image_paths.append(test_dataset.image_paths[idx])

            # Convert normalized input image to RGB numpy array [H, W, C] for visualization
            img_np = images.squeeze().cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            # Denormalize (transforms.py uses ImageNet mean/std)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np * std) + mean
            img_np = np.clip(img_np, 0, 1)
            raw_images.append(img_np)

    y_true_img = np.array(y_true_img)
    y_scores_img = np.array(y_scores_img)
    y_true_pix = np.array(y_true_pix)
    y_scores_pix = np.array(y_scores_pix)

    logger.info("Computing evaluation metrics...")

    # Calculate AUROC values
    image_auroc = roc_auc_score(y_true_img, y_scores_img)
    
    # Subsample pixel maps/masks for AUROC computation to optimize execution memory limits
    y_true_pix_flat = y_true_pix.flatten()[::4]
    y_scores_pix_flat = y_scores_pix.flatten()[::4]
    pixel_auroc = roc_auc_score(y_true_pix_flat, y_scores_pix_flat)

    # Calculate AUPRO score
    pro_score = calculate_pro_score(y_true_pix, y_scores_pix, fpr_limit=settings.PRO_FPR_LIMIT)

    # Compute both threshold selection statistics
    youden_th = select_threshold(y_true_img, y_scores_img, method="youden")
    f1_max_th = select_threshold(y_true_img, y_scores_img, method="f1_max")

    selected_th = youden_th if settings.THRESHOLD_METHOD == "youden" else f1_max_th
    logger.info(f"Thresholds Calibration - Youden's J: {youden_th:.6f} | F1-Max: {f1_max_th:.6f}")
    logger.info(f"Selected Deployment Threshold ({settings.THRESHOLD_METHOD}): {selected_th:.6f}")

    # Compute predictions at the selected threshold
    y_pred_img = (y_scores_img >= selected_th).astype(int)

    # Compute confusion matrix and related classification metrics
    cm = confusion_matrix(y_true_img, y_pred_img)
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Note: Accuracy is explicitly omitted due to dataset class imbalances
    # (e.g. anomalous samples often outnumber normal test samples in MVTec AD).
    # Precision, Recall, and F1 provide much more robust measures of calibration quality.
    logger.info(
        f"Evaluation Metrics at Threshold:\n"
        f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n"
        f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}"
    )

    # Write deployment thresholds to isolated json file
    threshold_config = {
        "threshold": selected_th,
        "threshold_method": settings.THRESHOLD_METHOD,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "pro_score": pro_score,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    
    config_path = os.path.join(checkpoint_dir, "threshold_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=4)
    logger.info(f"Saved deployment configurations to {config_path}")

    # Perform visual error analysis
    fp_idx = np.where((y_true_img == 0) & (y_pred_img == 1))[0]
    fn_idx = np.where((y_true_img == 1) & (y_pred_img == 0))[0]

    screenshots_dir = "docs/screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)

    if len(fp_idx) > 0:
        idx = fp_idx[0]
        fp_out = os.path.join(screenshots_dir, "false_positive.png")
        save_error_analysis_image(image_paths[idx], raw_images[idx], y_true_pix[idx], y_scores_pix[idx], fp_out)
        logger.info(f"Saved False Positive overlay image to {fp_out}")
    else:
        logger.info("No False Positives detected in test set.")

    if len(fn_idx) > 0:
        idx = fn_idx[0]
        fn_out = os.path.join(screenshots_dir, "false_negative.png")
        save_error_analysis_image(image_paths[idx], raw_images[idx], y_true_pix[idx], y_scores_pix[idx], fn_out)
        logger.info(f"Saved False Negative overlay image to {fn_out}")
    else:
        logger.info("No False Negatives detected in test set.")

    # Benchmark comparison report against official STFPM published numbers
    logger.info("*" * 60)
    logger.info("EVALUATION GENERALIZATION AND BENCHMARK RESULTS")
    logger.info("*" * 60)
    logger.info(f"Category: {settings.CATEGORY} | Backbone: {settings.BACKBONE}")
    logger.info(f"Image-level AUROC: {image_auroc:.4f}")
    logger.info(f"Pixel-level AUROC: {pixel_auroc:.4f}")
    logger.info(f"PRO-score (AUPRO): {pro_score:.4f}")
    logger.info("*" * 60)
    
    # Official paper values for resnet18 bottle: Image AUROC=0.988, Pixel AUROC=0.974, PRO=0.953
    logger.info("Published STFPM Paper benchmarks (ResNet18 bottle):")
    logger.info("  Image-level AUROC: 0.9880 | Pixel-level AUROC: 0.9740 | AUPRO: 0.9530")
    logger.info("*" * 60)
    logger.info("WARNING: The threshold-dependent metrics (Precision, Recall, F1) were")
    logger.info("calibrated on the test set because the validation set does not contain")
    logger.info("anomalous samples. These threshold-dependent metrics represent calibration")
    logger.info("performance, while AUROC and PRO-score remain threshold-free generalization metrics.")
    logger.info("*" * 60)

if __name__ == "__main__":
    evaluate_model()
