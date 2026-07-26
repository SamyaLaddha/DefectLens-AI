"""
Training pipeline execution script for DefectLens AI.

Performs training and validation loops, logs metrics, checks early stopping,
and saves model checkpoints based on validation loss performance.
"""

import os
import csv
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from app.config import settings
from app.utils.seed import set_seed
from app.utils.logging_utils import get_logger
from app.data.dataset import MVTecDataset
from app.data.transforms import get_transforms
from app.models.stfpm import STFPM, compute_discrepancy_loss, save_checkpoint

logger = get_logger("training")

def run_sanity_check(train_dataset, val_dataset):
    """
    Perform dataset sizes and zero-overlap sanity validation checks.
    """
    logger.info("Executing pre-run dataset sanity checks...")
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    
    logger.info(f"Loaded training samples: {train_size}")
    logger.info(f"Loaded validation samples: {val_size}")

    if train_size == 0:
        raise ValueError("Training dataset cannot have 0 samples.")

    # Convert paths to sets to check for overlap
    train_paths = set(train_dataset.image_paths)
    val_paths = set(val_dataset.image_paths)
    
    overlap = train_paths.intersection(val_paths)
    if len(overlap) > 0:
        raise ValueError(f"Sanity Check Failed: Overlap detected between train and val splits ({len(overlap)} files).")
    
    logger.info("Dataset sanity checks passed successfully (zero overlap confirmed).")

def train_model(epochs: int = None):
    """
    Main training execution function.
    """
    # Enforce deterministic random states
    set_seed(settings.SEED, settings.DETERMINISTIC)

    # Determine execution device
    if settings.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(settings.DEVICE)
    logger.info(f"Training pipeline running on device: {device}")

    # Initialize category and run-specific folders
    checkpoint_dir = os.path.join(settings.CHECKPOINT_DIR, settings.CATEGORY, settings.RUN_NAME)
    log_dir = os.path.join(settings.LOG_DIR, settings.CATEGORY, settings.RUN_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Instantiate datasets
    train_transform, test_transform = get_transforms(
        image_size=settings.IMAGE_SIZE,
        horizontal_flip=settings.HORIZONTAL_FLIP
    )

    train_dataset = MVTecDataset(
        dataset_path=settings.DATASET_PATH,
        category=settings.CATEGORY,
        split="train",
        transform=train_transform,
        val_split=settings.VAL_SPLIT,
        seed=settings.SEED,
        image_size=settings.IMAGE_SIZE
    )

    val_dataset = MVTecDataset(
        dataset_path=settings.DATASET_PATH,
        category=settings.CATEGORY,
        split="val",
        transform=test_transform,  # Validation uses standard test/eval transforms
        val_split=settings.VAL_SPLIT,
        seed=settings.SEED,
        image_size=settings.IMAGE_SIZE
    )

    # Perform dataset checks prior to running training loop
    run_sanity_check(train_dataset, val_dataset)

    # Setup DataLoader wrappers
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=settings.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        num_workers=settings.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # Instantiate STFPM model
    model = STFPM(backbone_name=settings.BACKBONE)
    model.to(device)

    # Setup optimization
    optimizer = optim.Adam(
        model.student.parameters(),
        lr=settings.LEARNING_RATE,
        weight_decay=1e-5
    )

    total_epochs = epochs if epochs is not None else settings.EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Initialize loggers
    writer = SummaryWriter(log_dir=log_dir)
    csv_path = os.path.join(log_dir, "history.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    try:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

        # Tracking parameters
        best_val_loss = float("inf")
        epochs_no_improve = 0
        patience = settings.PATIENCE

        logger.info("Starting training loop execution...")

        for epoch in range(1, total_epochs + 1):
            # 1. Training step
            model.train()  # Student in train, Teacher in eval
            train_loss = 0.0
            
            for images, _, _ in train_loader:
                images = images.to(device)
                optimizer.zero_grad()
                
                # Forward pass: teacher features extracted without gradients (torch.no_grad in model forward)
                teacher_feats, student_feats = model(images)
                loss = compute_discrepancy_loss(teacher_feats, student_feats)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

            epoch_train_loss = train_loss / len(train_dataset)

            # 2. Validation step
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for val_images, _, _ in val_loader:
                    val_images = val_images.to(device)
                    t_feats, s_feats = model(val_images)
                    v_loss = compute_discrepancy_loss(t_feats, s_feats)
                    val_loss += v_loss.item() * val_images.size(0)

            epoch_val_loss = val_loss / len(val_dataset)
            current_lr = scheduler.get_last_lr()[0]

            # Step learning rate
            scheduler.step()

            # Write to logs
            csv_writer.writerow([epoch, epoch_train_loss, epoch_val_loss, current_lr])
            csv_file.flush()
            
            writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
            writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
            writer.add_scalar("LR", current_lr, epoch)

            logger.info(
                f"Epoch {epoch}/{total_epochs} | "
                f"Train Loss: {epoch_train_loss:.6f} | "
                f"Val Loss: {epoch_val_loss:.6f} | "
                f"LR: {current_lr:.6f}"
            )

            # 3. Model checkpointing
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                
                metadata = {
                    "category": settings.CATEGORY,
                    "run_name": settings.RUN_NAME,
                    "best_val_loss": best_val_loss
                }
                best_path = os.path.join(checkpoint_dir, "best.pth")
                save_checkpoint(model, best_path, epoch, metadata)
                logger.info(f"Saved new best model checkpoint to {best_path}")
            else:
                epochs_no_improve += 1

            # 4. Early stopping validation
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no val loss improvement for {patience} epochs).")
                break

        # Save final model
        final_path = os.path.join(checkpoint_dir, "final.pth")
        save_checkpoint(
            model, 
            final_path, 
            epoch, 
            metadata={
                "category": settings.CATEGORY, 
                "run_name": settings.RUN_NAME, 
                "final_val_loss": epoch_val_loss
            }
        )
        logger.info(f"Saved final model checkpoint to {final_path}")

    finally:
        # Close file handles and loggers
        csv_file.close()
        writer.close()
        logger.info("Training pipeline execution completed successfully.")

if __name__ == "__main__":
    train_model()
