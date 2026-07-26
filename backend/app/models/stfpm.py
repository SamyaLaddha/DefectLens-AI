"""
STFPM Model Architecture definitions.

This module contains the student and teacher feature extraction models,
multi-scale pyramid feature maps matching, discrepancy loss definitions,
and save/load helpers.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_backbone(backbone_name: str, pretrained: bool = True):
    """
    Instantiate and return the requested torchvision ResNet backbone.

    Supports resnet18, resnet34, resnet50, and wide_resnet50_2.
    Handles older and newer torchvision weight APIs gracefully.
    """
    model_fn = getattr(models, backbone_name, None)
    if model_fn is None:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if pretrained:
        # Modern torchvision weights API
        try:
            weight_mapping = {
                "resnet18": "ResNet18_Weights",
                "resnet34": "ResNet34_Weights",
                "resnet50": "ResNet50_Weights",
                "wide_resnet50_2": "Wide_ResNet50_2_Weights"
            }
            enum_name = weight_mapping.get(backbone_name)
            if enum_name and hasattr(models, enum_name):
                enum_class = getattr(models, enum_name)
                weights = getattr(enum_class, "IMAGENET1K_V1", None) or getattr(enum_class, "DEFAULT", None)
                return model_fn(weights=weights)
        except Exception:
            # Fall back to older pretrained parameter if loading weights fails
            pass
        return model_fn(pretrained=True)
    else:
        try:
            return model_fn(weights=None)
        except Exception:
            return model_fn(pretrained=False)

class FeatureExtractor(nn.Module):
    """
    Wraps a standard ResNet backbone and extracts layer1, layer2, and layer3 features.
    """
    def __init__(self, backbone_model):
        """
        Manually hook required stages from the torchvision ResNet backbone.
        """
        super().__init__()
        self.conv1 = backbone_model.conv1
        self.bn1 = backbone_model.bn1
        self.relu = backbone_model.relu
        self.maxpool = backbone_model.maxpool
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3

    def forward(self, x: torch.Tensor):
        """
        Execute sequential layers and gather intermediate outputs.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        
        return features

class STFPM(nn.Module):
    """
    Student-Teacher Feature Pyramid Matching (STFPM) network.
    """
    def __init__(self, backbone_name: str = "resnet18"):
        """
        Initialize frozen teacher and trainable student networks.
        """
        super().__init__()
        self.backbone_name = backbone_name

        # Load teacher with pretrained ImageNet weights
        teacher_base = get_backbone(backbone_name, pretrained=True)
        self.teacher = FeatureExtractor(teacher_base)
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Enforce evaluation mode on teacher by default
        self.teacher.eval()

        # Load student with randomly initialized weights
        student_base = get_backbone(backbone_name, pretrained=False)
        self.student = FeatureExtractor(student_base)
        self.student.train()

    def train(self, mode: bool = True):
        """
        Overridden train method to ensure the frozen teacher remains in evaluation mode.
        """
        super().train(mode)
        self.student.train(mode)
        # Force the teacher to remain strictly in evaluation mode (important for BatchNorm running statistics)
        self.teacher.eval()

    def forward(self, x: torch.Tensor):
        """
        Extract teacher and student feature pyramids.
        """
        with torch.no_grad():
            teacher_features = self.teacher(x)
        student_features = self.student(x)
        return teacher_features, student_features

def compute_discrepancy_loss(teacher_features, student_features):
    """
    Compute pixel-wise feature discrepancy loss between teacher and student.

    L2 normalizes feature maps channel-wise and computes the average squared L2 distance.
    """
    loss = 0.0
    for ft, fs in zip(teacher_features, student_features):
        # L2 normalization along channel dimension (dim=1)
        ft_norm = F.normalize(ft, p=2, dim=1)
        fs_norm = F.normalize(fs, p=2, dim=1)
        # Compute pixel-wise squared L2 difference (summed over channels)
        sq_diff = (ft_norm - fs_norm) ** 2
        # Average across batch, height, and width
        layer_loss = torch.mean(torch.sum(sq_diff, dim=1))
        loss += layer_loss
    return loss

def save_checkpoint(model: STFPM, filepath: str, epoch: int, metadata: dict = None):
    """
    Save student weights and training metadata (epoch, category, etc.) to disk.
    """
    checkpoint = {
        "student_state_dict": model.student.state_dict(),
        "epoch": epoch,
        "backbone_name": model.backbone_name,
        "metadata": metadata or {}
    }
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    torch.save(checkpoint, filepath)

def load_checkpoint(model: STFPM, filepath: str):
    """
    Load student weights and retrieve checkpoint metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location="cpu")
    
    # Verify loaded backbone name matches current model configuration
    loaded_backbone = checkpoint.get("backbone_name")
    if loaded_backbone and loaded_backbone != model.backbone_name:
        raise ValueError(
            f"Mismatched backbone: model expects '{model.backbone_name}', "
            f"but checkpoint contains '{loaded_backbone}'"
        )
        
    model.student.load_state_dict(checkpoint["student_state_dict"])
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metadata": checkpoint.get("metadata", {})
    }
