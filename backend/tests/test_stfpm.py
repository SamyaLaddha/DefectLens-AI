"""
Unit tests for STFPM model architecture, loss function, and checkpoints.
"""

import os
import tempfile
import pytest
# pyrefly: ignore [missing-import]
import torch
import torchvision.models as models
from unittest.mock import patch

from app.models.stfpm import STFPM, compute_discrepancy_loss, save_checkpoint, load_checkpoint

@pytest.fixture
def mock_resnet_builder():
    """
    Patch resnet constructors so they always instantiate without pretrained weights
    to avoid internet downloads, while recording the requested arguments.
    """
    original_resnet18 = models.resnet18
    original_resnet34 = models.resnet34

    calls = {"resnet18": [], "resnet34": []}

    def mock_r18(*args, **kwargs):
        calls["resnet18"].append((args, kwargs))
        kwargs_copy = kwargs.copy()
        if "weights" in kwargs_copy:
            kwargs_copy["weights"] = None
        if "pretrained" in kwargs_copy:
            kwargs_copy["pretrained"] = False
        return original_resnet18(*args, **kwargs_copy)

    def mock_r34(*args, **kwargs):
        calls["resnet34"].append((args, kwargs))
        kwargs_copy = kwargs.copy()
        if "weights" in kwargs_copy:
            kwargs_copy["weights"] = None
        if "pretrained" in kwargs_copy:
            kwargs_copy["pretrained"] = False
        return original_resnet34(*args, **kwargs_copy)

    with patch("torchvision.models.resnet18", side_effect=mock_r18), \
         patch("torchvision.models.resnet34", side_effect=mock_r34):
        yield calls

def test_stfpm_weights_initialization(mock_resnet_builder):
    """
    Verifies that the teacher is requested with pretrained weights
    and the student is requested with no pretrained weights.
    """
    model = STFPM(backbone_name="resnet18")

    r18_calls = mock_resnet_builder["resnet18"]
    assert len(r18_calls) == 2  # 1 teacher, 1 student

    # First call (teacher) should have requested weights (ImageNet weights)
    _, teacher_kwargs = r18_calls[0]
    assert "weights" in teacher_kwargs or "pretrained" in teacher_kwargs
    if "weights" in teacher_kwargs:
        assert teacher_kwargs["weights"] is not None

    # Second call (student) should not request pretrained weights
    _, student_kwargs = r18_calls[1]
    if "weights" in student_kwargs:
        assert student_kwargs["weights"] is None
    if "pretrained" in student_kwargs:
        assert student_kwargs["pretrained"] is False

def test_stfpm_grad_and_eval_modes(mock_resnet_builder):
    """
    Verifies that the teacher remains frozen and in eval mode even when STFPM.train() is called.
    """
    model = STFPM(backbone_name="resnet18")

    # 1. Teacher parameters do not require gradients
    for name, param in model.teacher.named_parameters():
        assert not param.requires_grad, f"Teacher parameter {name} should be frozen"

    # 2. Student parameters require gradients
    for name, param in model.student.named_parameters():
        assert param.requires_grad, f"Student parameter {name} should be trainable"

    # 3. Model train mode changes student mode but forces teacher to eval mode
    model.train(True)
    assert model.student.training is True
    assert model.teacher.training is False

    model.train(False)
    assert model.student.training is False
    assert model.teacher.training is False

def test_stfpm_feature_dimensions(mock_resnet_builder):
    """
    Verifies feature dimensions extracted by teacher and student.
    """
    model = STFPM(backbone_name="resnet18")
    dummy_input = torch.randn(2, 3, 256, 256)

    teacher_feats, student_feats = model(dummy_input)

    # Check that 3 feature maps are returned
    assert len(teacher_feats) == 3
    assert len(student_feats) == 3

    # Check resolutions: layer1 (H/4, W/4), layer2 (H/8, W/8), layer3 (H/16, W/16)
    expected_shapes = [
        (2, 64, 64, 64),
        (2, 128, 32, 32),
        (2, 256, 16, 16)
    ]
    for i, shape in enumerate(expected_shapes):
        assert teacher_feats[i].shape == shape
        assert student_feats[i].shape == shape

def test_discrepancy_loss():
    """
    Verifies compute_discrepancy_loss calculation.
    """
    # Create matching dummy features
    feat1 = [torch.randn(2, 64, 32, 32), torch.randn(2, 128, 16, 16)]

    # 1. Identical inputs should yield exactly 0 discrepancy loss
    loss_zero = compute_discrepancy_loss(feat1, feat1)
    assert torch.isclose(loss_zero, torch.tensor(0.0), atol=1e-6)

    # 2. Different inputs should yield positive discrepancy loss
    feat2 = [torch.randn(2, 64, 32, 32), torch.randn(2, 128, 16, 16)]
    loss_val = compute_discrepancy_loss(feat1, feat2)
    assert loss_val.item() > 0.0

def test_checkpoint_save_load(mock_resnet_builder):
    """
    Verifies saving and loading checkpoints (restoring student weights and metadata, omitting thresholds).
    """
    model = STFPM(backbone_name="resnet18")

    # Get initial student weights state
    init_state = {k: v.clone() for k, v in model.student.state_dict().items()}

    # Mutate student weights slightly
    with torch.no_grad():
        for param in model.student.parameters():
            param.add_(1.0)

    mutated_state = {k: v.clone() for k, v in model.student.state_dict().items()}
    assert not torch.allclose(init_state["layer1.0.conv1.weight"], mutated_state["layer1.0.conv1.weight"])

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pth")

        # Save mutated state
        metadata = {"category": "bottle", "best_val_loss": 0.05}
        save_checkpoint(model, checkpoint_path, epoch=10, metadata=metadata)

        # Restore initial state to model
        model.student.load_state_dict(init_state)
        assert torch.allclose(model.student.layer1[0].conv1.weight, init_state["layer1.0.conv1.weight"])

        # Load from checkpoint
        loaded_info = load_checkpoint(model, checkpoint_path)

        # Verify loaded weights match mutated state
        assert torch.allclose(model.student.layer1[0].conv1.weight, mutated_state["layer1.0.conv1.weight"])

        # Verify loaded metadata
        assert loaded_info["epoch"] == 10
        assert loaded_info["metadata"]["category"] == "bottle"
        assert loaded_info["metadata"]["best_val_loss"] == 0.05
        # Ensure threshold does not exist/was not saved
        assert "threshold" not in loaded_info["metadata"]
        assert "threshold" not in loaded_info

def test_checkpoint_backbone_safety(mock_resnet_builder):
    """
    Verifies that loading a checkpoint with mismatched backbone raises a ValueError.
    """
    model_r18 = STFPM(backbone_name="resnet18")
    model_r34 = STFPM(backbone_name="resnet34")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "r18_checkpoint.pth")
        save_checkpoint(model_r18, checkpoint_path, epoch=1)

        # Attempting to load ResNet18 checkpoint into ResNet34 model should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            load_checkpoint(model_r34, checkpoint_path)
        assert "Mismatched backbone" in str(excinfo.value)
