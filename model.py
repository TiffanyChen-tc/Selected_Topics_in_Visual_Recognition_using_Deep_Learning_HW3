"""Mask R-CNN (ResNet-50 + FPN) for cell instance segmentation."""

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 5  # background + class1..class4


def get_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build Mask R-CNN with replaced box/mask heads.

    Uses a ResNet-50 + FPN backbone with smaller anchors tuned for small,
    dense cells and an increased detection limit per image.

    Args:
        num_classes: Total number of classes including background.
        pretrained: If True, load ImageNet-pretrained backbone weights.

    Returns:
        Mask R-CNN model ready for training or inference.
    """
    weights = "DEFAULT" if pretrained else None

    # Smaller anchors for small, dense cells (default: 32/64/128/256/512).
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model = maskrcnn_resnet50_fpn(
        weights=weights, rpn_anchor_generator=anchor_generator
    )

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, dim_reduced=256, num_classes=num_classes
    )

    # Allow more detections per image (default: 100); cells can be very dense.
    model.roi_heads.detections_per_img = 300
    model.rpn.pre_nms_top_n_train = 4000
    model.rpn.post_nms_top_n_train = 2000
    model.rpn.pre_nms_top_n_test = 2000
    model.rpn.post_nms_top_n_test = 1000

    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        Integer count of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_model(pretrained=False)
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")
    assert n_params < 200_000_000, "Model exceeds 200M parameter limit!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.no_grad():
        out = model([torch.rand(3, 256, 256).to(device)])
    print("Output keys:", list(out[0].keys()))
    print("Smoke-test passed.")