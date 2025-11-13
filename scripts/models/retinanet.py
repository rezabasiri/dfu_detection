"""
RetinaNet with EfficientNet backbone
Single-stage detector with Focal Loss for handling class imbalance
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5,
    efficientnet_b6, efficientnet_b7
)

from .base_model import BaseDetector


class RetinaNetDetector(BaseDetector):
    """
    RetinaNet with configurable EfficientNet backbone

    Single-stage detector with Feature Pyramid Network (FPN) and Focal Loss

    Key advantages over Faster R-CNN:
    - Single-stage (no RPN bottleneck)
    - Focal Loss handles class imbalance (perfect for medical detection!)
    - Generally higher recall, slightly lower precision
    - Faster inference

    Focal Loss formula:
        FL(pt) = -alpha * (1-pt)^gamma * log(pt)

    Where:
        - alpha: Weighting factor for positive class (default: 0.25)
        - gamma: Focusing parameter for hard examples (default: 2.0)
        - Higher gamma = more focus on hard negatives
    """

    # Same backbone configs as Faster R-CNN
    BACKBONE_CONFIGS = {
        "efficientnet_b0": (efficientnet_b0, 1280),
        "efficientnet_b1": (efficientnet_b1, 1280),
        "efficientnet_b2": (efficientnet_b2, 1408),
        "efficientnet_b3": (efficientnet_b3, 1536),
        "efficientnet_b4": (efficientnet_b4, 1792),
        "efficientnet_b5": (efficientnet_b5, 2048),
        "efficientnet_b6": (efficientnet_b6, 2304),
        "efficientnet_b7": (efficientnet_b7, 2560),
    }

    # Channel configurations for FPN input
    # Format: [channels at layer 2, layer 3, layer 5]
    # These are the intermediate feature channels used for multi-scale detection
    BACKBONE_FPN_CHANNELS = {
        "efficientnet_b0": [24, 40, 112],
        "efficientnet_b1": [24, 40, 112],
        "efficientnet_b2": [24, 48, 120],
        "efficientnet_b3": [32, 48, 136],
        "efficientnet_b4": [32, 56, 160],
        "efficientnet_b5": [40, 64, 176],
        "efficientnet_b6": [40, 72, 200],
        "efficientnet_b7": [48, 80, 224],
    }

    def __init__(self, num_classes: int, config: dict):
        """
        Initialize RetinaNet detector

        Args:
            num_classes: Number of classes (2 for DFU: background + ulcer)
            config: Configuration dict with keys:
                - backbone: EfficientNet variant (default: 'efficientnet_b3')
                - pretrained: Use ImageNet pretrained weights (default: True)
                - anchor_sizes: List of anchor sizes (default: [32, 64, 128, 256, 512])
                - aspect_ratios: List of aspect ratios (default: [0.5, 1.0, 2.0])
                - focal_loss_alpha: Alpha for focal loss (default: 0.25)
                - focal_loss_gamma: Gamma for focal loss (default: 2.0)
                - score_thresh: Score threshold for predictions (default: 0.05)
                - nms_thresh: NMS threshold (default: 0.5)
                - detections_per_img: Max detections per image (default: 100)
                - topk_candidates: Top K candidates before NMS (default: 1000)
        """
        super().__init__(num_classes, config)

        # Extract config parameters with defaults
        self._backbone_name = config.get('backbone', 'efficientnet_b3')
        self.pretrained = config.get('pretrained', True)
        self.anchor_sizes = tuple(config.get('anchor_sizes', [32, 64, 128, 256, 512]))
        self.aspect_ratios = tuple(config.get('aspect_ratios', [0.5, 1.0, 2.0]))
        self.focal_loss_alpha = config.get('focal_loss_alpha', 0.25)
        self.focal_loss_gamma = config.get('focal_loss_gamma', 2.0)
        self.score_thresh = config.get('score_thresh', 0.05)
        self.nms_thresh = config.get('nms_thresh', 0.5)
        self.detections_per_img = config.get('detections_per_img', 100)
        self.topk_candidates = config.get('topk_candidates', 1000)

        # Validate backbone
        if self._backbone_name not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Unsupported backbone: {self._backbone_name}. "
                f"Supported: {list(self.BACKBONE_CONFIGS.keys())}"
            )

        # Create model
        self.model = self._create_model()

    def _create_efficientnet_fpn_backbone(self):
        """
        Create EfficientNet backbone with Feature Pyramid Network (FPN)

        Returns:
            backbone with FPN that outputs multi-scale features
        """
        # Get backbone configuration
        efficientnet_fn, out_channels = self.BACKBONE_CONFIGS[self._backbone_name]

        # Create EfficientNet
        weights = "IMAGENET1K_V1" if self.pretrained else None
        efficientnet = efficientnet_fn(weights=weights)

        # Extract intermediate feature channels for FPN
        # Use correct channel counts for this specific EfficientNet variant
        # We'll use layers at different strides for multi-scale features
        # Using 3 levels (P3, P4, P5) instead of 5 to save memory
        in_channels_list = self.BACKBONE_FPN_CHANNELS[self._backbone_name]
        return_layers = {'2': '0', '3': '1', '5': '2'}  # Layer indices

        # Create IntermediateLayerGetter to extract multi-scale features
        from torchvision.models._utils import IntermediateLayerGetter
        body = IntermediateLayerGetter(efficientnet.features, return_layers=return_layers)

        # Create FPN on top of the backbone
        # FPN normalizes all feature channels to out_channels_fpn
        # Using 128 channels instead of 256 to save memory
        out_channels_fpn = 128
        backbone = BackboneWithFPN(
            body,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=out_channels_fpn,
            extra_blocks=LastLevelP6P7(out_channels_fpn, out_channels_fpn),  # P6, P7 for RetinaNet
        )

        backbone.out_channels = out_channels_fpn

        return backbone

    def _create_model(self) -> RetinaNet:
        """Create RetinaNet model with EfficientNet backbone"""

        # Create backbone with FPN
        backbone = self._create_efficientnet_fpn_backbone()

        # Create anchor generator
        # RetinaNet uses anchors at multiple scales and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=(self.anchor_sizes,) * 5,  # 5 feature levels
            aspect_ratios=(self.aspect_ratios,) * 5
        )

        # Create RetinaNet model
        model = RetinaNet(
            backbone=backbone,
            num_classes=self.num_classes,
            anchor_generator=anchor_generator,
            # Focal loss parameters (built into RetinaNet)
            # Note: torchvision's RetinaNet doesn't expose alpha/gamma directly
            # but uses sigmoid_focal_loss internally
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            detections_per_img=self.detections_per_img,
            topk_candidates=self.topk_candidates,
        )

        return model

    def get_model(self) -> nn.Module:
        """Return the underlying RetinaNet model"""
        return self.model

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None) -> Union[Dict, List[Dict]]:
        """
        Forward pass

        Args:
            images: List of image tensors [Tensor(3,H,W), ...]
            targets: Optional list of target dicts for training

        Returns:
            Training mode: Dict with losses
            Inference mode: List of prediction dicts
        """
        return self.model(images, targets)

    def set_train_mode(self):
        """Set model to training mode"""
        self.model.train()

    def set_eval_mode(self):
        """Set model to evaluation mode"""
        self.model.eval()

    def to(self, device: torch.device):
        """Move model to device"""
        self.model = self.model.to(device)
        self.device = device
        return self

    def get_optimizer_params(self) -> List[nn.Parameter]:
        """Get trainable parameters"""
        return [p for p in self.model.parameters() if p.requires_grad]

    @property
    def name(self) -> str:
        """Return model name"""
        return "retinanet"

    @property
    def backbone_name(self) -> str:
        """Return backbone name"""
        return self._backbone_name

    def get_config_summary(self) -> str:
        """Get human-readable config summary"""
        return (
            f"RetinaNet Configuration:\n"
            f"  Backbone: {self._backbone_name}\n"
            f"  Pretrained: {self.pretrained}\n"
            f"  Anchor sizes: {self.anchor_sizes}\n"
            f"  Aspect ratios: {self.aspect_ratios}\n"
            f"  Focal Loss alpha: {self.focal_loss_alpha}\n"
            f"  Focal Loss gamma: {self.focal_loss_gamma}\n"
            f"  Score threshold: {self.score_thresh}\n"
            f"  NMS threshold: {self.nms_thresh}\n"
            f"  Max detections: {self.detections_per_img}\n"
        )
