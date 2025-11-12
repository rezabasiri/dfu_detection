"""
Faster R-CNN with EfficientNet backbone
Refactored from train_efficientdet.py
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5,
    efficientnet_b6, efficientnet_b7
)

from .base_model import BaseDetector


class FasterRCNNDetector(BaseDetector):
    """
    Faster R-CNN with configurable EfficientNet backbone

    Two-stage detector:
    1. Region Proposal Network (RPN) generates proposals
    2. ROI Head classifies and refines boxes

    Supported backbones:
    - efficientnet_b0: 1280 channels, ~5M params, 224x224 input
    - efficientnet_b1: 1280 channels, ~8M params, 240x240 input
    - efficientnet_b2: 1408 channels, ~9M params, 260x260 input
    - efficientnet_b3: 1536 channels, ~12M params, 300x300 input
    - efficientnet_b4: 1792 channels, ~19M params, 380x380 input (needs ~12GB VRAM)
    - efficientnet_b5: 2048 channels, ~30M params, 456x456 input (needs ~16GB VRAM)
    - efficientnet_b6: 2304 channels, ~43M params, 528x528 input (needs ~24GB VRAM)
    - efficientnet_b7: 2560 channels, ~66M params, 600x600 input (needs ~32GB VRAM)
    """

    # Backbone configurations: (function, output_channels)
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

    def __init__(self, num_classes: int, config: dict):
        """
        Initialize Faster R-CNN detector

        Args:
            num_classes: Number of classes (2 for DFU: background + ulcer)
            config: Configuration dict with keys:
                - backbone: EfficientNet variant (default: 'efficientnet_b5')
                - pretrained: Use ImageNet pretrained weights (default: True)
                - anchor_sizes: List of anchor box sizes (default: [32, 64, 128, 256, 512])
                - aspect_ratios: List of aspect ratios (default: [0.5, 1.0, 2.0])
                - rpn_positive_iou: IoU threshold for positive anchors (default: 0.7)
                - rpn_negative_iou: IoU threshold for negative anchors (default: 0.3)
                - box_nms_thresh: NMS threshold for final boxes (default: 0.5)
                - box_score_thresh: Score threshold for predictions (default: 0.05)
                - rpn_pre_nms_top_n_train: Number of proposals before NMS (train) (default: 2000)
                - rpn_pre_nms_top_n_test: Number of proposals before NMS (test) (default: 1000)
                - rpn_post_nms_top_n_train: Number of proposals after NMS (train) (default: 2000)
                - rpn_post_nms_top_n_test: Number of proposals after NMS (test) (default: 1000)
        """
        super().__init__(num_classes, config)

        # Extract config parameters with defaults
        self._backbone_name = config.get('backbone', 'efficientnet_b5')
        self.pretrained = config.get('pretrained', True)
        self.anchor_sizes = tuple(config.get('anchor_sizes', [32, 64, 128, 256, 512]))
        self.aspect_ratios = tuple(config.get('aspect_ratios', [0.5, 1.0, 2.0]))
        self.rpn_positive_iou = config.get('rpn_positive_iou', 0.7)
        self.rpn_negative_iou = config.get('rpn_negative_iou', 0.3)
        self.box_nms_thresh = config.get('box_nms_thresh', 0.5)
        self.box_score_thresh = config.get('box_score_thresh', 0.05)
        self.rpn_pre_nms_top_n_train = config.get('rpn_pre_nms_top_n_train', 2000)
        self.rpn_pre_nms_top_n_test = config.get('rpn_pre_nms_top_n_test', 1000)
        self.rpn_post_nms_top_n_train = config.get('rpn_post_nms_top_n_train', 2000)
        self.rpn_post_nms_top_n_test = config.get('rpn_post_nms_top_n_test', 1000)

        # Validate backbone
        if self._backbone_name not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Unsupported backbone: {self._backbone_name}. "
                f"Supported: {list(self.BACKBONE_CONFIGS.keys())}"
            )

        # Create model
        self.model = self._create_model()

    def _create_model(self) -> FasterRCNN:
        """Create Faster R-CNN model with EfficientNet backbone"""

        # Get backbone configuration
        efficientnet_fn, out_channels = self.BACKBONE_CONFIGS[self._backbone_name]

        # Create EfficientNet backbone
        weights = "IMAGENET1K_V1" if self.pretrained else None
        efficientnet = efficientnet_fn(weights=weights)

        # Extract feature extractor (remove classifier and pooling)
        backbone_model = nn.Sequential(*list(efficientnet.children())[:-2])
        backbone_model.out_channels = out_channels

        # Create anchor generator (single feature map from EfficientNet)
        anchor_generator = AnchorGenerator(
            sizes=(self.anchor_sizes,),  # Single feature map
            aspect_ratios=(self.aspect_ratios,)
        )

        # Create ROI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2
        )

        # Create Faster R-CNN model
        model = FasterRCNN(
            backbone=backbone_model,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            # RPN parameters
            rpn_pre_nms_top_n_train=self.rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=self.rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=self.rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=self.rpn_post_nms_top_n_test,
            rpn_positive_fraction=0.5,
            rpn_fg_iou_thresh=self.rpn_positive_iou,
            rpn_bg_iou_thresh=self.rpn_negative_iou,
            # Box parameters
            box_score_thresh=self.box_score_thresh,
            box_nms_thresh=self.box_nms_thresh,
            box_detections_per_img=100,
            box_positive_fraction=0.25,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
        )

        return model

    def get_model(self) -> nn.Module:
        """Return the underlying Faster R-CNN model"""
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
        return "faster_rcnn"

    @property
    def backbone_name(self) -> str:
        """Return backbone name"""
        return self._backbone_name

    def get_config_summary(self) -> str:
        """Get human-readable config summary"""
        return (
            f"Faster R-CNN Configuration:\n"
            f"  Backbone: {self._backbone_name}\n"
            f"  Pretrained: {self.pretrained}\n"
            f"  Anchor sizes: {self.anchor_sizes}\n"
            f"  Aspect ratios: {self.aspect_ratios}\n"
            f"  RPN positive IoU: {self.rpn_positive_iou}\n"
            f"  RPN negative IoU: {self.rpn_negative_iou}\n"
            f"  Box NMS threshold: {self.box_nms_thresh}\n"
            f"  Box score threshold: {self.box_score_thresh}\n"
        )
