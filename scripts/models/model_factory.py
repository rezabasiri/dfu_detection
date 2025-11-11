"""
Model Factory for creating detection models
Unified interface for Faster R-CNN, RetinaNet, and YOLO
"""

import yaml
from pathlib import Path
from typing import Dict, Optional
import torch

from .base_model import BaseDetector
from .faster_rcnn import FasterRCNNDetector
from .retinanet import RetinaNetDetector
from .yolo import YOLODetector


class ModelFactory:
    """
    Factory for creating detection models

    Usage:
        # Create from config file
        model = ModelFactory.create_model('faster_rcnn', num_classes=2, config_path='configs/faster_rcnn_b5.yaml')

        # Create with default config
        model = ModelFactory.create_model('retinanet', num_classes=2)

        # Create from checkpoint (auto-detects model type)
        model = ModelFactory.create_from_checkpoint('checkpoints/faster_rcnn/best_model.pth')
    """

    # Registry of available models
    MODELS = {
        'faster_rcnn': FasterRCNNDetector,
        'retinanet': RetinaNetDetector,
        'yolo': YOLODetector
    }

    @classmethod
    def create_model(
        cls,
        model_name: str,
        num_classes: int,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ) -> BaseDetector:
        """
        Create detection model

        Args:
            model_name: Model type ('faster_rcnn', 'retinanet', 'yolo')
            num_classes: Number of classes (including background for Faster R-CNN/RetinaNet)
            config: Configuration dictionary (optional)
            config_path: Path to YAML config file (optional, overrides config dict)

        Returns:
            BaseDetector instance

        Raises:
            ValueError: If model_name is not recognized
        """
        # Validate model name
        if model_name not in cls.MODELS:
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Available models: {list(cls.MODELS.keys())}"
            )

        # Load config from file if provided
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)

            # Extract model config from file (may have training/eval sections too)
            if 'model' in file_config:
                config = file_config['model']
            else:
                config = file_config

        # Use default config if none provided
        if config is None:
            config = cls.get_default_config(model_name)

        # Create model instance
        model_class = cls.MODELS[model_name]
        model = model_class(num_classes=num_classes, config=config)

        return model

    @classmethod
    def create_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> BaseDetector:
        """
        Create model from checkpoint (auto-detects model type)

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on (default: CPU)

        Returns:
            BaseDetector instance with loaded weights

        Raises:
            ValueError: If checkpoint doesn't contain model metadata
        """
        if device is None:
            device = torch.device('cpu')

        # Load checkpoint to inspect metadata
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract model information
        if 'model_name' not in checkpoint:
            raise ValueError(
                f"Checkpoint does not contain 'model_name' metadata. "
                f"Cannot auto-detect model type. Available keys: {list(checkpoint.keys())}"
            )

        model_name = checkpoint['model_name']
        num_classes = checkpoint.get('num_classes', 2)
        model_config = checkpoint.get('model_config', {})

        # Create model
        model = cls.create_model(
            model_name=model_name,
            num_classes=num_classes,
            config=model_config
        )

        # Load weights
        model.load_checkpoint(checkpoint_path, device=device)

        print(f"✓ Loaded {model_name} model from checkpoint")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Backbone: {model.backbone_name}")

        return model

    @staticmethod
    def get_default_config(model_name: str) -> Dict:
        """
        Get default configuration for a model

        Args:
            model_name: Model type

        Returns:
            Dict with default configuration
        """
        defaults = {
            'faster_rcnn': {
                'backbone': 'efficientnet_b5',
                'pretrained': True,
                'anchor_sizes': [32, 64, 128, 256, 512],
                'aspect_ratios': [0.5, 1.0, 2.0],
                'rpn_positive_iou': 0.5,  # Lowered from 0.7 for better recall
                'rpn_negative_iou': 0.3,
                'box_nms_thresh': 0.5,
                'box_score_thresh': 0.05,
                'rpn_pre_nms_top_n_train': 2000,
                'rpn_pre_nms_top_n_test': 1000,
                'rpn_post_nms_top_n_train': 2000,
                'rpn_post_nms_top_n_test': 1000,
            },
            'retinanet': {
                'backbone': 'efficientnet_b3',  # Lighter than B5
                'pretrained': True,
                'anchor_sizes': [32, 64, 128, 256, 512],
                'aspect_ratios': [0.5, 1.0, 2.0],
                'focal_loss_alpha': 0.25,
                'focal_loss_gamma': 2.0,
                'score_thresh': 0.05,
                'nms_thresh': 0.5,
                'detections_per_img': 100,
                'topk_candidates': 1000,
            },
            'yolo': {
                'model_size': 'yolov8m',  # Medium size (good balance)
                'img_size': 640,
                'conf_thresh': 0.25,
                'iou_thresh': 0.45,
                'max_det': 100,
                'pretrained': True,
            }
        }

        return defaults.get(model_name, {})

    @classmethod
    def list_available_models(cls) -> list:
        """Return list of available model names"""
        return list(cls.MODELS.keys())

    @classmethod
    def print_model_info(cls, model_name: str):
        """Print information about a model type"""
        if model_name not in cls.MODELS:
            print(f"Unknown model: {model_name}")
            print(f"Available: {cls.list_available_models()}")
            return

        model_class = cls.MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"Class: {model_class.__name__}")
        print(f"Module: {model_class.__module__}")
        print(f"\nDescription:")
        if model_class.__doc__:
            print(model_class.__doc__)
        print(f"\nDefault Configuration:")
        config = cls.get_default_config(model_name)
        for key, value in config.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")


# Utility function for easy imports
def create_model(model_name: str, num_classes: int, config_path: Optional[str] = None) -> BaseDetector:
    """
    Convenience function to create a model

    Args:
        model_name: 'faster_rcnn', 'retinanet', or 'yolo'
        num_classes: Number of classes
        config_path: Optional path to config file

    Returns:
        BaseDetector instance
    """
    return ModelFactory.create_model(model_name, num_classes, config_path=config_path)


def create_from_checkpoint(checkpoint_path: str, device: Optional[torch.device] = None) -> BaseDetector:
    """
    Convenience function to create model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        BaseDetector instance with loaded weights
    """
    return ModelFactory.create_from_checkpoint(checkpoint_path, device)
