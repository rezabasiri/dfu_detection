"""
Detection Models Zoo

Unified interface for multiple object detection architectures:
- Faster R-CNN with EfficientNet backbone
- RetinaNet with EfficientNet backbone
- YOLOv8

Usage:
    from models import ModelFactory, create_model

    # Create model from config
    model = create_model('faster_rcnn', num_classes=2, config_path='configs/faster_rcnn.yaml')

    # Create model from checkpoint (auto-detects type)
    model = ModelFactory.create_from_checkpoint('checkpoints/best_model.pth')
"""

from .base_model import BaseDetector
from .faster_rcnn import FasterRCNNDetector
from .retinanet import RetinaNetDetector
from .yolo import YOLODetector
from .model_factory import ModelFactory, create_model, create_from_checkpoint

__all__ = [
    'BaseDetector',
    'FasterRCNNDetector',
    'RetinaNetDetector',
    'YOLODetector',
    'ModelFactory',
    'create_model',
    'create_from_checkpoint',
]
