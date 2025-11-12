"""
Base abstract class for all detection models
Provides unified interface for Faster R-CNN, RetinaNet, and YOLO
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union


class BaseDetector(ABC):
    """
    Abstract base class for all detection models

    All models must implement this interface to ensure:
    - Consistent training loop
    - Consistent evaluation metrics
    - Consistent inference format
    - Consistent checkpointing
    """

    def __init__(self, num_classes: int, config: dict):
        """
        Initialize detector

        Args:
            num_classes: Number of classes (including background)
            config: Model configuration dictionary
        """
        self.num_classes = num_classes
        self.config = config
        self.device = None

    @abstractmethod
    def get_model(self) -> nn.Module:
        """
        Return the underlying PyTorch model

        Returns:
            nn.Module: The detection model
        """
        pass

    @abstractmethod
    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None) -> Union[Dict, List[Dict]]:
        """
        Forward pass through the model

        Args:
            images: List of image tensors [Tensor(3,H,W), ...]
            targets: Optional list of target dicts for training
                Each dict contains:
                    - boxes: Tensor(N, 4) in [xmin, ymin, xmax, ymax] format
                    - labels: Tensor(N) with class labels
                    - image_id: Tensor(1) with image ID

        Returns:
            Training mode (targets provided):
                Dict with loss values: {'loss_classifier': ..., 'loss_box_reg': ..., ...}

            Inference mode (targets=None):
                List of prediction dicts: [{'boxes': Tensor, 'scores': Tensor, 'labels': Tensor}, ...]
        """
        pass

    @abstractmethod
    def set_train_mode(self):
        """Set model to training mode"""
        pass

    @abstractmethod
    def set_eval_mode(self):
        """Set model to evaluation mode"""
        pass

    @abstractmethod
    def to(self, device: torch.device):
        """
        Move model to device

        Args:
            device: torch.device to move model to
        """
        pass

    @abstractmethod
    def get_optimizer_params(self) -> List[nn.Parameter]:
        """
        Get parameters for optimizer

        Returns:
            List of parameters that require gradients
        """
        pass

    def load_checkpoint(self, checkpoint_path: str, device: torch.device = None) -> Dict:
        """
        Load model weights from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load checkpoint on

        Returns:
            Dict with checkpoint metadata
        """
        if device is None:
            device = torch.device('cpu')

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load model state dict
        model = self.get_model()
        model.load_state_dict(checkpoint['model_state_dict'])

        return checkpoint

    def save_checkpoint(self, checkpoint_path: str, epoch: int, optimizer_state: Dict = None, **metadata) -> None:
        """
        Save model checkpoint with metadata

        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            **metadata: Additional metadata to save (losses, metrics, etc.)
        """
        model = self.get_model()

        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_name': self.name,
            'model_config': self.config,
            'num_classes': self.num_classes,
            **metadata
        }

        if optimizer_state is not None:
            checkpoint_dict['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint_dict, checkpoint_path)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return model name for identification"""
        pass

    @property
    @abstractmethod
    def backbone_name(self) -> str:
        """Return backbone name (e.g., 'efficientnet_b5')"""
        pass

    def get_model_info(self) -> Dict:
        """
        Get model information summary

        Returns:
            Dict with model parameters, backbone, etc.
        """
        model = self.get_model()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'name': self.name,
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'config': self.config
        }

    def print_model_info(self):
        """Print model information"""
        info = self.get_model_info()
        print(f"\n{'='*60}")
        print(f"Model: {info['name']}")
        print(f"Backbone: {info['backbone']}")
        print(f"{'='*60}")
        print(f"Number of classes: {info['num_classes']}")
        print(f"Total parameters: {info['total_params']:,}")
        print(f"Trainable parameters: {info['trainable_params']:,}")
        print(f"{'='*60}\n")
