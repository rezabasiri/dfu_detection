"""
YOLOv8 wrapper for unified interface
Handles format conversion between YOLO and our standard format
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics package not found. YOLO models will not be available.")
    print("Install with: pip install ultralytics")

from .base_model import BaseDetector


class YOLODetector(BaseDetector):
    """
    YOLOv8 wrapper for unified interface

    Challenges:
    - YOLO uses different input format (expects batch of tensors or numpy)
    - YOLO uses different output format (Results objects)
    - YOLO has its own training loop (we need to adapt)

    Solutions:
    - Convert between our format and YOLO format
    - Wrap YOLO's training in our interface
    - Standardize outputs to match Faster R-CNN format

    Supported models:
    - yolov8n: Nano (3.2M params, fastest)
    - yolov8s: Small (11.2M params)
    - yolov8m: Medium (25.9M params) - recommended balance
    - yolov8l: Large (43.7M params)
    - yolov8x: XLarge (68.2M params, most accurate)
    """

    def __init__(self, num_classes: int, config: dict):
        """
        Initialize YOLO detector

        Args:
            num_classes: Number of classes (excluding background)
                Note: YOLO doesn't have explicit background class
            config: Configuration dict with keys:
                - model_size: YOLO variant ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
                - img_size: Input image size (default: 640)
                - conf_thresh: Confidence threshold (default: 0.25)
                - iou_thresh: IoU threshold for NMS (default: 0.45)
                - max_det: Maximum detections per image (default: 100)
                - pretrained: Use COCO pretrained weights (default: True)
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics package is required for YOLO models. "
                "Install with: pip install ultralytics"
            )

        super().__init__(num_classes, config)

        # Extract config parameters
        self.model_size = config.get('model_size', 'yolov8m')
        self.img_size = config.get('img_size', 640)
        self.conf_thresh = config.get('conf_thresh', 0.25)
        self.iou_thresh = config.get('iou_thresh', 0.45)
        self.max_det = config.get('max_det', 100)
        self.pretrained = config.get('pretrained', True)

        # YOLO doesn't include background in num_classes
        # Our num_classes includes background, so subtract 1
        self.yolo_num_classes = num_classes - 1

        # Create model
        self.model = self._create_model()
        self._is_training = False

    def _create_model(self):
        """Create YOLO model"""

        # YOLO needs to know the number of classes at creation time
        # We can't just load pretrained and change nc - that freezes parameters
        # Instead, load architecture from YAML and optionally load pretrained weights

        if self.pretrained:
            # Start with pretrained model
            model = YOLO(f"{self.model_size}.pt")

            # For transfer learning with different number of classes:
            # We need to ensure all parameters are trainable
            # YOLO automatically handles this during training, but we need to explicitly unfreeze
            if hasattr(model, 'model'):
                for param in model.model.parameters():
                    param.requires_grad = True

                # Update number of classes in the model
                # This will be used when the detection head is rebuilt during first training step
                if hasattr(model.model, 'nc'):
                    model.model.nc = self.yolo_num_classes

                # Also update in the Detect layer if it exists
                for module in model.model.modules():
                    if module.__class__.__name__ == 'Detect':
                        module.nc = self.yolo_num_classes
        else:
            # Load architecture only (no pretrained weights)
            model = YOLO(f"{self.model_size}.yaml")
            # Set number of classes in config
            if hasattr(model, 'model') and hasattr(model.model, 'nc'):
                model.model.nc = self.yolo_num_classes

        return model

    def get_model(self) -> nn.Module:
        """Return the underlying YOLO model"""
        # YOLO's actual PyTorch model is in model.model
        return self.model.model if hasattr(self.model, 'model') else self.model

    def _convert_targets_to_yolo(self, images: List[torch.Tensor], targets: List[Dict]) -> Dict:
        """
        Convert our target format to YOLO format

        Our format:
            targets: List[dict] with 'boxes' (N,4) in [xmin,ymin,xmax,ymax], 'labels' (N,)

        YOLO format:
            Expects normalized [x_center, y_center, width, height] in range [0,1]
            Format: (batch_idx, class_id, x_center, y_center, width, height)

        Returns:
            Dict with YOLO-formatted data
        """
        batch_size = len(images)
        yolo_targets = []

        for batch_idx, (img, target) in enumerate(zip(images, targets)):
            # Get image dimensions
            _, h, w = img.shape

            boxes = target['boxes']  # (N, 4) [xmin, ymin, xmax, ymax]
            labels = target['labels']  # (N,)

            for box, label in zip(boxes, labels):
                # Skip background class (label 0)
                if label == 0:
                    continue

                # Convert to YOLO format
                xmin, ymin, xmax, ymax = box.tolist()

                # Convert to center format and normalize
                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h

                # YOLO uses 0-indexed classes (no background)
                yolo_label = label - 1

                yolo_targets.append([batch_idx, yolo_label, x_center, y_center, width, height])

        if len(yolo_targets) > 0:
            yolo_targets = torch.tensor(yolo_targets)
        else:
            yolo_targets = torch.zeros((0, 6))

        return {'targets': yolo_targets}

    def _convert_yolo_to_our_format(self, yolo_results, image_shapes: List[tuple]) -> List[Dict]:
        """
        Convert YOLO Results objects to our standard format

        YOLO format:
            Results object with .boxes attribute containing:
            - xyxy: boxes in [xmin, ymin, xmax, ymax]
            - conf: confidence scores
            - cls: class predictions

        Our format:
            List[dict] with 'boxes', 'scores', 'labels' tensors

        Args:
            yolo_results: YOLO Results objects (one per image)
            image_shapes: Original image shapes [(H,W), ...]

        Returns:
            List of prediction dicts matching our format
        """
        predictions = []

        for result in yolo_results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Extract predictions
                boxes = result.boxes.xyxy.cpu()  # [N, 4] in [xmin, ymin, xmax, ymax]
                scores = result.boxes.conf.cpu()  # [N]
                labels = result.boxes.cls.cpu()  # [N]

                # Convert labels: YOLO uses 0-indexed, we use 1-indexed (0=background)
                labels = labels + 1

                predictions.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels.long()
                })
            else:
                # No detections
                predictions.append({
                    'boxes': torch.zeros((0, 4)),
                    'scores': torch.zeros(0),
                    'labels': torch.zeros(0, dtype=torch.long)
                })

        return predictions

    def __call__(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None) -> Union[Dict, List[Dict]]:
        """Make the detector callable (nn.Module compatible)"""
        return self.forward(images, targets)

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None) -> Union[Dict, List[Dict]]:
        """
        Forward pass

        Args:
            images: List of image tensors [Tensor(3,H,W), ...]
            targets: Optional list of target dicts for training

        Returns:
            Training mode: Dict with loss values
            Inference mode: List of prediction dicts
        """
        if targets is not None:
            # Training mode - compute losses using YOLO's internal loss function
            # Stack images into batch tensor
            batch_tensor = torch.stack(images)  # [B, C, H, W]

            # Convert targets to YOLO format
            # YOLO expects targets as tensor: [batch_idx, class, x_center, y_center, width, height]
            yolo_targets = self._convert_targets_to_yolo(images, targets)['targets']

            # Move targets to same device as model
            if batch_tensor.is_cuda:
                yolo_targets = yolo_targets.to(batch_tensor.device)

            # YOLO's model.model is the actual PyTorch model
            # Forward pass through the model
            preds = self.model.model(batch_tensor)

            # Compute loss using YOLO's loss function
            # YOLO's loss is computed by the model's criterion
            if hasattr(self.model, 'criterion'):
                # Use YOLO's built-in loss
                loss, loss_items = self.model.criterion(preds, yolo_targets)
            elif hasattr(self.model.model, 'loss'):
                # Alternative: use model's loss method
                loss_dict = self.model.model.loss(preds, yolo_targets)
                loss = loss_dict['loss']
            else:
                # Fallback: manual loss computation (simplified)
                # This is not ideal but allows training to proceed
                # YOLO loss = box loss + objectness loss + classification loss
                # For now, return a dummy loss to indicate training mode works
                loss = torch.tensor(0.0, device=batch_tensor.device, requires_grad=True)
                print("Warning: YOLO loss computation not fully implemented. Using dummy loss.")

            # Return loss in format expected by training loop
            return {'loss': loss}
        else:
            # Inference mode
            # Convert images to YOLO format (batch of numpy arrays or tensors)
            image_shapes = [(img.shape[1], img.shape[2]) for img in images]

            # YOLO expects images in [0, 255] range or normalized [0, 1]
            # Our images are normalized, so scale to [0, 255] if needed
            if images[0].max() <= 1.0:
                images_for_yolo = [img * 255.0 for img in images]
            else:
                images_for_yolo = images

            # Stack into batch tensor
            batch_tensor = torch.stack(images_for_yolo)

            # Run YOLO inference
            # Note: YOLO's __call__ handles batching automatically
            results = self.model(
                batch_tensor,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det,
                verbose=False
            )

            # Convert to our format
            predictions = self._convert_yolo_to_our_format(results, image_shapes)

            return predictions

    def train_yolo(self, data_yaml: str, epochs: int, batch_size: int, img_size: int,
                   project: str, name: str, **kwargs) -> Dict:
        """
        Train YOLO model using its native training loop

        Args:
            data_yaml: Path to YOLO data.yaml config file
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size
            project: Project directory for saving results
            name: Experiment name
            **kwargs: Additional YOLO training arguments

        Returns:
            Dict with training results

        Note:
            This method uses YOLO's built-in training loop.
            For our unified training loop, we would need to extract
            YOLO's loss computation and integrate it.
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=project,
            name=name,
            **kwargs
        )

        return results

    def set_train_mode(self):
        """Set model to training mode"""
        self._is_training = True
        if hasattr(self.model, 'model'):
            self.model.model.train()

    def set_eval_mode(self):
        """Set model to evaluation mode"""
        self._is_training = False
        if hasattr(self.model, 'model'):
            self.model.model.eval()

    def train(self, mode=True):
        """Set model to training mode (nn.Module compatible)"""
        if mode:
            self.set_train_mode()
        else:
            self.set_eval_mode()
        return self

    def eval(self):
        """Set model to evaluation mode (nn.Module compatible)"""
        return self.train(False)

    def parameters(self):
        """Get model parameters (nn.Module compatible)"""
        if hasattr(self.model, 'model'):
            return self.model.model.parameters()
        return iter([])

    def state_dict(self):
        """Get model state dict (nn.Module compatible)"""
        if hasattr(self.model, 'model'):
            return self.model.model.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        """Load model state dict (nn.Module compatible)"""
        if hasattr(self.model, 'model'):
            self.model.model.load_state_dict(state_dict)

    def to(self, device: torch.device):
        """Move model to device"""
        if hasattr(self.model, 'model'):
            self.model.model = self.model.model.to(device)
        self.device = device
        return self

    def get_optimizer_params(self) -> List[nn.Parameter]:
        """Get trainable parameters"""
        if hasattr(self.model, 'model'):
            return [p for p in self.model.model.parameters() if p.requires_grad]
        return []

    def load_checkpoint(self, checkpoint_path: str, device: torch.device = None) -> Dict:
        """
        Load YOLO checkpoint

        YOLO checkpoints have different format, so we need special handling
        """
        if device is None:
            device = torch.device('cpu')

        # Check if this is a YOLO checkpoint or our standard checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if 'model_name' in checkpoint and checkpoint['model_name'] == 'yolo':
            # Our standard checkpoint format
            self.model = YOLO(checkpoint_path)  # Load YOLO model
            return checkpoint
        else:
            # Assume it's a native YOLO checkpoint
            self.model = YOLO(checkpoint_path)
            return {'model_name': 'yolo', 'checkpoint_path': checkpoint_path}

    def save_checkpoint(self, checkpoint_path: str, epoch: int, optimizer_state: Dict = None, **metadata) -> None:
        """
        Save YOLO checkpoint

        Note: YOLO models are saved in their native format
        """
        # Save using YOLO's format
        if hasattr(self.model, 'save'):
            self.model.save(checkpoint_path)

        # Also save our metadata separately
        metadata_dict = {
            'epoch': epoch,
            'model_name': self.name,
            'model_config': self.config,
            'num_classes': self.num_classes,
            'yolo_checkpoint': checkpoint_path,
            **metadata
        }

        metadata_path = str(Path(checkpoint_path).with_suffix('.meta.pt'))
        torch.save(metadata_dict, metadata_path)

    @property
    def name(self) -> str:
        """Return model name"""
        return "yolo"

    @property
    def backbone_name(self) -> str:
        """Return backbone name"""
        return self.model_size

    def get_config_summary(self) -> str:
        """Get human-readable config summary"""
        return (
            f"YOLOv8 Configuration:\n"
            f"  Model size: {self.model_size}\n"
            f"  Image size: {self.img_size}\n"
            f"  Confidence threshold: {self.conf_thresh}\n"
            f"  IoU threshold: {self.iou_thresh}\n"
            f"  Max detections: {self.max_det}\n"
            f"  Pretrained: {self.pretrained}\n"
            f"  Number of classes: {self.yolo_num_classes} (YOLO format)\n"
        )
