import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class Loss(nn.Module):
    """
    Custom loss function for BCA staging diagnosis.
    Supports multiple loss combinations and class weights.
    """

    def __init__(self,
                 class_weights: Optional[torch.Tensor] = None,
                 alpha: float = 1.0,  # Weight for cross-entropy loss
                 beta: float = 0.1,  # Weight for feature separation loss
                 gamma: float = 0.01,  # Weight for gating network regularization
                 loss_type: str = 'cross_entropy'):
        super(Loss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

        # Main classification loss
        if loss_type == 'cross_entropy':
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == 'focal':
            self.classification_loss = FocalLoss(weight=class_weights)
        elif loss_type == 'label_smooth':
            self.classification_loss = LabelSmoothingLoss(classes=3, smoothing=0.1)
        else:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss.

        Args:
            outputs: Model output dictionary, containing:
                - logits: classification logits [B, num_classes]
                - fused_features: fused features [B, feature_dim]
                - resnet_features: ResNet features [B, feature_dim]
                - dinov2_features: DINOv2 features [B, feature_dim]
                - gate_weights: gating weights tuple (a_res, a_dino)
            targets: ground truth labels [B]

        Returns:
            Total loss value
        """
        logits = outputs['logits']

        # Main classification loss
        classification_loss = self.classification_loss(logits, targets)

        total_loss = self.alpha * classification_loss

        # Feature separation loss (encourages features of different classes to separate in the feature space)
        if self.beta > 0:
            feature_separation_loss = self.compute_feature_separation_loss(
                outputs['fused_features'], targets
            )
            total_loss += self.beta * feature_separation_loss

        # Gating network regularization (encourages balanced usage of both encoders)
        if self.gamma > 0 and 'gate_weights' in outputs:
            gate_regularization_loss = self.compute_gate_regularization_loss(
                outputs['gate_weights']
            )
            total_loss += self.gamma * gate_regularization_loss

        return total_loss

    def compute_feature_separation_loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute feature separation loss, encouraging features of different classes to separate better in the feature space.
        """
        # Add safety check: ensure targets is a valid integer tensor
        if len(targets) == 0:
            return torch.tensor(0.0, device=features.device)

        # Ensure targets is of integer type
        if targets.dtype != torch.long:
            targets = targets.long()

        # Check if targets contains invalid values (negative or out-of-range values)
        if (targets < 0).any():
            # If negative values exist, return 0 loss
            return torch.tensor(0.0, device=features.device)

        unique_classes = torch.unique(targets)
        if len(unique_classes) < 2:
            return torch.tensor(0.0, device=features.device)

        # Compute feature centers for each class
        class_centers = []
        for cls in unique_classes:
            cls_features = features[targets == cls]
            if len(cls_features) > 0:
                class_centers.append(cls_features.mean(dim=0))

        if len(class_centers) < 2:
            return torch.tensor(0.0, device=features.device)

        class_centers = torch.stack(class_centers)

        # Compute distances between class centers
        n_classes = len(class_centers)
        center_distances = []

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                distance = F.pairwise_distance(
                    class_centers[i].unsqueeze(0),
                    class_centers[j].unsqueeze(0),
                    p=2
                )
                center_distances.append(distance)

        if not center_distances:
            return torch.tensor(0.0, device=features.device)

        center_distances = torch.stack(center_distances)

        # We want to minimize the inverse of distances between similar classes (encourage separation)
        # Use negative mean distance, because we want to maximize the distance
        separation_loss = -torch.mean(center_distances)

        return separation_loss

    def compute_gate_regularization_loss(self, gate_weights: tuple) -> torch.Tensor:
        """
        Compute gating network regularization loss, encouraging balanced usage of both encoders.
        """
        a_res, a_dino = gate_weights

        # Compute entropy of both gating weights to encourage diversity
        entropy_res = -torch.mean(a_res * torch.log(a_res + 1e-8))
        entropy_dino = -torch.mean(a_dino * torch.log(a_dino + 1e-8))

        # Encourage usage ratios of the two encoders to be close
        usage_ratio = torch.mean(a_res) / (torch.mean(a_dino) + 1e-8)
        balance_loss = torch.abs(usage_ratio - 1.0)

        # Total regularization loss is a combination of entropy and balance loss
        regularization_loss = entropy_res + entropy_dino + balance_loss

        return regularization_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance issues.
    """

    def __init__(self, alpha=None, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha = torch.tensor(self.alpha, device=inputs.device)
            else:
                alpha = self.alpha[targets]
            focal_loss = alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss to prevent the model from becoming overconfident.
    """

    def __init__(self, classes=3, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def calculate_class_weights(dataset, num_classes=3):
    """
    Calculate class weights to handle class imbalance.

    Args:
        dataset: Dataset containing labels
        num_classes: Number of classes

    Returns:
        Class weights tensor
    """
    # Count the number of samples in each class
    class_counts = torch.zeros(num_classes)

    for _, label in dataset:
        class_counts[label] += 1

    # Calculate weights (fewer samples means higher weight)
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)

    return class_weights


# Loss function factory function
def get_loss_function(config: Dict[str, Any], dataset=None):
    """
    Get loss function based on configuration.

    Args:
        config: Configuration dictionary
        dataset: Dataset for calculating class weights (optional)

    Returns:
        Loss function instance
    """
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'cross_entropy')

    # Calculate class weights (if dataset is provided)
    class_weights = None
    if dataset is not None and loss_config.get('use_class_weights', False):
        class_weights = calculate_class_weights(dataset)
        print(f"Calculated class weights: {class_weights}")

    if loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('alpha', None),
            gamma=loss_config.get('gamma', 2.0),
            weight=class_weights
        )
    elif loss_type == 'label_smooth':
        return LabelSmoothingLoss(
            classes=config['model'].get('num_classes', 3),
            smoothing=loss_config.get('smoothing', 0.1)
        )
    else:
        return Loss(
            class_weights=class_weights,
            alpha=loss_config.get('alpha', 1.0),
            beta=loss_config.get('beta', 0.1),
            gamma=loss_config.get('gamma', 0.01),
            loss_type=loss_type
        )


# Update default configuration to include loss function settings
DEFAULT_CONFIG = {
    'model': {
        'num_classes': 3,
        'dropout': 0.1,
        'dino_model_name': 'vit_small_patch14_dinov2',
        'use_pretrained': True
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 40,
        'weight_decay': 1e-4
    },
    'loss': {
        'type': 'cross_entropy',  # 'cross_entropy', 'focal', 'label_smooth'
        'alpha': 1.0,  # Main loss weight
        'beta': 0.1,  # Feature separation loss weight
        'gamma': 0.01,  # Gating regularization weight
        'use_class_weights': True
    },
    'data': {
        'image_size': 448,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    }
}
