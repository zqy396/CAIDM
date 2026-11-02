import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import timm
from typing import Tuple, Optional, Dict, Any
import yaml


def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Get model based on configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized model instance
    """
    model_config = config.get('model', {})

    model = EGJAFoundationModel(
        num_classes=model_config.get('num_classes', 3),
        dropout=model_config.get('dropout', 0.1),
        dino_model_name=model_config.get('dino_model_name', 'vit_small_patch14_dinov2'),
        use_pretrained=model_config.get('use_pretrained', True)
    )

    return model


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Configuration file path

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Default configuration
DEFAULT_CONFIG = {
    'model': {
        'num_classes': 4,
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
    'data': {
        'image_size': 448,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    }
}


class ModelManager:
    """
    Model manager for unified management of model creation, saving, and loading.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_model(self) -> nn.Module:
        """Create model"""
        self.model = get_model(self.config)
        self.model = self.model.to(self.device)
        return self.model

    def save_checkpoint(self, filepath: str, optimizer=None, scheduler=None, epoch=None, metrics=None):
        """Save checkpoint"""
        if self.model is None:
            raise ValueError("Model not initialized, please call create_model() first")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'metrics': metrics or {}
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str, optimizer=None, scheduler=None):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Update config (if any)
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])

        # Create model
        if self.model is None:
            self.model = self.create_model()

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and scheduler states (if provided)
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})

        print(f"Checkpoint loaded: {filepath} (epoch: {epoch})")
        return epoch, metrics


# ResNet50 Encoder
class ResNet50Encoder(nn.Module):
    """
    ResNet50 encoder for extracting local features.
    """

    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection layer, projecting 2048-dim features to 384-dim
        self.projection = nn.Linear(2048, 384)

    def forward(self, x):
        # Extract feature map
        feature_map = self.resnet(x)

        # Global average pooling
        features = self.avgpool(feature_map)
        features = features.view(features.size(0), -1)

        # Project to target dimension
        features = self.projection(features)

        return features, feature_map


# DINOv2 Encoder
class DINOv2Encoder(nn.Module):
    """
    DINOv2 encoder for extracting global features.
    """

    def __init__(self, model_name='vit_small_patch14_dinov2'):
        super(DINOv2Encoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.out_channels = self.model.num_features

    def forward(self, x):
        features = self.model(x)
        return features


# Gating Network
class GatingNetwork(nn.Module):
    """
    Gating network for dynamically fusing features from two encoders.
    """

    def __init__(self, input_dim, hidden_dim=512, output_dim=384, dropout=0.1):
        super(GatingNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim * 2),  # Output two weights
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.network(x)
        batch_size = x.shape[0]
        a_res = weights[:, :weights.shape[1]//2]
        a_dino = weights[:, weights.shape[1]//2:]
        return a_res, a_dino


# Update model definition to support more configuration options
class EGJAFoundationModel(nn.Module):
    """
    AI Foundation Model for EGJA staging diagnosis.
    """

    def __init__(self,
                 num_classes: int = 3,
                 dropout: float = 0.1,
                 dino_model_name: str = "vit_small_patch14_dinov2",
                 use_pretrained: bool = True):
        super(EGJAFoundationModel, self).__init__()

        # Two encoders
        self.resnet_encoder = ResNet50Encoder(pretrained=use_pretrained)
        self.dinov2_encoder = DINOv2Encoder(model_name=dino_model_name)

        # Get feature dimensions
        resnet_dim = 384  # Dimension after ResNet projection
        dino_dim = self.dinov2_encoder.out_channels

        # Ensure the output dimensions of the two encoders match
        if resnet_dim != dino_dim:
            print(f"Warning: ResNet dimension ({resnet_dim}) does not match DINOv2 dimension ({dino_dim}), projecting...")
            self.resnet_projection = nn.Linear(resnet_dim, dino_dim)
            resnet_dim = dino_dim
        else:
            self.resnet_projection = nn.Identity()

        # Gating network
        self.gating_network = GatingNetwork(
            input_dim=resnet_dim * 2,  # Concatenation of features from both encoders
            hidden_dim=512,
            output_dim=dino_dim,
            dropout=dropout
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dino_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Store config
        self.num_classes = num_classes
        self.dropout = dropout

    def forward(self, x):
        batch_size = x.shape[0]

        # Extract local features via ResNet50 encoder
        resnet_features, resnet_feature_map = self.resnet_encoder(x)
        resnet_features = self.resnet_projection(resnet_features)

        # Extract global features via DINOv2 encoder
        dinov2_features = self.dinov2_encoder(x)

        # Concatenate features
        concatenated_features = torch.cat([resnet_features, dinov2_features], dim=1)

        # Get weights via gating network
        a_res, a_dino = self.gating_network(concatenated_features)

        # Element-wise feature fusion
        fused_features = a_res * resnet_features + a_dino * dinov2_features

        # Classification
        logits = self.classifier(fused_features)

        return {
            'logits': logits,
            'fused_features': fused_features,
            'resnet_features': resnet_features,
            'dinov2_features': dinov2_features,
            'gate_weights': (a_res, a_dino),
            'resnet_feature_map': resnet_feature_map
        }


# Usage Example
# def main():
#     # Method 1: Use default config
#     config = DEFAULT_CONFIG
#     model = get_model(config)
#
#     # Method 2: Use model manager
#     model_manager = ModelManager()
#     model = model_manager.create_model()
#
#     # Print model info
#     print(f"Model num_classes: {model.num_classes}")
#     print(f"Dropout rate: {model.dropout}")
#
#     # Test forward pass
#     batch_size = 2
#     input_size = 518
#     x = torch.randn(batch_size, 3, input_size, input_size)
#
#     with torch.no_grad():
#         outputs = model(x)
#
#     print(f"Input shape: {x.shape}")
#     print(f"Output logits shape: {outputs['logits'].shape}")
#     print(f"Fused features shape: {outputs['fused_features'].shape}")
#
#     # Checkpoint saving example
#     # model_manager.save_checkpoint('checkpoint.pth', epoch=0)
#
#
# # If you want to save config to file
# def save_config(config: Dict[str, Any], filepath: str):
#     """Save config to YAML file"""
#     with open(filepath, 'w') as f:
#         yaml.dump(config, f, default_flow_style=False)
#
#
# if __name__ == "__main__":
#     main()
