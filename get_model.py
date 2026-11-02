import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import timm
from typing import Tuple, Optional, Dict, Any
import yaml


def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置获取模型

    Args:
        config: 模型配置字典

    Returns:
        初始化好的模型实例
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
    从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# 默认配置
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
    模型管理器，用于统一管理模型的创建、保存和加载
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_model(self) -> nn.Module:
        """创建模型"""
        self.model = get_model(self.config)
        self.model = self.model.to(self.device)
        return self.model

    def save_checkpoint(self, filepath: str, optimizer=None, scheduler=None, epoch=None, metrics=None):
        """保存检查点"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用 create_model()")

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
        print(f"检查点已保存: {filepath}")

    def load_checkpoint(self, filepath: str, optimizer=None, scheduler=None):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # 更新配置（如果有）
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])

        # 创建模型
        if self.model is None:
            self.model = self.create_model()

        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器和调度器状态（如果提供）
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})

        print(f"检查点已加载: {filepath} (epoch: {epoch})")
        return epoch, metrics


# ResNet50编码器
class ResNet50Encoder(nn.Module):
    """
    ResNet50编码器，用于提取局部特征
    """

    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)

        # 移除最后的分类层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 投影层，将2048维特征投影到384维
        self.projection = nn.Linear(2048, 384)

    def forward(self, x):
        # 提取特征图
        feature_map = self.resnet(x)

        # 全局平均池化
        features = self.avgpool(feature_map)
        features = features.view(features.size(0), -1)

        # 投影到目标维度
        features = self.projection(features)

        return features, feature_map


# DINOv2编码器
class DINOv2Encoder(nn.Module):
    """
    DINOv2编码器，用于提取全局特征
    """

    def __init__(self, model_name='vit_small_patch14_dinov2'):
        super(DINOv2Encoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.out_channels = self.model.num_features

    def forward(self, x):
        features = self.model(x)
        return features


# 门控网络
class GatingNetwork(nn.Module):
    """
    门控网络，用于动态融合两个编码器的特征
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
            nn.Linear(hidden_dim // 2, output_dim * 2),  # 输出两个权重
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.network(x)
        batch_size = x.shape[0]
        a_res = weights[:, :weights.shape[1]//2]
        a_dino = weights[:, weights.shape[1]//2:]
        return a_res, a_dino


# 更新模型定义以支持更多配置选项
class EGJAFoundationModel(nn.Module):
    """
    用于EGJA分期诊断的AI基础模型
    """

    def __init__(self,
                 num_classes: int = 3,
                 dropout: float = 0.1,
                 dino_model_name: str = "vit_small_patch14_dinov2",
                 use_pretrained: bool = True):
        super(EGJAFoundationModel, self).__init__()

        # 两个编码器
        self.resnet_encoder = ResNet50Encoder(pretrained=use_pretrained)
        self.dinov2_encoder = DINOv2Encoder(model_name=dino_model_name)

        # 获取特征维度
        resnet_dim = 384  # ResNet投影后的维度
        dino_dim = self.dinov2_encoder.out_channels

        # 确保两个编码器的输出维度一致
        if resnet_dim != dino_dim:
            print(f"警告: ResNet维度({resnet_dim})与DINOv2维度({dino_dim})不匹配，将进行投影")
            self.resnet_projection = nn.Linear(resnet_dim, dino_dim)
            resnet_dim = dino_dim
        else:
            self.resnet_projection = nn.Identity()

        # 门控网络
        self.gating_network = GatingNetwork(
            input_dim=resnet_dim * 2,  # 两个编码器的特征拼接
            hidden_dim=512,
            output_dim=dino_dim,
            dropout=dropout
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dino_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 存储配置
        self.num_classes = num_classes
        self.dropout = dropout

    def forward(self, x):
        batch_size = x.shape[0]

        # 通过ResNet50编码器提取局部特征
        resnet_features, resnet_feature_map = self.resnet_encoder(x)
        resnet_features = self.resnet_projection(resnet_features)

        # 通过DINOv2编码器提取全局特征
        dinov2_features = self.dinov2_encoder(x)

        # 拼接特征
        concatenated_features = torch.cat([resnet_features, dinov2_features], dim=1)

        # 通过门控网络获取权重
        a_res, a_dino = self.gating_network(concatenated_features)

        # 逐元素融合特征
        fused_features = a_res * resnet_features + a_dino * dinov2_features

        # 分类
        logits = self.classifier(fused_features)

        return {
            'logits': logits,
            'fused_features': fused_features,
            'resnet_features': resnet_features,
            'dinov2_features': dinov2_features,
            'gate_weights': (a_res, a_dino),
            'resnet_feature_map': resnet_feature_map
        }


# 使用示例
# def main():
#     # 方法1: 使用默认配置
#     config = DEFAULT_CONFIG
#     model = get_model(config)
#
#     # 方法2: 使用模型管理器
#     model_manager = ModelManager()
#     model = model_manager.create_model()
#
#     # 打印模型信息
#     print(f"模型类别数: {model.num_classes}")
#     print(f"Dropout率: {model.dropout}")
#
#     # 测试前向传播
#     batch_size = 2
#     input_size = 518
#     x = torch.randn(batch_size, 3, input_size, input_size)
#
#     with torch.no_grad():
#         outputs = model(x)
#
#     print(f"输入尺寸: {x.shape}")
#     print(f"输出logits尺寸: {outputs['logits'].shape}")
#     print(f"融合特征尺寸: {outputs['fused_features'].shape}")
#
#     # 保存检查点示例
#     # model_manager.save_checkpoint('checkpoint.pth', epoch=0)
#
#
# # 如果要将配置保存到文件
# def save_config(config: Dict[str, Any], filepath: str):
#     """保存配置到YAML文件"""
#     with open(filepath, 'w') as f:
#         yaml.dump(config, f, default_flow_style=False)
#
#
# if __name__ == "__main__":
#     main()