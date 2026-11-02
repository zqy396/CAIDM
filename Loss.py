import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class EGJALoss(nn.Module):
    """
    用于EGJA分期诊断的定制损失函数
    支持多种损失组合和类别权重
    """

    def __init__(self,
                 class_weights: Optional[torch.Tensor] = None,
                 alpha: float = 1.0,  # 交叉熵损失的权重
                 beta: float = 0.1,  # 特征分离损失的权重
                 gamma: float = 0.01,  # 门控网络正则化的权重
                 loss_type: str = 'cross_entropy'):
        super(EGJALoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

        # 主分类损失
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
        计算总损失

        Args:
            outputs: 模型输出字典，包含：
                - logits: 分类logits [B, num_classes]
                - fused_features: 融合特征 [B, feature_dim]
                - resnet_features: ResNet特征 [B, feature_dim]
                - dinov2_features: DINOv2特征 [B, feature_dim]
                - gate_weights: 门控权重元组 (a_res, a_dino)
            targets: 真实标签 [B]

        Returns:
            总损失值
        """
        logits = outputs['logits']

        # 主分类损失
        classification_loss = self.classification_loss(logits, targets)

        total_loss = self.alpha * classification_loss

        # 特征分离损失（鼓励不同类别的特征在特征空间中分离）
        if self.beta > 0:
            feature_separation_loss = self.compute_feature_separation_loss(
                outputs['fused_features'], targets
            )
            total_loss += self.beta * feature_separation_loss

        # 门控网络正则化（鼓励平衡使用两个编码器）
        if self.gamma > 0 and 'gate_weights' in outputs:
            gate_regularization_loss = self.compute_gate_regularization_loss(
                outputs['gate_weights']
            )
            total_loss += self.gamma * gate_regularization_loss

        return total_loss

    def compute_feature_separation_loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算特征分离损失，鼓励不同类别的特征在特征空间中更好分离
        """
        # 添加安全检查：确保targets是有效的整数张量
        if len(targets) == 0:
            return torch.tensor(0.0, device=features.device)

        # 确保targets是整数类型
        if targets.dtype != torch.long:
            targets = targets.long()

        # 检查targets中是否有无效值（负数或超出范围的值）
        if (targets < 0).any():
            # 如果存在负值，返回0损失
            return torch.tensor(0.0, device=features.device)

        unique_classes = torch.unique(targets)
        if len(unique_classes) < 2:
            return torch.tensor(0.0, device=features.device)

        # 计算每个类别的特征中心
        class_centers = []
        for cls in unique_classes:
            cls_features = features[targets == cls]
            if len(cls_features) > 0:
                class_centers.append(cls_features.mean(dim=0))

        if len(class_centers) < 2:
            return torch.tensor(0.0, device=features.device)

        class_centers = torch.stack(class_centers)

        # 计算类别中心之间的距离
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

        # 我们希望最小化相似类别间的距离的倒数（鼓励分离）
        # 使用负的平均距离，因为我们要最大化距离
        separation_loss = -torch.mean(center_distances)

        return separation_loss

    def compute_gate_regularization_loss(self, gate_weights: tuple) -> torch.Tensor:
        """
        计算门控网络正则化损失，鼓励平衡使用两个编码器
        """
        a_res, a_dino = gate_weights

        # 计算两个门控权重的熵，鼓励多样性
        entropy_res = -torch.mean(a_res * torch.log(a_res + 1e-8))
        entropy_dino = -torch.mean(a_dino * torch.log(a_dino + 1e-8))

        # 鼓励两个编码器的使用比例接近
        usage_ratio = torch.mean(a_res) / (torch.mean(a_dino) + 1e-8)
        balance_loss = torch.abs(usage_ratio - 1.0)

        # 总的正则化损失是熵和平衡损失的组合
        regularization_loss = entropy_res + entropy_dino + balance_loss

        return regularization_loss


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡问题
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
    标签平滑损失，防止模型过度自信
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
    计算类别权重以处理类别不平衡

    Args:
        dataset: 包含标签的数据集
        num_classes: 类别数量

    Returns:
        类别权重张量
    """
    # 统计每个类别的样本数
    class_counts = torch.zeros(num_classes)

    for _, label in dataset:
        class_counts[label] += 1

    # 计算权重（样本数越少，权重越高）
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)

    return class_weights


# 损失函数工厂函数
def get_loss_function(config: Dict[str, Any], dataset=None):
    """
    根据配置获取损失函数

    Args:
        config: 配置字典
        dataset: 用于计算类别权重的数据集（可选）

    Returns:
        损失函数实例
    """
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'cross_entropy')

    # 计算类别权重（如果提供数据集）
    class_weights = None
    if dataset is not None and loss_config.get('use_class_weights', False):
        class_weights = calculate_class_weights(dataset)
        print(f"计算得到的类别权重: {class_weights}")

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
        return EGJALoss(
            class_weights=class_weights,
            alpha=loss_config.get('alpha', 1.0),
            beta=loss_config.get('beta', 0.1),
            gamma=loss_config.get('gamma', 0.01),
            loss_type=loss_type
        )


# 更新默认配置以包含损失函数设置
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
        'alpha': 1.0,  # 主损失权重
        'beta': 0.1,  # 特征分离损失权重
        'gamma': 0.01,  # 门控正则化权重
        'use_class_weights': True
    },
    'data': {
        'image_size': 448,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225]
    }
}

