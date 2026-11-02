import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np


def plot_model_architecture():
    """绘制CAIDM模型整体架构图"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 颜色定义
    colors = {
        'input': '#FFE4E1',
        'resnet': '#E6E6FA',
        'dinov2': '#F0FFF0',
        'fusion': '#FFF0F5',
        'classifier': '#F5F5DC',
        'output': '#E0FFFF'
    }

    # 1. 输入层
    input_box = FancyBboxPatch((1, 6), 2, 0.8, boxstyle="round,pad=0.02",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 6.4, '输入图像\n(3×518×518)', ha='center', va='center', fontsize=10, fontweight='bold')

    # 2. 双编码器分支
    # ResNet50分支
    resnet_box = FancyBboxPatch((0.5, 4), 2, 1.5, boxstyle="round,pad=0.02",
                                facecolor=colors['resnet'], edgecolor='black', linewidth=2)
    ax.add_patch(resnet_box)
    ax.text(1.5, 4.75, 'ResNet50编码器\n(局部特征提取)', ha='center', va='center', fontsize=10, fontweight='bold')

    # ResNet内部结构
    resnet_layers = [
        ('Conv1', 1, 4.4),
        ('ResBlock1', 1, 4.2),
        ('ResBlock2', 1, 4.0),
        ('ResBlock3', 1, 3.8),
        ('ResBlock4', 1, 3.6),
        ('Projection', 1, 3.4)
    ]

    for i, (name, x, y) in enumerate(resnet_layers):
        ax.text(x, y, name, ha='center', va='center', fontsize=8)
        if i < len(resnet_layers) - 1:
            ax.plot([x, x], [y - 0.1, y - 0.2], 'k-', lw=1)

    # DINOv2分支
    dino_box = FancyBboxPatch((3, 4), 2, 1.5, boxstyle="round,pad=0.02",
                              facecolor=colors['dinov2'], edgecolor='black', linewidth=2)
    ax.add_patch(dino_box)
    ax.text(4, 4.75, 'DINOv2编码器\n(全局特征提取)', ha='center', va='center', fontsize=10, fontweight='bold')

    # DINOv2内部结构
    dino_layers = [
        ('Patch Embed', 4, 4.4),
        ('Pos Embed', 4, 4.2),
        ('Transformer\nBlocks×12', 4, 4.0),
        ('CLS Token', 4, 3.8),
        ('Feature\nExtraction', 4, 3.6)
    ]

    for i, (name, x, y) in enumerate(dino_layers):
        ax.text(x, y, name, ha='center', va='center', fontsize=8)
        if i < len(dino_layers) - 1:
            ax.plot([x, x], [y - 0.1, y - 0.2], 'k-', lw=1)

    # 3. 特征输出
    resnet_feat = FancyBboxPatch((0.8, 2.8), 0.8, 0.4, boxstyle="round,pad=0.02",
                                 facecolor=colors['resnet'], edgecolor='black', linewidth=1)
    ax.add_patch(resnet_feat)
    ax.text(1.2, 3.0, 'F_Res\n(1×384)', ha='center', va='center', fontsize=8)

    dino_feat = FancyBboxPatch((3.2, 2.8), 0.8, 0.4, boxstyle="round,pad=0.02",
                               facecolor=colors['dinov2'], edgecolor='black', linewidth=1)
    ax.add_patch(dino_feat)
    ax.text(3.6, 3.0, 'F_DINO\n(1×384)', ha='center', va='center', fontsize=8)

    # 4. 门控网络 (MoE)
    gate_box = FancyBboxPatch((1.8, 2.2), 1.4, 1.0, boxstyle="round,pad=0.02",
                              facecolor=colors['fusion'], edgecolor='black', linewidth=2)
    ax.add_patch(gate_box)
    ax.text(2.5, 2.7, '门控网络\n(MoE架构)', ha='center', va='center', fontsize=10, fontweight='bold')

    # 门控网络内部
    gate_layers = [
        ('Concatenate\n(1×768)', 2.5, 2.4),
        ('Conv1d×3', 2.5, 2.2),
        ('Softmax', 2.5, 2.0),
        ('A_Res + A_DINO', 2.5, 1.8)
    ]

    for i, (name, x, y) in enumerate(gate_layers):
        ax.text(x, y, name, ha='center', va='center', fontsize=7)
        if i < len(gate_layers) - 1:
            ax.plot([x, x], [y - 0.08, y - 0.16], 'k-', lw=1)

    # 5. 特征融合
    fusion_box = FancyBboxPatch((4.5, 2.2), 1.0, 0.6, boxstyle="round,pad=0.02",
                                facecolor=colors['fusion'], edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5.0, 2.5, '特征融合\n逐元素加权', ha='center', va='center', fontsize=9, fontweight='bold')

    # 6. 分类器
    classifier_box = FancyBboxPatch((6, 2.0), 1.5, 1.0, boxstyle="round,pad=0.02",
                                    facecolor=colors['classifier'], edgecolor='black', linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(6.75, 2.5, '分类器', ha='center', va='center', fontsize=10, fontweight='bold')

    # 分类器内部
    cls_layers = [
        ('FC(384→256)', 6.75, 2.3),
        ('ReLU', 6.75, 2.1),
        ('Dropout', 6.75, 1.9),
        ('FC(256→2)', 6.75, 1.7)
    ]

    for i, (name, x, y) in enumerate(cls_layers):
        ax.text(x, y, name, ha='center', va='center', fontsize=8)
        if i < len(cls_layers) - 1:
            ax.plot([x, x], [y - 0.08, y - 0.16], 'k-', lw=1)

    # 7. 输出层
    output_box = FancyBboxPatch((8, 2.0), 1.0, 0.6, boxstyle="round,pad=0.02",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8.5, 2.3, '输出\n[HGC, LGC]', ha='center', va='center', fontsize=10, fontweight='bold')

    # 8. 连接箭头
    # 输入到编码器
    ax.annotate('', xy=(2, 5.2), xytext=(2, 6), arrowprops=dict(arrowstyle='->', lw=2))
    ax.plot([1.5, 2.5], [5.2, 5.2], 'k-', lw=2)
    ax.annotate('', xy=(1.5, 5.2), xytext=(1.5, 5.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(2.5, 5.2), xytext=(2.5, 5.5), arrowprops=dict(arrowstyle='->', lw=2))

    # 编码器到特征
    ax.annotate('', xy=(1.2, 3.2), xytext=(1.2, 4), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(3.6, 3.2), xytext=(3.6, 4), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 特征到门控网络
    ax.annotate('', xy=(2.2, 2.8), xytext=(1.6, 2.8), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(2.8, 2.8), xytext=(3.4, 2.8), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 门控网络到融合
    ax.annotate('', xy=(4.5, 2.5), xytext=(3.2, 2.5), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 融合到分类器
    ax.annotate('', xy=(6, 2.5), xytext=(5.5, 2.5), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 分类器到输出
    ax.annotate('', xy=(8, 2.3), xytext=(7.5, 2.3), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 9. 添加说明文本
    info_text = (
        " Foundation Model 架构说明:\n"
        "• 输入: RGB endoscopic images (518×518)\n"
        "• ResNet50: 提取局部细节特征\n"
        "• DINOv2: 提取全局外观特征  \n"
        "• 门控网络(MoE): 自适应特征融合\n"
        "• 输出: HGC/LGC 二分类"
    )
    ax.text(0.5, 0.8, info_text, fontsize=11, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    # 10. 添加标题
    ax.text(5, 7.5, ' Foundation Model 架构图',
            ha='center', va='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('model_architecture.pdf', bbox_inches='tight')
    plt.show()


def plot_data_flow_diagram():
    """绘制数据流图"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 数据流节点
    nodes = [
        ('输入图像', 1, 5, '#FFE4E1'),
        ('数据预处理', 3, 5, '#F0FFF0'),
        ('ResNet50\n编码器', 5, 4, '#E6E6FA'),
        ('DINOv2\n编码器', 5, 3, '#F0FFF0'),
        ('特征拼接', 7, 3.5, '#FFF0F5'),
        ('门控网络\n(MoE)', 9, 3.5, '#F5F5DC'),
        ('分类器', 11, 3.5, '#E0FFFF'),
        ('预测输出', 13, 3.5, '#FFE4E1')
    ]

    # 绘制节点
    for i, (name, x, y, color) in enumerate(nodes):
        circle = plt.Circle((x, y), 0.4, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

        # 添加节点编号
        ax.text(x, y - 0.6, f'({i + 1})', ha='center', va='center', fontsize=8, fontweight='bold')

    # 数据流箭头
    flows = [
        (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 7), (7, 8)
    ]

    for start, end in flows:
        x1, y1 = nodes[start - 1][1], nodes[start - 1][2]
        x2, y2 = nodes[end - 1][1], nodes[end - 1][2]

        # 计算箭头方向
        dx = x2 - x1
        dy = y2 - y1

        ax.annotate('', xy=(x2 - 0.4 * dx / abs(dx), y2 - 0.4 * dy / abs(dy)),
                    xytext=(x1 + 0.4 * dx / abs(dx), y1 + 0.4 * dy / abs(dy)),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    # 添加数据维度说明
    dim_texts = [
        (2, 5.3, '3×518×518'),
        (4, 5.3, '归一化处理'),
        (5, 4.7, 'F_Res: 1×384'),
        (5, 2.7, 'F_DINO: 1×384'),
        (7, 2.7, '拼接: 1×768'),
        (9, 2.7, '权重学习'),
        (11, 2.7, 'FC Layers'),
        (13, 2.7, '[p_HGC, p_LGC]')
    ]

    for x, y, text in dim_texts:
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    ax.set_title('模型数据流图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('data_flow_diagram.pdf', bbox_inches='tight')
    plt.show()


# 运行绘图
if __name__ == "__main__":
    print("绘制模型架构图...")
    plot_model_architecture()

    print("绘制数据流图...")
    plot_data_flow_diagram()

    print("图表已保存为 PNG 和 PDF 格式")
