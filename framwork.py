import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np


def plot_model_architecture():
    """Plot the overall architecture of the CAIDM model."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Color definitions
    colors = {
        'input': '#FFE4E1',
        'resnet': '#E6E6FA',
        'dinov2': '#F0FFF0',
        'fusion': '#FFF0F5',
        'classifier': '#F5F5DC',
        'output': '#E0FFFF'
    }

    # 1. Input Layer
    input_box = FancyBboxPatch((1, 6), 2, 0.8, boxstyle="round,pad=0.02",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 6.4, 'Input Image\n(3×518×518)', ha='center', va='center', fontsize=10, fontweight='bold')

    # 2. Dual-Encoder Branches
    # ResNet50 Branch
    resnet_box = FancyBboxPatch((0.5, 4), 2, 1.5, boxstyle="round,pad=0.02",
                                facecolor=colors['resnet'], edgecolor='black', linewidth=2)
    ax.add_patch(resnet_box)
    ax.text(1.5, 4.75, 'ResNet50 Encoder\n(Local Feature Extraction)', ha='center', va='center', fontsize=10,
            fontweight='bold')

    # ResNet Internal Structure
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

    # DINOv2 Branch
    dino_box = FancyBboxPatch((3, 4), 2, 1.5, boxstyle="round,pad=0.02",
                              facecolor=colors['dinov2'], edgecolor='black', linewidth=2)
    ax.add_patch(dino_box)
    ax.text(4, 4.75, 'DINOv2 Encoder\n(Global Feature Extraction)', ha='center', va='center', fontsize=10,
            fontweight='bold')

    # DINOv2 Internal Structure
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

    # 3. Feature Output
    resnet_feat = FancyBboxPatch((0.8, 2.8), 0.8, 0.4, boxstyle="round,pad=0.02",
                                 facecolor=colors['resnet'], edgecolor='black', linewidth=1)
    ax.add_patch(resnet_feat)
    ax.text(1.2, 3.0, 'F_Res\n(1×384)', ha='center', va='center', fontsize=8)

    dino_feat = FancyBboxPatch((3.2, 2.8), 0.8, 0.4, boxstyle="round,pad=0.02",
                               facecolor=colors['dinov2'], edgecolor='black', linewidth=1)
    ax.add_patch(dino_feat)
    ax.text(3.6, 3.0, 'F_DINO\n(1×384)', ha='center', va='center', fontsize=8)

    # 4. Gating Network (MoE)
    gate_box = FancyBboxPatch((1.8, 2.2), 1.4, 1.0, boxstyle="round,pad=0.02",
                              facecolor=colors['fusion'], edgecolor='black', linewidth=2)
    ax.add_patch(gate_box)
    ax.text(2.5, 2.7, 'Gating Network\n(MoE Architecture)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Gating Network Internal Structure
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

    # 5. Feature Fusion
    fusion_box = FancyBboxPatch((4.5, 2.2), 1.0, 0.6, boxstyle="round,pad=0.02",
                                facecolor=colors['fusion'], edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5.0, 2.5, 'Feature Fusion\nElement-wise', ha='center', va='center', fontsize=9, fontweight='bold')

    # 6. Classifier
    classifier_box = FancyBboxPatch((6, 2.0), 1.5, 1.0, boxstyle="round,pad=0.02",
                                    facecolor=colors['classifier'], edgecolor='black', linewidth=2)
    ax.add_patch(classifier_box)
    ax.text(6.75, 2.5, 'Classifier', ha='center', va='center', fontsize=10, fontweight='bold')

    # Classifier Internal Structure
    cls_layers = [
        ('FC(384→256)', 6.75, 2.3),
        ('ReLU', 6.75, 2.1),
        ('Dropout', 6.75, 1.9),
        ('FC(256→3)', 6.75, 1.7)  # Updated to 3 classes
    ]

    for i, (name, x, y) in enumerate(cls_layers):
        ax.text(x, y, name, ha='center', va='center', fontsize=8)
        if i < len(cls_layers) - 1:
            ax.plot([x, x], [y - 0.08, y - 0.16], 'k-', lw=1)

    # 7. Output Layer
    output_box = FancyBboxPatch((8, 2.0), 1.2, 0.6, boxstyle="round,pad=0.02",
                                facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8.6, 2.3, 'Output\n[HGC, LGC, MIBC]', ha='center', va='center', fontsize=9, fontweight='bold')

    # 8. Connection Arrows
    # Input to Encoders
    ax.annotate('', xy=(2, 5.2), xytext=(2, 6), arrowprops=dict(arrowstyle='->', lw=2))
    ax.plot([1.5, 2.5], [5.2, 5.2], 'k-', lw=2)
    ax.annotate('', xy=(1.5, 5.2), xytext=(1.5, 5.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(2.5, 5.2), xytext=(2.5, 5.5), arrowprops=dict(arrowstyle='->', lw=2))

    # Encoders to Features
    ax.annotate('', xy=(1.2, 3.2), xytext=(1.2, 4), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(3.6, 3.2), xytext=(3.6, 4), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Features to Gating Network
    ax.annotate('', xy=(2.2, 2.8), xytext=(1.6, 2.8), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(2.8, 2.8), xytext=(3.4, 2.8), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Gating Network to Fusion
    ax.annotate('', xy=(4.5, 2.5), xytext=(3.2, 2.5), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Fusion to Classifier
    ax.annotate('', xy=(6, 2.5), xytext=(5.5, 2.5), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Classifier to Output
    ax.annotate('', xy=(8, 2.3), xytext=(7.5, 2.3), arrowprops=dict(arrowstyle='->', lw=1.5))

    # 9. Additional Notes
    info_text = (
        " Foundation Model Architecture Description:\n"
        "• Input: RGB endoscopic images (518×518)\n"
        "• ResNet50: Extracts local detail features\n"
        "• DINOv2: Extracts global appearance features\n"
        "• Gating Network (MoE): Adaptive feature fusion\n"
        "• Output: HGC / LGC / MIBC (3-class classification)"
    )
    ax.text(0.5, 0.8, info_text, fontsize=11, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    # 10. Title
    ax.text(5, 7.5, ' Foundation Model Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('model_architecture.pdf', bbox_inches='tight')
    plt.show()


def plot_data_flow_diagram():
    """Plot the data flow diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Data Flow Nodes
    nodes = [
        ('Input Image', 1, 5, '#FFE4E1'),
        ('Data\nPreprocessing', 3, 5, '#F0FFF0'),
        ('ResNet50\nEncoder', 5, 4, '#E6E6FA'),
        ('DINOv2\nEncoder', 5, 3, '#F0FFF0'),
        ('Feature\nConcatenation', 7, 3.5, '#FFF0F5'),
        ('Gating Network\n(MoE)', 9, 3.5, '#F5F5DC'),
        ('Classifier', 11, 3.5, '#E0FFFF'),
        ('Prediction\nOutput', 13, 3.5, '#FFE4E1')
    ]

    # Draw nodes
    for i, (name, x, y, color) in enumerate(nodes):
        circle = plt.Circle((x, y), 0.4, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

        # Add node number
        ax.text(x, y - 0.6, f'({i + 1})', ha='center', va='center', fontsize=8, fontweight='bold')

    # Data flow arrows
    flows = [
        (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 7), (7, 8)
    ]

    for start, end in flows:
        x1, y1 = nodes[start - 1][1], nodes[start - 1][2]
        x2, y2 = nodes[end - 1][1], nodes[end - 1][2]

        # Calculate arrow direction
        dx = x2 - x1
        dy = y2 - y1

        ax.annotate('', xy=(x2 - 0.4 * dx / abs(dx), y2 - 0.4 * dy / abs(dy)),
                    xytext=(x1 + 0.4 * dx / abs(dx), y1 + 0.4 * dy / abs(dy)),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    # Add data dimension descriptions
    dim_texts = [
        (2, 5.3, '3×518×518'),
        (4, 5.3, 'Normalization'),
        (5, 4.7, 'F_Res: 1×384'),
        (5, 2.7, 'F_DINO: 1×384'),
        (7, 2.7, 'Concat: 1×768'),
        (9, 2.7, 'Weight Learning'),
        (11, 2.7, 'FC Layers'),
        (13, 2.7, '[p_HGC, p_LGC, p_MIBC]')  # Updated for 3 classes
    ]

    for x, y, text in dim_texts:
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    ax.set_title('Model Data Flow Diagram', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('data_flow_diagram.pdf', bbox_inches='tight')
    plt.show()


# Run plotting
if __name__ == "__main__":
    print("Plotting model architecture...")
    plot_model_architecture()

    print("Plotting data flow diagram...")
    plot_data_flow_diagram()

    print("Figures saved in PNG and PDF formats.")
