
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from get_model import get_model, ModelManager
from Loss import Loss
import torch.nn.functional as F
from torchvision import transforms, datasets
import os
import json
from datetime import datetime
from tqdm import tqdm
import time
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 1000000000

# ================= 全局类别定义（改回三类） =================
CLASS_NAMES = ['HGC', 'LGC', 'MIBC']
NUM_CLASSES = len(CLASS_NAMES)


class Trainer:
    """训练器类"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None,
                 save_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.epoch_times = []
        self.start_time = None

        self.best_val_acc = 0.0
        self.best_model_state = None
        self.best_epoch = 0

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        batch_bar = tqdm(total=len(self.train_loader),
                         desc=f"Epoch {epoch}/{total_epochs}",
                         position=0, leave=True, ncols=100)

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs['logits'], 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)

            batch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(predicted == targets).float().mean().item():.4f}"
            )
            batch_bar.update(1)

        batch_bar.close()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct_predictions / total_predictions
        return epoch_loss, epoch_accuracy

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = torch.max(outputs['logits'], 1)
                correct_predictions += (predicted == targets).sum().item()
                total_predictions += targets.size(0)

        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = correct_predictions / total_predictions
        return epoch_loss, epoch_accuracy

    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}分钟"
        else:
            return f"{seconds / 3600:.1f}小时"

    def print_epoch_summary(self, epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, epoch_time):
        self.epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = total_epochs - epoch
        estimated_remaining_time = avg_epoch_time * remaining_epochs

        print(f"\nEpoch {epoch}/{total_epochs}:")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"  本轮时间: {self.format_time(epoch_time)}")
        print(f"  平均每轮时间: {self.format_time(avg_epoch_time)}")
        print(f"  预计剩余时间: {self.format_time(estimated_remaining_time)}")

        if val_acc > self.best_val_acc:
            print(f"  🎯 新最佳模型! 验证准确率: {val_acc:.4f}")

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'epoch_times': self.epoch_times,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'class_names': CLASS_NAMES,
            'num_classes': NUM_CLASSES,
        }

        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_model_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f'✅ 最佳模型已保存，验证准确率: {self.best_val_acc:.4f}')

    def train(self, num_epochs, save_interval=5):
        print("开始训练...")
        self.start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            train_loss, train_acc = self.train_epoch(epoch, num_epochs)
            val_loss, val_acc = self.validate_epoch()

            if self.scheduler:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start_time

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.print_epoch_summary(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, epoch_time)

            is_best = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_model_state = self.model.state_dict().copy()
                is_best = True

            self.save_checkpoint(epoch, is_best=is_best)

        total_time = time.time() - self.start_time
        print(f"\n🎉 训练完成！")
        print(f"   总训练时间: {self.format_time(total_time)}")
        print(f"   最佳验证准确率: {self.best_val_acc:.4f} (第 {self.best_epoch} 轮)")
        print(f"   平均每轮时间: {self.format_time(np.mean(self.epoch_times))}")

        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies


def evaluate_model(model, test_loader, device, class_names=CLASS_NAMES):
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            probabilities = F.softmax(outputs['logits'], dim=1)
            _, predicted = torch.max(outputs['logits'], 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions, labels=list(range(len(class_names))))
    report = classification_report(all_targets, all_predictions, target_names=class_names,
                                    labels=list(range(len(class_names))), zero_division=0)

    print(f"测试准确率: {accuracy:.4f}")
    print("\n混淆矩阵:")
    print(cm)
    print("\n分类报告:")
    print(report)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }


def plot_training_history(trainer, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(trainer.train_losses) + 1)
    ax1.plot(epochs, trainer.train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, trainer.val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if trainer.best_epoch > 0:
        ax1.axvline(x=trainer.best_epoch, color='g', linestyle='--', alpha=0.7,
                    label=f'最佳轮次 ({trainer.best_epoch})')
        ax1.legend()

    ax2.plot(epochs, trainer.train_accuracies, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, trainer.val_accuracies, 'r-', label='验证准确率', linewidth=2)
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_epoch_idx = trainer.best_epoch - 1
    if trainer.best_epoch > 0 and 0 <= best_epoch_idx < len(trainer.val_accuracies):
        ax2.axvline(x=trainer.best_epoch, color='g', linestyle='--', alpha=0.7,
                    label=f'最佳准确率: {trainer.best_val_acc:.4f}')
        ax2.plot(trainer.best_epoch, trainer.val_accuracies[best_epoch_idx], 'go', markersize=8)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'training_history.pdf'), bbox_inches='tight')
    plt.close(fig)

    history_data = {
        'train_losses': [float(x) for x in trainer.train_losses],
        'val_losses': [float(x) for x in trainer.val_losses],
        'train_accuracies': [float(x) for x in trainer.train_accuracies],
        'val_accuracies': [float(x) for x in trainer.val_accuracies],
        'best_val_acc': float(trainer.best_val_acc),
        'best_epoch': int(trainer.best_epoch),
        'class_names': CLASS_NAMES,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(save_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)


def plot_roc_curve(results, class_names=CLASS_NAMES, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    probabilities = np.array(results['probabilities'])
    targets = np.array(results['targets'])

    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fig = plt.figure(figsize=(10, 8))

    # 多分类 One-vs-Rest ROC
    for i in range(n_classes):
        binary_targets = (targets == i).astype(int)
        if binary_targets.sum() == 0:
            print(f"⚠️ 类别 {class_names[i]} 在测试集中没有样本，跳过该类别 ROC")
            continue
        fpr[i], tpr[i], _ = roc_curve(binary_targets, probabilities[:, i])
        roc_auc[i] = float(auc(fpr[i], tpr[i]))

        plt.plot(fpr[i], tpr[i], linewidth=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('多分类ROC曲线 (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'roc_curve.pdf'), bbox_inches='tight')
    plt.close(fig)

    roc_auc_named = {class_names[k]: float(v) for k, v in roc_auc.items()}

    roc_data = {
        'fpr': {class_names[k]: v.tolist() for k, v in fpr.items()},
        'tpr': {class_names[k]: v.tolist() for k, v in tpr.items()},
        'auc': roc_auc_named,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(save_dir, 'roc_data.json'), 'w', encoding='utf-8') as f:
        json.dump(roc_data, f, indent=4, ensure_ascii=False)

    return roc_auc_named


def plot_confusion_matrix(results, class_names=CLASS_NAMES, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    cm = results['confusion_matrix']

    fig = plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
    plt.close(fig)


def save_final_results(results, trainer, roc_auc, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    final_results = {
        'test_accuracy': float(results['accuracy']),
        'best_validation_accuracy': float(trainer.best_val_acc),
        'best_epoch': int(trainer.best_epoch),
        'roc_auc': {k: float(v) for k, v in roc_auc.items()},
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report'],
        'training_history': {
            'train_losses': [float(x) for x in trainer.train_losses],
            'val_losses': [float(x) for x in trainer.val_losses],
            'train_accuracies': [float(x) for x in trainer.train_accuracies],
            'val_accuracies': [float(x) for x in trainer.val_accuracies]
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_config': {
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'dropout': 0.1
        }
    }

    with open(os.path.join(save_dir, 'final_results.json'), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    with open(os.path.join(save_dir, 'final_report.txt'), 'w', encoding='utf-8') as f:
        f.write("模型训练最终报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"训练时间: {final_results['timestamp']}\n")
        f.write(f"类别: {CLASS_NAMES}\n")
        f.write(f"测试准确率: {final_results['test_accuracy']:.4f}\n")
        f.write(f"最佳验证准确率: {final_results['best_validation_accuracy']:.4f} "
                f"(第{final_results['best_epoch']}轮)\n\n")

        f.write("ROC AUC分数:\n")
        for class_name, auc_score in final_results['roc_auc'].items():
            f.write(f"  {class_name}: {auc_score:.4f}\n")
        f.write("\n")

        f.write("混淆矩阵:\n")
        f.write(str(results['confusion_matrix']) + "\n\n")

        f.write("分类报告:\n")
        f.write(results['classification_report'] + "\n")


# ================= 使用示例 =================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    config = {
        'num_classes': NUM_CLASSES,
        'dropout': 0.1
    }

    model = get_model(config).to(device)
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")
        model = nn.DataParallel(model)

    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        ]),
        'valid': transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    dataset = '/mnt/data4/膀胱镜诊断/dataset'
    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')
    test_directory = os.path.join(dataset, 'test')

    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
    }

    print(f"📂 训练集类别: {data['train'].classes}  | 数量: {[len([x for x in data['train'].targets if x == i]) for i in range(len(data['train'].classes))]}")
    print(f"📂 验证集类别: {data['valid'].classes}")
    print(f"📂 测试集类别: {data['test'].classes}")

    if data['train'].classes != CLASS_NAMES:
        print(f"⚠️ 警告: 数据集实际类别顺序 {data['train'].classes} 与 CLASS_NAMES {CLASS_NAMES} 不一致！"
              f"请调整 CLASS_NAMES 顺序或文件夹名，否则 label 与类别名会错配。")

    train_loader = DataLoader(data['train'], batch_size=30, shuffle=True)
    val_loader = DataLoader(data['valid'], batch_size=30, shuffle=True)
    test_loader = DataLoader(data['test'], batch_size=30, shuffle=True)

    criterion = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler,
                      save_dir='checkpoints')

    train_losses, val_losses, train_accs, val_accs = trainer.train(num_epochs=20)

    if trainer.best_model_state is not None:
        print("🔄 加载最佳模型权重进行测试...")
        state_dict = trainer.best_model_state
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    results = evaluate_model(model, test_loader, device, class_names=CLASS_NAMES)

    plot_training_history(trainer, save_dir='results')
    roc_auc = plot_roc_curve(results, class_names=CLASS_NAMES, save_dir='results')
    plot_confusion_matrix(results, class_names=CLASS_NAMES, save_dir='results')

    save_final_results(results, trainer, roc_auc, save_dir='results')

    print("模型训练和评估完成！所有结果已保存在 results 文件夹中。")
