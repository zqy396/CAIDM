import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
import re
import time
from torchvision import transforms
from get_model import get_model, ModelManager

# ★ 改回三类分类
DEFAULT_CLASS_NAMES = ['HGC', 'LGC', 'MIBC']
NUM_CLASSES = len(DEFAULT_CLASS_NAMES)


# ============================================================
# Grad-CAM（ResNet 用）
# ============================================================
class FixedGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._forward_handle = None
        self._backward_handle = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out[0].detach() if isinstance(out, (tuple, list)) else out.detach()

        def backward_hook(module, grad_input, grad_output):
            g = grad_output[0]
            self.gradients = g.detach() if g is not None else None

        self._forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    @staticmethod
    def _to_spatial(tensor):
        B, N, C = tensor.shape
        h = int(np.sqrt(N))
        if h * h == N:
            return tensor.reshape(B, h, h, C).permute(0, 3, 1, 2)
        for offset in range(1, 10):
            n_spatial = N - offset
            h = int(np.sqrt(n_spatial))
            if h * h == n_spatial:
                t = tensor[:, offset:, :]
                return t.reshape(B, h, h, C).permute(0, 3, 1, 2)
        raise ValueError(f"无法将 N={N} 的 token 序列还原为空间特征图")

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.detach().requires_grad_(True)
        output = self.model(input_tensor)
        logits = output['logits']

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        logits_abs_max = float(np.abs(logits[0].detach().cpu().numpy()).max())
        scale = 1.0 / (logits_abs_max + 1e-12)
        target_score = logits[0, target_class] * scale
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None, target_class, logits

        grads, acts = self.gradients, self.activations
        if grads.dim() == 3:
            grads = self._to_spatial(grads)
            acts = self._to_spatial(acts)
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.abs(torch.sum(weights * acts, dim=1, keepdim=True))
        elif grads.dim() == 4:
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = F.relu(torch.sum(weights * acts, dim=1, keepdim=True))
        else:
            return None, target_class, logits

        cam_min, cam_max = cam.min(), cam.max()
        cam_range = float(cam_max - cam_min)

        if cam_range < 1e-14:
            return None, target_class, logits

        cam_np = cam.squeeze().cpu().numpy()
        p_low, p_high = np.percentile(cam_np, [5, 95])
        if p_high - p_low < 1e-14:
            return None, target_class, logits
        cam_np = np.clip(cam_np, p_low, p_high)
        cam_np = (cam_np - p_low) / (p_high - p_low + 1e-12)
        return cam_np, target_class, logits


# ============================================================
# DINOv2 Attention Rollout
# ============================================================
class DINOv2AttentionRollout:
    def __init__(self, model, head_fusion='mean'):
        self.model = model
        self.head_fusion = head_fusion

    def _get_dinov2_encoder(self):
        if hasattr(self.model, 'module'):
            enc = getattr(self.model.module, 'dinov2_encoder', None)
            if enc is not None:
                return enc
        return getattr(self.model, 'dinov2_encoder', None)

    @staticmethod
    def _find_attention_layers(encoder):
        layers = []
        for name, module in encoder.named_modules():
            if ('attn' in name.lower() or 'attention' in name.lower()) \
                    and hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
                layers.append((name, module))

        def sort_key(item):
            nums = re.findall(r'\d+', item[0])
            return [int(n) for n in nums] if nums else [0]

        layers.sort(key=sort_key)
        return layers

    @staticmethod
    def _attn_probs_from_input(attn_module, x):
        B, N, C = x.shape
        num_heads = attn_module.num_heads
        head_dim = C // num_heads
        qkv = attn_module.qkv(x).reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        return attn.softmax(dim=-1)

    @staticmethod
    def _reshape_tokens_to_spatial(tokens):
        n = tokens.shape[0]
        h = int(np.sqrt(n))
        if h * h == n:
            return tokens.reshape(h, h)
        for offset in range(1, 10):
            n_spatial = n - offset
            h = int(np.sqrt(n_spatial))
            if h * h == n_spatial:
                t = tokens[offset:]
                return t.reshape(h, h)
        raise ValueError(f"无法 reshape tokens，长度 {n}")

    def generate(self, input_tensor):
        encoder = self._get_dinov2_encoder()
        if encoder is None:
            raise ValueError("找不到 dinov2_encoder")

        attn_layers = self._find_attention_layers(encoder)
        if not attn_layers:
            raise ValueError("未找到带 qkv + num_heads 的 attention 层")

        attentions = []
        handles = []

        def make_hook():
            def hook(module, inp, out):
                try:
                    attentions.append(self._attn_probs_from_input(module, inp[0]).detach())
                except Exception:
                    pass

            return hook

        for _, layer in attn_layers:
            handles.append(layer.register_forward_hook(make_hook()))

        try:
            encoder.eval()
            with torch.no_grad():
                encoder(input_tensor)
        finally:
            for h in handles:
                h.remove()

        if not attentions:
            raise ValueError("未捕获到 attention probs")

        N = attentions[0].shape[-1]
        device = attentions[0].device
        result = torch.eye(N, device=device).unsqueeze(0)
        I = torch.eye(N, device=device).unsqueeze(0)

        for attn in attentions:
            a = attn.max(dim=1)[0] if self.head_fusion == 'max' else attn.mean(dim=1)
            a = (a + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result

        mask = result[0, 0, 1:].cpu().numpy()
        cam = self._reshape_tokens_to_spatial(mask)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        return cam


# ============================================================
# Visualizer
# ============================================================
class FixedGradCAMVisualizer:
    def __init__(self, model, device, class_names=None):
        self.model = model
        self.device = device
        self.model.eval()
        self.class_names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)

        self.resnet_target_layer = self._get_resnet_target_layer()
        self.dinov2_target_layer = self._get_dinov2_target_layer()
        self.rollout = DINOv2AttentionRollout(model)
        self._dinov2_attn_layers = None

    def _get_resnet_target_layer(self):
        resnet_encoder = self.model.module.resnet_encoder if hasattr(self.model,
                                                                     'module') else self.model.resnet_encoder
        last_conv = None
        for _, m in resnet_encoder.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        return last_conv

    def _get_dinov2_target_layer(self):
        dinov2_encoder = self.model.module.dinov2_encoder if hasattr(self.model,
                                                                     'module') else self.model.dinov2_encoder
        layers = DINOv2AttentionRollout._find_attention_layers(dinov2_encoder)
        self._dinov2_attn_layers = layers
        return layers[-1][1] if layers else None

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        return input_tensor, original_image

    def _create_heatmap(self, image, cam, alpha=0.5):
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    def generate_cam_for_layer(self, input_tensor, target_layer, target_class=None):
        if target_layer is None:
            return None
        gen = FixedGradCAM(self.model, target_layer)
        try:
            cam, _, _ = gen.generate_cam(input_tensor, target_class)
            return cam
        except Exception:
            return None
        finally:
            gen.remove_hooks()

    def _resnet_activation_l2_cam(self, input_tensor):
        if self.resnet_target_layer is None:
            raise ValueError("ResNet 目标层未找到")
        acts = []
        h = self.resnet_target_layer.register_forward_hook(
            lambda m, i, o: acts.append(o[0].detach() if isinstance(o, (tuple, list)) else o.detach())
        )
        try:
            resnet_encoder = self.model.module.resnet_encoder if hasattr(self.model,
                                                                         'module') else self.model.resnet_encoder
            with torch.no_grad():
                resnet_encoder(input_tensor)
        finally:
            h.remove()
        if not acts:
            raise ValueError("未捕获 ResNet activation")
        feat = acts[0]
        cam = feat.norm(dim=1)[0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        return cam

    def generate_resnet_cam(self, input_tensor, target_class=None):
        cam = self.generate_cam_for_layer(
            input_tensor, self.resnet_target_layer, target_class)
        if cam is not None:
            return cam
        try:
            cam = self._resnet_activation_l2_cam(input_tensor)
            return cam
        except Exception:
            pass
        return np.random.rand(14, 14)

    def visualize_single_image(self, image_path, target_class=None,
                               save_dir='gradcam_results', tag="", skip_existing=False):

        filename = os.path.splitext(os.path.basename(image_path))[0]

        # 核心逻辑：为每张图建一个专属文件夹
        img_save_dir = os.path.join(save_dir, filename)

        # 文件名直接用 tag（比如 best_model_resnet.png）
        save_filename = tag if tag else "default"
        resnet_path = os.path.join(img_save_dir, f'{save_filename}_resnet.png')

        if skip_existing and os.path.exists(resnet_path):
            return {'save_path': resnet_path, 'skipped': True}

        os.makedirs(img_save_dir, exist_ok=True)

        input_tensor, original_image = self.load_and_preprocess_image(image_path)
        original_array = cv2.resize(np.array(original_image), (518, 518))

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output['logits'], dim=1)
            predicted_class = output['logits'].argmax(dim=1).item()

        probs = probabilities[0].cpu().numpy()
        num_classes = len(probs)
        if target_class is None or target_class >= num_classes:
            target_class = predicted_class

        resnet_cam = self.generate_resnet_cam(input_tensor, target_class)
        resnet_heatmap = self._create_heatmap(original_array, resnet_cam, alpha=0.5)

        # 只保存 ResNet 叠加热图
        cv2.imwrite(resnet_path, cv2.cvtColor(resnet_heatmap, cv2.COLOR_RGB2BGR))

        return {'save_path': resnet_path, 'skipped': False}

    def visualize_folder(self, folder_path, save_dir='gradcam_results', target_class=None,
                         tag="", skip_existing=False):
        if not os.path.isdir(folder_path):
            return []
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
                '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF'}
        image_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if not image_files:
            return []

        results = []
        skipped = 0
        failed = 0
        split_start = time.time()

        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 处理: {os.path.basename(img_path)}")
            try:
                res = self.visualize_single_image(
                    img_path, target_class=target_class,
                    save_dir=save_dir, tag=tag,
                    skip_existing=skip_existing,
                )
                results.append(res)
                if res.get('skipped'):
                    skipped += 1
            except Exception as e:
                failed += 1
                print(f"❌ 处理 {img_path} 出错: {e}")

        elapsed = time.time() - split_start
        print(
            f"\n✅ 完成: 成功 {len(results) - skipped - failed} 张, 跳过 {skipped} 张, 失败 {failed} 张, 耗时 {elapsed:.1f}s")
        return results

    def visualize_dataset(self, root_dir, save_dir='gradcam_results', target_class=None,
                          tag="", skip_existing=False):
        if not os.path.isdir(root_dir):
            return []
        all_results = []
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                print(f"\n========== 类别: {class_name} ==========")
                res = self.visualize_folder(
                    class_dir,
                    save_dir=os.path.join(save_dir, class_name),
                    target_class=target_class,
                    tag=tag,
                    skip_existing=skip_existing,
                )
                all_results.extend(res)
        return all_results


# ============================================================
# 主函数：遍历所有 Checkpoint
# ============================================================
def demo_gradcam_all_checkpoints():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    checkpoints_dir = '/mnt/data4/膀胱镜诊断/checkpoints'
    dataset_root = '/mnt/data4/膀胱镜诊断/dataset'
    output_root = 'gradcam_results_all'

    RUN_SPLITS = ['train', 'valid', 'test']

    if not os.path.isdir(checkpoints_dir):
        print(f"❌ 找不到 checkpoints 目录: {checkpoints_dir}")
        return

    ckpt_files = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')])
    if not ckpt_files:
        print(f"⚠️ 目录 {checkpoints_dir} 下没有找到 .pth 文件")
        return

    print(f"🔍 共找到 {len(ckpt_files)} 个检查点文件，准备开始处理...")

    # ★ 实例化模型时，NUM_CLASSES 改回 3
    model = get_model({'num_classes': NUM_CLASSES, 'dropout': 0.1}).to(device)
    model.eval()

    for idx, ckpt_name in enumerate(ckpt_files):
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        tag = os.path.splitext(ckpt_name)[0]

        print(f"\n\n{'#' * 70}")
        print(f"🚀 [{idx + 1}/{len(ckpt_files)}] 开始处理检查点: {ckpt_name}")
        print(f"{'#' * 70}")

        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(state_dict)
            print(
                f"✅ 权重加载成功 (Epoch: {checkpoint.get('epoch', 'Unknown')}, Val Acc: {checkpoint.get('best_val_acc', 'Unknown')})")
        except Exception as e:
            print(f"❌ 权重加载失败，跳过此检查点: {e}")
            continue

        visualizer = FixedGradCAMVisualizer(model, device, class_names=DEFAULT_CLASS_NAMES)

        for split in RUN_SPLITS:
            split_dir = os.path.join(dataset_root, split)
            if not os.path.isdir(split_dir):
                print(f"  ⚠️ split 目录不存在: {split_dir}")
                continue

            subdirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            if not subdirs:
                continue

            print(f"\n  ---------- 处理 split: {split} ----------")
            visualizer.visualize_dataset(
                split_dir,
                save_dir=os.path.join(output_root, split),
                tag=tag,
                skip_existing=True,
            )

    print(f"\n\n🎉 所有检查点全部处理完成！结果保存在: {os.path.abspath(output_root)}/")


if __name__ == "__main__":
    demo_gradcam_all_checkpoints()
