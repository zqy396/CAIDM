CAIDM

This project is a deep learning framework designed for the  classification of  images.
---

## 📂 Project Structure

```text
├── main.py           # Main training and testing script (includes augmentation, training loop, metrics)
├── get_model.py      # Model definition (ResNet50 + DINOv2 dual encoders, Gating network, Classifier)
├── Loss.py           # Custom loss function (Cross-Entropy + Feature Separation Loss + Gate Regularization)
├── dataset.py        # Dataset splitting & black-border cropping (Connected Component Analysis)
├── Grad-CAM.py       # Explainability analysis (ResNet Grad-CAM + DINOv2 Attention Rollout)
├── framwork.py       # Automatically draws model architecture and data flow diagrams (PNG/PDF)
├── fps.py            # Video frame extraction tool
└── README.md         # Documentation
```

---

## 🛠️ Environment Setup

Ensure you have installed the following Python libraries (Anaconda environment is recommended):

```bash
pip install torch torchvision torchaudio
pip install timm opencv-python scikit-learn matplotlib tqdm pillow pyyaml
```

---

## 🚀 Usage Workflow

### 1. Video Frame Extraction (Optional)
If your raw data is in video format, use `fps.py` to extract frames:
```bash
python fps.py
```
*(By default, it extracts from `test.mp4`, saving one frame every 30 frames to the `frames/` directory)*

### 2. Data Preprocessing & Splitting
Place your original images into the class folders under `dataset/all/` (e.g., `dataset/all/mibc/``dataset/all/lg/` and `dataset/all/hg/`).
Run `dataset.py` to automatically perform:
- **Black-Border Cropping**: Uses Connected Component Analysis to locate the circular endoscopic field of view, ignoring watermarks/logos in the corners, and crops the image into a square.
- **Data Splitting**: Splits the dataset into `train`, `valid`, and `test` sets at a 70% / 15% / 15% ratio.

```bash
python dataset.py
```

### 3. Model Architecture Plotting (Optional)
Run `framwork.py` to automatically generate publication-quality model architecture and data flow diagrams (saved as PNG and PDF):
```bash
python framwork.py
```

### 4. Model Training & Evaluation
Run `main.py` to start training. The training pipeline applies strong data augmentation (Random Horizontal/Vertical Flips, ±15° Rotation, Color Jittering, Gaussian Blur, and Random Erasing) to prevent overfitting.
- **Automatic Checkpointing**: Saves `checkpoint_epoch_X.pth` at the end of every epoch, and separately saves the best-performing `best_model.pth`.
- **Automated Reports**: After training, it automatically loads the best model for testing, outputs the confusion matrix, classification report, and ROC curve, and saves them to the `results/` directory.

```bash
python main.py
```

### 5. Explainability Analysis (Grad-CAM)
Run `Grad-CAM.py`, which automatically iterates through all `.pth` files in the `checkpoints/` directory.
- For the **ResNet branch**, it computes standard Grad-CAM (using percentile clipping to enhance contrast).
- For the **DINOv2 branch**, it uses the **Attention Rollout** algorithm (a ViT-specific visualization method) to extract attention heatmaps.
- **Output Format**: Results are saved in a tree structure under `gradcam_results_all/`, making it extremely convenient to compare the heatmap evolution of the same image across different checkpoints (Epochs).

```bash
python Grad-CAM.py
```
*Example output path: `gradcam_results_all/test/bca/case_001/best_model_resnet.png`*

---

## 🧠 Core Module Overview

### 1. Model Architecture (`get_model.py`)
- **`ResNet50Encoder`**: Removes the final FC layer, outputs local feature maps, pools them, and projects them to 384 dimensions.
- **`DINOv2Encoder`**: Based on `vit_small_patch14_dinov2` from the `timm` library, extracting 384-dimensional global features.
- **`GatingNetwork` (MoE)**: Concatenates the two features and uses an MLP to learn dynamic weights (via Softmax), achieving adaptive feature fusion.

### 2. Custom Loss Function (`Loss.py`)
In addition to standard Cross-Entropy, it combines:
- **Feature Separation Loss**: Calculates the distance between class feature centers, forcing features to be more dispersed in the latent space.
- **Gate Regularization Loss**: Computes entropy and balance ratios to prevent the model from overly relying on a single branch during training.

### 3. Attention Rollout (`Grad-CAM.py`)
Since DINOv2 is a ViT model (often frozen or self-supervised pre-trained), traditional Grad-CAM often fails. This project implements `DINOv2AttentionRollout`, which extracts the attention matrices from all Transformer Blocks and performs matrix multiplication accumulation, perfectly mapping out the model's visual focus.

---

## 📊 Output Directory Guide
- `checkpoints/`: Saves all `.pth` weight files.
- `results/`:
  - `training_history.png` / `.json`: Training loss and accuracy curves.
  - `confusion_matrix.png`: Confusion matrix.
  - `roc_curve.png`: AUC curve.
  - `final_report.txt`: Aggregated text report.
- `gradcam_results_all/`: Grad-CAM visualization images, stored in a nested structure: `Dataset(train/valid/test) -> Class(bca/nor) -> Image Name -> Checkpoint(Epoch)`.
