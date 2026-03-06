# -Fashion-MNIST-Image-Classifier-PyTorch-CNN
# 👗 Fashion MNIST Image Classifier — PyTorch CNN

A deep learning project that classifies clothing items from the **Fashion MNIST** dataset using a custom Convolutional Neural Network (CNN) built with PyTorch. Achieved a **test accuracy of ~92.49%** through advanced data augmentation and regularization techniques.

---

## 📊 Results

| Optimizer | Augmentation | Best Test Accuracy |
|-----------|-------------|-------------------|
| SGD | ✅ Yes | 90.26% |
| AdamW | ✅ Yes | 91.14% (early run) |
| AdamW | ✅ Yes | **92.49%** (final run, 20 epochs) |

> Training was performed on a **NVIDIA T4 GPU** (Google Colab) — total training time: ~718 seconds.

---

## 🗂️ Dataset

- **Source**: `torchvision.datasets.FashionMNIST`
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image size**: 28×28 grayscale
- **Classes (10)**:

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 🏗️ Model Architecture — `FashionMNISTModelV2`

A deep CNN with 4 convolutional blocks followed by a fully connected classifier:

```
Input: [B, 1, 28, 28]
  ↓ Conv2d(1→32, 5×5) → ReLU → BatchNorm → MaxPool     → [B, 32, 14, 14]
  ↓ Conv2d(32→64, 3×3) → ReLU → BatchNorm → MaxPool    → [B, 64, 7, 7]
  ↓ Conv2d(64→128, 3×3) → ReLU → BatchNorm → MaxPool   → [B, 128, 3, 3]
  ↓ Conv2d(128→256, 3×3) → ReLU → BatchNorm             → [B, 256, 3, 3]
  ↓ Flatten
  ↓ Linear(2304 → 256) → ReLU → Dropout(0.5)
  ↓ Linear(256 → 10)
Output: [B, 10] logits
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Loss Function | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Epochs | 20 |
| Batch Size | 32 |

---

## 🔧 Data Augmentation (Training Only)

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])
```

Test images use only `ToTensor()` + `Normalize`.

---

## 📈 Training Log (20 Epochs)

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
|-------|-----------|-----------|----------|---------|
| 0 | 0.25054 | 90.72% | 0.23640 | 91.68% |
| 5 | 0.22756 | 91.55% | 0.23187 | 91.93% |
| 10 | 0.25455 | 90.82% | 0.22152 | 92.19% |
| 12 | 0.25017 | 91.01% | 0.21195 | **92.35%** |
| 17 | 0.23795 | 91.24% | 0.21180 | **92.49%** |
| 19 | 0.23511 | 91.48% | 0.22086 | 92.27% |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision tqdm
```

### Run the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fashion-mnist-cnn.git
   cd fashion-mnist-cnn
   ```

2. Open the notebook in Jupyter or Google Colab:
   ```bash
   jupyter notebook FashionMNIST_CNN.ipynb
   ```

3. For GPU acceleration, enable a CUDA-compatible GPU (recommended: Google Colab T4 GPU).

---

## 📁 Project Structure

```
fashion-mnist-cnn/
│
├── FashionMNIST_CNN.ipynb   # Main training notebook
├── helper_functions.py       # Utility functions (accuracy, timing)
├── data/                     # Auto-downloaded FashionMNIST dataset
└── README.md
```

---

## 🛠️ Tech Stack

- **Python** 3.x
- **PyTorch** + **torchvision**
- **Google Colab** (NVIDIA T4 GPU)
- **Matplotlib** for visualization

---

## 📌 Key Takeaways

- **BatchNorm** after every conv layer significantly stabilized training.
- **Dropout(0.5)** before the final layer reduced overfitting.
- **AdamW** with weight decay outperformed plain SGD by ~2%.
- **ReduceLROnPlateau** helped fine-tune convergence in later epochs.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
