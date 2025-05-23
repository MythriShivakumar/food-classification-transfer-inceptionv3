# Food Classification Using Transfer Learning (InceptionV3)

This notebook showcases transfer learning with a pre-trained **InceptionV3** model from PyTorch to classify images from a custom multi-class food dataset. It includes data preprocessing, model fine-tuning, and evaluation.

## 🍽️ Dataset

- Custom dataset of food images
- Folder structure:
  - `training/`
  - `validation/`
  - `evaluation/`
- Format compatible with `torchvision.datasets.ImageFolder`

## 🧠 Model

- **Base:** `InceptionV3` (pretrained on ImageNet)
- **Modification:** Final layer adjusted for number of food classes
- **Optimizer:** Adam
- **Loss:** Cross-Entropy Loss

## 🔍 Preprocessing

- Resize, crop, and normalize based on ImageNet mean/std
- Random horizontal flips and resized cropping for training data
- Center cropping for validation and evaluation

## 🧪 Metrics

- Accuracy on validation and evaluation sets
- Optionally includes loss tracking and visualization

## 🧰 Requirements

```bash
pip install torch torchvision matplotlib numpy scikit-learn
