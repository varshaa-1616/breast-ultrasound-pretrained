# Breast Ultrasound Lesion Segmentation using Pretrained CNN and Transformer Models

---

## Overview
This repository contains the implementation and experimental code for a **mini-project on breast ultrasound lesion segmentation** using **U-Net** with **pretrained CNN and Transformer encoders**.

The goal of this project is to evaluate and compare the performance of different **ImageNet-pretrained backbones** when used as encoders in a U-Net architecture across multiple breast ultrasound datasets.

---

## Key Objectives
- Use pretrained CNN encoders within a U-Net framework for lesion segmentation  
- Explore transformer-based encoders for capturing global contextual information  
- Compare performance across multiple breast ultrasound datasets  
- Maintain reproducibility through clean and modular code  

---

## Models and Architectures

### Base Architecture
- **Model**: U-Net  
- **Input size**: 256 × 256  
- **Output**: Binary segmentation mask  
- **Encoder weights**: ImageNet pretrained  

---

### CNN-based Pretrained Encoders
- ResNet34  
- ResNet50  
- DenseNet121  
- EfficientNet (B0 – B7)  
- VGG19 with Batch Normalization  
- MobileNetV2  

---

### Transformer-based Pretrained Encoders
- MiT (Mix Transformer) family (B0 – B5)  
- PVTv2 (Pyramid Vision Transformer)  
- Swin Transformer  
- Vision Transformer (ViT)  
- Inception-based transformer hybrids  

Encoders are integrated using:
- `segmentation_models_pytorch`
- `timm`

---

## Datasets

| Dataset  | Country | Approx. Images | Classes |
|--------|--------|----------------|--------|
| BUSI    | Egypt   | ~780  | Benign, Malignant, Normal |
| BUS-BRA | Brazil  | ~1875 | Benign, Malignant |
| BrEaST  | Poland  | ~256  | Benign, Malignant, Normal |

> **Note:**  
> Datasets are **not included** due to licensing restrictions.  
> Users must download them separately and update dataset paths in the training scripts.

---

## Project Structure
```text
breast-ultrasound-pretrained/
├── train_cnn_mit.py          # U-Net with CNN and MiT encoders
├── train_transformer.py     # U-Net with TIMM transformer encoders
├── requirements.txt
└── README.md
```


## Methodology

### Data Splitting
The dataset is divided into three mutually exclusive subsets to ensure fair evaluation and reproducibility:

- **Training set**: 70%  
- **Validation set**: 15%  
- **Testing set**: 15%  

A **fixed random seed** is used during data splitting to ensure that the same samples are selected across multiple runs, enabling consistent comparison of model performance.


### Loss Function
Breast ultrasound images often suffer from severe class imbalance, where lesion regions occupy a significantly smaller portion of the image compared to the background. To address this, a **combined loss function** is employed:

- **Weighted Binary Cross-Entropy (BCE) Loss**
- **Dice Loss**

The weighted BCE loss penalizes misclassification of lesion pixels, while Dice loss directly optimizes region overlap, leading to improved foreground segmentation and more stable training.

---

### Evaluation Metrics
Model performance is evaluated using the following standard segmentation metrics:

- **Precision** – Measures the correctness of predicted lesion pixels  
- **Recall** – Measures the ability to capture all lesion pixels  
- **F1-score** – Harmonic mean of precision and recall  
- **Intersection over Union (IoU)** – Measures overlap between prediction and ground truth  
- **Dice Coefficient** – Measures similarity between predicted and true masks  
- **Accuracy** – Reported for completeness, though less informative for imbalanced segmentation tasks  

---

## Training Configuration
- **Input resolution**: 256 × 256  
- **Encoder initialization**: ImageNet pretrained weights  
- **Optimizer**: Adam  
- **Learning rate**: Fixed learning rate  
- **Batch size**: Selected based on GPU memory constraints  
- **Epochs**: Limited to maintain mini-project scope  

A fixed random seed is used across all experiments to ensure reproducibility.

---

## Reproducibility
To ensure reproducibility of experimental results:

- All random number generators are initialized with a fixed seed  
- ImageNet pretrained weights are used consistently across models  
- Identical data splits are maintained for all architectures  
- Modular and well-documented code structure is followed  

---

## Limitations
The following limitations were intentionally accepted to keep the project within a mini-project scope:

- No extensive hyperparameter tuning  
- No post-processing applied to predicted segmentation masks  
- Limited training epochs due to computational constraints  
- Performance on smaller datasets may be affected by limited sample size  

---

## Future Work
Potential directions for extending this work include:

- Cross-dataset training and evaluation for improved generalization  
- Lightweight transformer architectures tailored for small datasets  
- Semi-supervised and weakly supervised segmentation approaches  
- Post-processing techniques for boundary refinement and noise reduction  

---

## Acknowledgements
The authors would like to acknowledge the following tools and resources:

- `segmentation_models_pytorch`
- `timm`
- Authors and contributors of the BUSI, BUS-BRA, and BrEaST datasets

