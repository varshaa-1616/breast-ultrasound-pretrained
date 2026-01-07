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
## Results

### BUSI Dataset

| Model | Precision | Recall | F1-score | IoU | Dice |
|------|----------|--------|---------|-----|------|
| ResNet34 | 0.7657 | 0.7315 | 0.7387 | 0.6007 | 0.7387 |
| ResNet50 | 0.6271 | 0.8177 | 0.6944 | 0.5431 | 0.6944 |
| DenseNet121 | 0.7357 | 0.7458 | 0.7334 | 0.5959 | 0.7334 |
| EfficientNet-B0 | 0.6723 | 0.7920 | 0.7159 | 0.5777 | 0.7159 |
| EfficientNet-B1 | 0.6258 | 0.7879 | 0.6782 | 0.5307 | 0.6782 |
| EfficientNet-B2 | 0.6569 | 0.8057 | 0.7032 | 0.5551 | 0.7032 |
| EfficientNet-B3 | 0.6769 | 0.7716 | 0.7080 | 0.5634 | 0.7080 |
| EfficientNet-B4 | 0.7183 | 0.7130 | 0.7034 | 0.5578 | 0.7034 |
| EfficientNet-B5 | 0.6812 | 0.7793 | 0.7171 | 0.5712 | 0.7171 |
| EfficientNet-B6 | 0.7071 | 0.7742 | 0.7245 | 0.5831 | 0.7245 |
| EfficientNet-B7 | 0.7133 | 0.8078 | 0.7324 | 0.5955 | 0.7324 |
| VGG19-BN | 0.7009 | 0.7598 | 0.7109 | 0.5635 | 0.7109 |
| MobileNetV2 | 0.5844 | 0.8181 | 0.6611 | 0.5125 | 0.6611 |
| MiT-B0 | 0.6964 | 0.7580 | 0.7182 | 0.5774 | 0.7182 |
| MiT-B1 | 0.6694 | 0.8358 | 0.7283 | 0.5914 | 0.7283 |
| MiT-B2 | 0.7088 | 0.7908 | 0.7356 | 0.6015 | 0.7356 |
| MiT-B3 | 0.7425 | 0.7884 | 0.7529 | 0.6272 | 0.7529 |
| MiT-B4 | 0.7210 | 0.8510 | 0.7609 | 0.6351 | 0.7609 |
| Swin-Tiny | 0.6630 | 0.7787 | 0.7042 | 0.5589 | 0.7042 |
| Swin-Base | 0.5076 | 0.7737 | 0.5954 | 0.4411 | 0.5954 |
| PVTv2-B0 | 0.6770 | 0.7647 | 0.7069 | 0.5584 | 0.7069 |
| PVTv2-B2 | 0.7253 | 0.7231 | 0.7165 | 0.5692 | 0.7165 |
| ViT-Base | 0.6242 | 0.7827 | 0.6810 | 0.5331 | 0.6810 |
| ViT-Small | 0.7070 | 0.7648 | 0.7231 | 0.5789 | 0.7231 |
| PVTv2-B3 | 0.7387 | 0.7992 | 0.7607 | 0.6251 | 0.7607 |
| PVTv2-B4 | 0.6656 | 0.7707 | 0.7048 | 0.5571 | 0.7048 |
| PVTv2-B5 | 0.7245 | 0.7733 | 0.7379 | 0.5974 | 0.7379 |

---

### BUS-BRA Dataset

| Model | Precision | Recall | F1-score | IoU | Dice |
|------|----------|--------|---------|-----|------|
| ResNet34 | 0.8621 | 0.9314 | 0.8942 | 0.8124 | 0.8942 |
| ResNet50 | 0.8564 | 0.9254 | 0.8873 | 0.8009 | 0.8873 |
| DenseNet121 | 0.8381 | 0.9428 | 0.8860 | 0.7989 | 0.8860 |
| EfficientNet-B0 | 0.8723 | 0.9211 | 0.8947 | 0.8124 | 0.8947 |
| EfficientNet-B1 | 0.8523 | 0.9365 | 0.8912 | 0.8065 | 0.8912 |
| EfficientNet-B2 | 0.8314 | 0.9425 | 0.8820 | 0.7924 | 0.8820 |
| EfficientNet-B3 | 0.8660 | 0.9388 | 0.8999 | 0.8202 | 0.8999 |
| EfficientNet-B4 | 0.8433 | 0.9428 | 0.8894 | 0.8041 | 0.8894 |
| EfficientNet-B5 | 0.8592 | 0.9388 | 0.8963 | 0.8151 | 0.8963 |
| EfficientNet-B6 | 0.8540 | 0.9441 | 0.8961 | 0.8148 | 0.8961 |
| EfficientNet-B7 | 0.8566 | 0.9457 | 0.8981 | 0.8172 | 0.8981 |
| VGG19-BN | 0.8565 | 0.9308 | 0.8912 | 0.8078 | 0.8912 |
| MobileNetV2 | 0.8333 | 0.9372 | 0.8811 | 0.7903 | 0.8811 |
| MiT-B0 | 0.8016 | 0.9489 | 0.8663 | 0.7695 | 0.8663 |
| Swin-Tiny | 0.8690 | 0.9246 | 0.8947 | 0.8125 | 0.8947 |
| Swin-Base | 0.8461 | 0.9275 | 0.8829 | 0.7942 | 0.8829 |
| PVTv2-B0 | 0.8512 | 0.9328 | 0.8884 | 0.8019 | 0.8884 |
| PVTv2-B2 | 0.8738 | 0.9239 | 0.8968 | 0.8158 | 0.8968 |
| InceptionResNetV2 | 0.8621 | 0.9430 | 0.9002 | 0.8207 | 0.9002 |
| InceptionV4 | 0.8412 | 0.9357 | 0.8844 | 0.7969 | 0.8844 |
| ViT-Base | 0.8188 | 0.9351 | 0.8714 | 0.7758 | 0.8714 |
| ViT-Small | 0.8134 | 0.9504 | 0.8749 | 0.7807 | 0.8749 |
| MiT-B3 | 0.8716 | 0.9388 | 0.9029 | 0.8257 | 0.9029 |
| MiT-B4 | 0.8394 | 0.9513 | 0.8903 | 0.8064 | 0.8903 |

---

### BrEaST Dataset

| Model | Precision | Recall | F1-score | IoU | Dice |
|------|----------|--------|---------|-----|------|
| ResNet34 | 0.1639 | 0.9675 | 0.2745 | 0.1624 | 0.2745 |
| ResNet50 | 0.2575 | 0.6598 | 0.3571 | 0.2183 | 0.3571 |
| DenseNet121 | 0.0992 | 0.8098 | 0.1739 | 0.0961 | 0.1739 |
| EfficientNet-B0 | 0.1228 | 0.4110 | 0.1821 | 0.1010 | 0.1821 |
| VGG19-BN | 0.3063 | 0.6581 | 0.4106 | 0.2589 | 0.4106 |
| MobileNetV2 | 0.2065 | 0.7881 | 0.3163 | 0.1899 | 0.3163 |
| InceptionResNetV2 | 0.2302 | 0.8135 | 0.3493 | 0.2150 | 0.3493 |
| InceptionV4 | 0.1351 | 0.9942 | 0.2340 | 0.1349 | 0.2340 |
| MiT-B0 | 0.2072 | 0.9242 | 0.3342 | 0.2037 | 0.3342 |
| MiT-B1 | 0.2151 | 0.9904 | 0.3468 | 0.2143 | 0.3468 |
| MiT-B2 | 0.2092 | 0.9814 | 0.3379 | 0.2079 | 0.3379 |
| MiT-B3 | 0.1933 | 0.9107 | 0.3117 | 0.1874 | 0.3117 |
| MiT-B4 | 0.1619 | 0.9968 | 0.2741 | 0.1617 | 0.2741 |
| Swin-Tiny | 0.7782 | 0.8343 | 0.8047 | 0.6737 | 0.8047 |
| Swin-Base | 0.7516 | 0.8607 | 0.8001 | 0.6704 | 0.8001 |
| PVTv2-B0 | 0.7560 | 0.8450 | 0.7943 | 0.6625 | 0.7943 |
| PVTv2-B2 | 0.6999 | 0.9026 | 0.7868 | 0.6514 | 0.7868 |
| ViT-Base | 0.7719 | 0.8339 | 0.8001 | 0.6678 | 0.8001 |
| ViT-Small | 0.7898 | 0.8463 | 0.8156 | 0.6895 | 0.8156 |

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


