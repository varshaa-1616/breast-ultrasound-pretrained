```markdown
# Breast Ultrasound Lesion Segmentation using Pretrained CNN and Transformer Models

## Overview
This repository contains the implementation and experimental code for a **mini-project on breast ultrasound lesion segmentation** using **U-Net** with **pretrained CNN and Transformer encoders**.

The objective of this work is to evaluate and compare the effectiveness of different **ImageNet-pretrained backbones** when used as encoders in a U-Net architecture across multiple breast ultrasound datasets.

---

## Key Objectives
- Apply pretrained CNN encoders within a U-Net framework for breast lesion segmentation
- Explore transformer-based encoders for capturing global contextual information
- Perform a comparative analysis across multiple datasets
- Ensure reproducibility through clean and modular code

---

## Models and Architectures

### Base Architecture
- Model: **U-Net**
- Input size: **256 × 256**
- Output: **Binary segmentation mask**
- Encoder initialization: **ImageNet pretrained weights**

---

### CNN-based Pretrained Encoders
The following CNN backbones were evaluated as U-Net encoders:

- ResNet34, ResNet50  
- DenseNet121  
- EfficientNet (B0 – B7)  
- VGG19 with Batch Normalization  
- MobileNetV2  

These models provide strong hierarchical feature representations and serve as reliable baselines.

---

### Transformer-based Pretrained Encoders
Transformer and hybrid architectures were also explored to model long-range dependencies:

- **MiT (Mix Transformer)** family (B0 – B5)  
- **PVTv2 (Pyramid Vision Transformer)**  

Transformer encoders were integrated using `segmentation_models_pytorch` and `timm`.

---

## Datasets
The experiments were conducted on the following publicly available breast ultrasound datasets:

| Dataset  | Country | Approx. Images | Classes |
|--------|--------|----------------|--------|
| BUSI    | Egypt  | 780            | Benign, Malignant, Normal |
| BUS-BRA | Brazil | 1875           | Benign, Malignant |
| BrEaST  | Poland | 256            | Benign, Malignant, Normal |

**Note:**  
Datasets are **not included** in this repository due to licensing restrictions.  
Users must download the datasets separately and update the dataset paths in the training scripts.

---

## Project Structure

```

breast-ultrasound-pretrained/
├── train_cnn_mit.py          # U-Net with CNN and MiT pretrained encoders
├── train_transformer.py     # U-Net with TIMM transformer encoders
├── requirements.txt
└── README.md

````

---

## Methodology

### Data Splitting
Each dataset is split into:
- 70% Training
- 15% Validation
- 15% Testing  

A fixed random seed is used to ensure reproducibility.

---

### Loss Function
To address class imbalance inherent in ultrasound images, a combined loss function is used:
- **Weighted Binary Cross-Entropy Loss**
- **Dice Loss**

This combination improves foreground segmentation and stabilizes training.

---

### Evaluation Metrics
Model performance is evaluated using:
- Precision  
- Recall  
- F1-score  
- Intersection over Union (IoU)  
- Dice Coefficient  
- Accuracy (reported for completeness)

---

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
````

---

### Step 2: Update Dataset Paths

In **both** `train_cnn_mit.py` and `train_transformer.py`, update the following block:

```python
# =====================================================
# DATASET PATHS (EDIT ONLY THIS BLOCK WHEN SWITCHING DATASETS)
# =====================================================

# -------- BUSI DATASET --------
# IMG_DIR  = "/path/to/BUSI/images"
# MASK_DIR = "/path/to/BUSI/masks"

# -------- BUS-BRA DATASET --------
IMG_DIR  = "/path/to/BUS-BRA/Images"
MASK_DIR = "/path/to/BUS-BRA/Masks"

# -------- BrEaST DATASET --------
# IMG_DIR  = "/path/to/BrEaST/images"
# MASK_DIR = "/path/to/BrEaST/masks"

img_dir = IMG_DIR
mask_dir = MASK_DIR
```

---

### Step 3: Run CNN and MiT Models

```bash
python train_cnn_mit.py
```

---

### Step 4: Run Transformer Models

```bash
python train_transformer.py
```

Training logs and evaluation metrics will be displayed in the console.

---

## Experimental Observations

* CNN-based encoders perform consistently well on BUSI and BUS-BRA datasets.
* Transformer-based models benefit from global context modeling but require sufficient data.
* Performance on the BrEaST dataset is comparatively lower due to limited sample size and higher variability.

---

## Reproducibility

* All experiments use a fixed random seed
* Pretrained ImageNet weights are used for all encoders
* Code is organized for easy reproduction and extension

---

## Limitations

* No extensive hyperparameter tuning
* No post-processing applied to segmentation masks
* Limited training epochs due to computational constraints

These limitations were intentionally accepted to keep the project within a mini-project scope.

---

## Future Work

* Cross-dataset training and generalization
* Lightweight transformer architectures for small datasets
* Semi-supervised learning approaches
* Post-processing for boundary refinement

---

## Acknowledgements

* `segmentation_models_pytorch`
* `timm`
* Authors of the BUSI, BUS-BRA, and BrEaST datasets

---



---


If you want, next I can help you **prepare viva answers** or **justify BrEaST results**.
```

