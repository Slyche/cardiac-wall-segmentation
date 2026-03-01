# Cardiac Structure Segmentation in 2D Echocardiography

Automated segmentation of cardiac structures (left ventricle, myocardium, left atrium) from 2D echocardiographic images using U-Net architectures trained on the [CAMUS dataset](https://humanheart-project.creatis.insa-lyon.fr).

> **Best model:** U-Net with pretrained ResNet34 encoder, SCSE attention, and deep supervision — achieving **0.910 mean foreground Dice** with 5-fold patient-level cross-validation.

![Qualitative Results](results/figures/best_predictions.png)

---

## Highlights

- **Systematic ablation study** across 5 architectures, 6 loss functions, 2 encoders, 2 resolutions, and multiple training strategies (20+ experiments)
- **Honest evaluation** with patient-level splitting (no data leakage) and 5-fold cross-validation
- **Key finding:** performance ceiling of ~0.91 Dice is data-limited, not model-limited — validated by exhaustive experimentation
- **Competitive results:** outperforms nnU-Net benchmarks on CAMUS (0.910 vs ~0.893)

---

## Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [References](#references)

---

## Dataset

**CAMUS** (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) is a publicly available dataset of 2D echocardiographic images with expert annotations.

| Property | Value |
|---|---|
| Patients | 500 |
| Views | 2-chamber (2CH) and 4-chamber (4CH) apical |
| Phases | End-diastole (ED) and end-systole (ES) + half-sequences |
| Total samples | 3000 image-mask pairs (6 per patient) |
| Classes | Background, LV cavity, myocardium, left atrium |
| Format | NIfTI, resized to 256×256 grayscale |

The dataset is available at [humanheart-project.creatis.insa-lyon.fr](https://humanheart-project.creatis.insa-lyon.fr).

---

## Methodology

### Architecture

The best-performing model uses a U-Net with a pretrained ResNet34 encoder from [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch), enhanced with SCSE decoder attention and deep supervision (24.5M parameters).

| Component | Configuration |
|---|---|
| Encoder | ResNet34 (ImageNet pretrained) |
| Decoder | U-Net with SCSE attention |
| Deep supervision | 4 auxiliary heads (weight 0.1 each) |
| Loss | DiceCE (CE=0.3, Dice=0.7) |
| Optimizer | AdamW (encoder LR=3e-5, decoder LR=3e-4, weight decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=1000, eta_min=1e-6) |
| Augmentation | HFlip, affine (±10% scale/shift, ±15° rotation) |
| Input | 256×256 single-channel grayscale |

### Evaluation Protocol

All results use **patient-level 5-fold cross-validation** following nnU-Net conventions (1000 epochs, 250 batches/epoch, patience 100). Images from the same patient never appear in both training and validation sets. Test-time augmentation (horizontal flip) is applied at inference.

---

## Results

### Overall Performance (5-Fold CV, TTA+PP)

| Metric | Value |
|---|---|
| **Dice Score** | **0.9103 ± 0.0024** |
| IoU (Jaccard) | 0.8407 ± 0.0033 |
| Hausdorff Distance | 7.49 ± 0.27 px |
| Mean Absolute Distance | 2.21 ± 0.06 px |

### Per-Class Breakdown

| Class | Dice | IoU | HD (px) | MAD (px) |
|---|---|---|---|---|
| LV Cavity | 0.9436 ± 0.0015 | 0.8949 ± 0.0025 | 6.46 ± 0.17 | 2.00 ± 0.04 |
| Myocardium | 0.8813 ± 0.0015 | 0.7907 ± 0.0022 | 8.33 ± 0.30 | 2.19 ± 0.05 |
| Left Atrium | 0.9059 ± 0.0063 | 0.8365 ± 0.0090 | 7.69 ± 0.38 | 2.45 ± 0.11 |

### Per-Fold Results

| Fold | Dice | IoU | HD (px) | MAD (px) |
|---|---|---|---|---|
| 0 | 0.9137 | 0.8452 | 7.23 | 2.14 |
| 1 | 0.9085 | 0.8380 | 7.80 | 2.24 |
| 2 | 0.9106 | 0.8406 | 7.35 | 2.21 |
| 3 | 0.9117 | 0.8433 | 7.25 | 2.17 |
| 4 | 0.9069 | 0.8363 | 7.83 | 2.30 |

### Per-View and Per-Phase Analysis

| View | Dice | IoU | HD (px) | n |
|---|---|---|---|---|
| 2-Chamber | 0.9078 | 0.8371 | 7.88 | 1500 |
| 4-Chamber | 0.9128 | 0.8442 | 7.10 | 1500 |

| Phase | Dice | IoU | HD (px) | n |
|---|---|---|---|---|
| End-Diastole | 0.9097 | 0.8401 | 7.53 | 1000 |
| End-Systole | 0.9111 | 0.8413 | 7.46 | 1000 |
| Half-Sequence | 0.9100 | 0.8406 | 7.48 | 1000 |

### Confusion Matrix

![Confusion Matrix](results/figures/confusion_matrix.png)

### Comparison with Literature

| Method | Mean FG Dice | Source |
|---|---|---|
| UwU-Net (263K params) | ~0.890 | Filarecki et al., 2025 |
| nnU-Net | ~0.893 | Lightweight ablation, 2025 |
| **Ours (5-fold CV + TTA)** | **0.910** | This work |
| TC-SegNet (ASPP+SE) | 0.928 | 2023 |

---

## Ablation Study

This project followed an extensive experimental journey across 20+ configurations, starting from a plain U-Net baseline and systematically testing architectural changes, loss functions, augmentation strategies, and training techniques.

### What Helped

| Technique | Impact |
|---|---|
| Pretrained encoder (ResNet34) | +0.006 Dice, reduced overfitting gap 0.09 → 0.07 |
| DiceCE loss (30/70 weighting) | Best loss across all experiments |
| Deep supervision | +0.011 on myocardium (hardest class) |
| SCSE decoder attention | Marginal but consistent improvement |
| Light affine augmentation (with pretrained encoder) | Small regularization benefit |
| Test-time augmentation (HFlip) | +0.001 for free |
| Patient-level split | Honest evaluation (−0.007 Dice vs leaked split) |

### What Didn't Help

| Technique | Finding |
|---|---|
| Attention gates (Exp3) | Within noise of plain U-Net |
| Squeeze-and-Excitation blocks (Exp6) | Within noise of plain U-Net |
| Focal Tversky Loss (Exp5, 5b, 5c) | Underperformed DiceCE in every test |
| Boundary loss (Exp7, 7b) | Underperformed DiceCE |
| Stochastic Weight Averaging (Exp8) | No improvement |
| Class weights in CE (v1) | Hurt performance |
| EfficientNet-B4 encoder (v5) | Same accuracy, 2× slower |
| 384×384 resolution (v5) | Same accuracy, 2× slower |
| OneCycleLR / AMP / gradient accumulation (v4) | Zero impact on accuracy |
| Label smoothing (v5) | No benefit |
| 5-fold ensemble (v6) | Didn't break the ceiling |
| 200–300 epochs (v4, v5) | Plateau by epoch 60–80; 100 is sufficient |

<details>
<summary><strong>Click to expand full experiment history (20+ configurations)</strong></summary>

#### Phase 1 — Sample-Level Split (biased due to data leakage)

| Rank | Experiment | Architecture | Loss | Val Dice |
|---|---|---|---|---|
| 1 | Exp1b | UNet | DiceCE 30/70 | 0.9125 |
| 2 | Exp6b | SE-Att-UNet | DiceCE 30/70 | 0.9123 |
| 3 | Exp3 | AttentionUNet | DiceCE 30/70 | 0.9117 |
| 4 | Exp5c | AttentionUNet | FTL+CE | 0.9113 |
| 5 | Exp8 | UNet+SWA | DiceCE 30/70 | 0.9108 |
| 6 | Exp5b | AttentionUNet | FTL γ=1.33 | 0.9106 |
| 7 | Exp7b | UNet | DiceCE+Boundary | 0.9103 |
| 8 | Exp6 | SE-UNet | DiceCE 30/70 | 0.9101 |
| 9 | Exp4 | DeepSup+AttUNet | DiceCE 30/70 | 0.9098 |
| 10 | Baseline | UNet | CrossEntropy | 0.9090 |

*These results are inflated due to sample-level splitting. Included for relative comparison only.*

#### Phase 2 — Patient-Level Split (honest evaluation)

| Rank | Version | Architecture | Val Dice (TTA) |
|---|---|---|---|
| 1 | **v3** | ResNet34 + SCSE + DeepSup | **0.9130** |
| 2 | v4 | v3 + OneCycleLR + AMP | 0.9125 |
| 3 | v5 | EfficientNet-B4, 384² | ~0.9116 |
| 4 | v2 | Custom UNet (clean) | 0.9053 |
| 5 | v1 | Custom UNet + aug + weights | 0.9031 |

</details>

---

## Key Findings

1. **The performance ceiling (~0.91) is data-limited, not model-limited.** After testing 5 architectures, 6 loss functions, 2 encoders, 2 resolutions, and ensemble methods, all converged to the same ceiling. The remaining error lies in inherently ambiguous echocardiographic boundaries, particularly in the thin myocardium.

2. **Pretrained encoders are the single most impactful improvement** — more than any architectural modification, loss function, or training trick.

3. **DiceCE (30/70) is the dominant loss function.** Every alternative (Focal Tversky, boundary loss, and their variants) underperformed without exception.

4. **Methodological rigor matters.** Patient-level splitting and cross-validation are essential for honest reporting in medical imaging. Sample-level splitting inflated our scores by ~0.007.

5. **Myocardium is the hardest structure** (0.881 Dice) — consistent with all published benchmarks on CAMUS. Deep supervision provided the largest improvement for this class (+0.011).

6. **4-chamber views are easier than 2-chamber** (0.913 vs 0.908 Dice), consistent with the literature.

### Failure Case Analysis

![Failure Cases](results/figures/failure_cases.png)

The worst predictions occur on images with poor acoustic windows, extreme cardiac geometries, or ambiguous myocardial boundaries — cases where even expert annotators show high inter-observer variability.

---

## Installation
```bash
git clone https://github.com/Slyche/cardiac-wall-segmentation.git
cd cardiac-wall-segmentation
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- pytorch-lightning
- segmentation-models-pytorch
- albumentations
- nibabel
- opencv-python
- scipy
- scikit-learn

The CAMUS dataset must be downloaded separately from [the official source](https://humanheart-project.creatis.insa-lyon.fr) and placed in the `data/` directory.

---

## Usage

### Training (5-fold cross-validation)

The full training pipeline is in `notebooks/training_cv.ipynb`, designed for Google Colab with A100 GPU.

### Evaluation

The evaluation pipeline with all metrics and visualizations is in `notebooks/evaluation.ipynb`.

### Quick Start
```python
from src.model import DeepSupSMPUNet
from src.dataset import CAMUSDataset

# Load model
model = DeepSupSMPUNet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=1,
    num_classes=4,
)
```

---

## Repository Structure
```
cardiac-wall-segmentation/
├── notebooks/
│   ├── training_cv.ipynb          # 5-fold CV training pipeline
│   └── evaluation.ipynb           # Full evaluation with all metrics
├── src/
│   ├── dataset.py                 # CAMUSDataset + DataModule
│   ├── model.py                   # DeepSupSMPUNet architecture
│   ├── losses.py                  # DiceCE + DeepSupLoss
│   ├── lightning_module.py        # CardiacSegModule
│   └── utils.py                   # Helpers (splits, post-processing)
├── results/
│   └── figures/
│       ├── best_predictions.png
│       ├── failure_cases.png
│       ├── confusion_matrix.png
│       ├── fold_dice_barplot.png
│       └── overlay_results.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## References

- Leclerc et al. (2019). *Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography.* IEEE TMI.
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Kervadec et al. (2019). *Boundary Loss for Highly Unbalanced Segmentation.* Medical Image Analysis.
- Isensee et al. (2021). *nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation.* Nature Methods.
- Cui et al. (2021). *Multiscale Attention Guided U-Net for Cardiac Segmentation.*
- Filarecki et al. (2025). *UwU-Net: Lightweight U-Net for Echocardiography.*

---

## Acknowledgments

This project was developed as a deep learning research study on cardiac image segmentation. The CAMUS dataset was provided by the CREATIS laboratory at INSA Lyon.
