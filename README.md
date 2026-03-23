# Cardiac Structure Segmentation in 2D Echocardiography

Cardiac structure segmentation (LV, myocardium, left atrium) from 2D echo, trained on the [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/) with a U-Net + ResNet34 encoder.

> **0.910 mean foreground Dice** with 5-fold patient-level cross-validation — outperforming nnU-Net (~0.893) on CAMUS through systematic experimentation across 20+ configurations.

![Qualitative Results](results/figures/best_predictions.png)

## Approach

U-Net with pretrained ResNet34 encoder, SCSE decoder attention, and deep supervision (24.5M parameters). Trained with DiceCE loss (CE=0.3, Dice=0.7), AdamW with differential learning rates (encoder 3e-5, decoder 3e-4), and CosineAnnealingLR. Evaluation follows nnU-Net conventions (1000 epochs, 250 batches/epoch, patience 100). Test-time augmentation is applied at inference.

All results use **patient-level 5-fold cross-validation** — images from the same patient never appear in both training and validation sets.

## Results

| Method | Mean FG Dice |
|---|---|
| UwU-Net (263K params) | ~0.890 |
| nnU-Net | ~0.893 |
| **Ours (5-fold CV + TTA)** | **0.910** |
| TC-SegNet (ASPP+SE) | 0.928 |

| Class | Dice | IoU | HD (px) | MAD (px) |
|---|---|---|---|---|
| LV Cavity | 0.944 ± 0.002 | 0.895 ± 0.003 | 6.46 ± 0.17 | 2.00 ± 0.04 |
| Myocardium | 0.881 ± 0.002 | 0.791 ± 0.002 | 8.33 ± 0.30 | 2.19 ± 0.05 |
| Left Atrium | 0.906 ± 0.006 | 0.837 ± 0.009 | 7.69 ± 0.38 | 2.45 ± 0.11 |

## What Worked (and What Didn't)

**Helped:** Pretrained encoder (+0.006 Dice), DiceCE loss (best across all experiments), deep supervision (+0.011 on myocardium), SCSE attention, patient-level splitting, test-time augmentation.

**Didn't help:** Attention gates, SE blocks, Focal Tversky loss, boundary loss, SWA, class weights, EfficientNet-B4 encoder, 384×384 resolution, OneCycleLR, label smoothing, 5-fold ensemble. All converged to the same ~0.91 ceiling — the bottleneck is data, not model capacity.

![Failure Cases](results/figures/failure_cases.png)

Worst predictions occur on images with poor acoustic windows, extreme cardiac geometries, or ambiguous myocardial boundaries.

## Dataset

[CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/): 500 patients, 2CH/4CH apical views, ED/ES phases + half-sequences. 3000 image-mask pairs, 4 classes, NIfTI format resized to 256×256. Must be downloaded separately.

## Installation
```bash
git clone https://github.com/Slyche/cardiac-wall-segmentation.git
cd cardiac-wall-segmentation
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch 2.0+, and a GPU.

## Usage

**Training** (5-fold CV): `notebooks/training_cv.ipynb` — designed for Google Colab with A100 GPU.

**Evaluation**: `notebooks/evaluation.ipynb` — full metrics, visualizations, and comparison figures.
```python
from src.model import DeepSupSMPUNet

model = DeepSupSMPUNet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=1,
    num_classes=4,
)
```

## References

- Leclerc et al. (2019). *Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography.* IEEE TMI.
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Isensee et al. (2021). *nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation.* Nature Methods.
- Kervadec et al. (2019). *Boundary Loss for Highly Unbalanced Segmentation.* Medical Image Analysis.

## Acknowledgments

The CAMUS dataset was provided by the CREATIS laboratory at INSA Lyon.
