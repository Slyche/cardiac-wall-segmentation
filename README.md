\# Cardiac Structure Segmentation in 2D Echocardiography



Automatic segmentation of cardiac structures in echocardiographic images using a U-Net deep learning architecture, trained on the public CAMUS dataset.



\## Motivation



Left ventricular segmentation is a critical first step in quantifying cardiac function. Manual delineation by clinicians is time-consuming and subject to inter-observer variability. This project implements and progressively improves an automated deep learning pipeline to segment four cardiac structures from 2D echocardiographic images, directly addressing the need for robust automated analysis in clinical workflows.



\## Dataset



CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)

\- 500 patients, 2-chamber and 4-chamber apical views

\- End-diastole (ED) and end-systole (ES) frames per patient

\- 2000 labeled samples total: 1600 train / 400 validation

\- 4 classes: background, left ventricle, myocardium, left atrium

\- Publicly available at: https://humanheart-project.creatis.insa-lyon.fr



\## Architecture



Standard U-Net with 4 encoder/decoder levels:

\- Double convolution blocks with batch normalization and ReLU

\- Max pooling for downsampling

\- Transposed convolutions for upsampling

\- Skip connections at each level

\- Final 1x1 convolution outputting 4-class segmentation map

\- Input size: 256x256, single channel grayscale



\## Ablation Study



All experiments trained for 20 epochs, batch size 4, Adam optimizer, learning rate 1e-4, CPU only.



| Experiment | Loss | Augmentation | Dropout | Val Dice |

|---|---|---|---|---|

| Baseline | CrossEntropy | None | No | 0.9096 |



\## Qualitative Results



!\[Predictions](results/predictions.png)



Top row: input echocardiographic images. Middle row: ground truth masks. Bottom row: model predictions.

Colors: cyan = left ventricle, yellow = myocardium, dark red = left atrium.



\## Repository Structure



cardiac-wall-segmentation/

├── data/                        # CAMUS dataset (not tracked)

├── notebooks/

│   └── exploration.ipynb        # Full training and evaluation notebook

├── src/

│   ├── dataset.py

│   ├── model.py

│   ├── train.py

│   └── predict.py

├── results/

│   └── predictions.png

├── requirements.txt

└── README.md





\## References



\- Leclerc et al. (2019). Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography. IEEE Transactions on Medical Imaging.

\- Kim et al. (2021). Automatic segmentation of the left ventricle in echocardiographic images using convolutional neural networks. Quantitative Imaging in Medicine and Surgery.

\- Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.

