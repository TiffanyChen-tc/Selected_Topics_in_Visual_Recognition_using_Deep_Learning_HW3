# Selected Topics in Visual Recognition using Deep Learning HW3

**Student ID:** 314553003  
**Name:** Yi-Chien Chen  

## Introduction

This repository contains my implementation for HW3 of *Selected Topics in Visual Recognition using Deep Learning*.  
The task is instance segmentation on colored medical images. The goal is to segment four types of cell instances and generate COCO-style RLE predictions for CodaBench submission.

The method is based on Mask R-CNN with a ResNet-50-FPN backbone. To better handle dense small-cell images, I modify the RPN anchors, increase the proposal and detection limits, and apply crop-based and flip-based test-time augmentation.

Main components:

- Mask R-CNN ResNet-50-FPN
- Small RPN anchors: `(8, 16, 32, 64, 128)`
- Increased detection limit: `detections_per_img = 300`
- Crop TTA with `crop_ratio = 0.6`, `stride_ratio = 0.3`
- Horizontal and vertical flip TTA
- COCO RLE output generation

## Environment Setup

Recommended environment:

- Python 3.9 or higher
- PyTorch
- Torchvision
- NumPy
- tifffile
- pycocotools
- tqdm
- matplotlib

Install dependencies:

```bash
pip install torch torchvision numpy tifffile pycocotools tqdm matplotlib
```

If running on Windows and encountering OpenMP runtime conflicts, the code sets:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

## Project Structure

The expected local project structure is shown below.  
The GitHub repository should contain only the source code files and this README.  
The dataset, checkpoints, and generated submission files are shown for reproducibility, but they should not be uploaded to GitHub.

```text
HW3/
├── dataset.py
├── model.py
├── train.py
├── infer.py
├── README.md
└── hw3-data-release/
```

Important source files:

- `dataset.py`: dataset loading, mask parsing, and training augmentations
- `model.py`: Mask R-CNN model definition and RPN modifications
- `train.py`: training, validation AP50 evaluation, loss curve, and confusion matrix generation
- `infer.py`: validation threshold search, crop/flip TTA inference, and submission JSON generation

## Usage

### 1. Train

Place the dataset under:

```text
hw3-data-release/train
```

Run training:

```bash
python train.py --data_dir hw3-data-release/train --batch_size 2
```

Training outputs are saved under:

```text
checkpoints/
```

Generated files include:

```text
best_model.pth
latest_model.pth
loss_curve.png
confusion_matrix.png
```

### 2. Inference

Run inference with TTA:

```bash
python infer.py --checkpoint checkpoints/best_model.pth
```

The output prediction file will be saved as:

```text
submission/test-results.json
```

## Method Summary

The baseline Mask R-CNN setting is adjusted for dense small-object segmentation.

### RPN Anchor Adjustment

The RPN anchor sizes are changed to:

```python
sizes = ((8,), (16,), (32,), (64,), (128,))
```

The anchor aspect ratios are:

```python
aspect_ratios = ((0.5, 1.0, 2.0),) * 5
```

### Detection Capacity

The model keeps more dense cell predictions by increasing:

```python
detections_per_img = 300
rpn.pre_nms_top_n_train = 4000
rpn.post_nms_top_n_train = 2000
rpn.pre_nms_top_n_test = 2000
rpn.post_nms_top_n_test = 1000
```

### Test-Time Augmentation

The final inference uses:

```text
original image
horizontal flip
vertical flip
overlapping crop TTA
```

Predictions are merged with per-class box NMS.

## Performance Snapshot

Final public leaderboard result:

```text
Public AP50: 0.5891
```

Validation TTA ablation:

| Inference setting | Validation AP50 |
|---|---:|
| No crop TTA | 0.5027 |
| Crop TTA only | 0.5088 |
| Crop TTA + horizontal flip | 0.5127 |
| Crop TTA + horizontal flip + vertical flip | **0.5177** |
| Crop TTA + horizontal flip + vertical flip + HV flip | 0.5163 |
