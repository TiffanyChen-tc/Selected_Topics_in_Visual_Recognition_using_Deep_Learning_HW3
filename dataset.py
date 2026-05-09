"""Dataset classes for cell instance segmentation."""

import os
import random

import numpy as np
import tifffile
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

CLASS_NAMES = ["class1", "class2", "class3", "class4"]


def _load_tif_as_rgb_uint8(path: str) -> np.ndarray:
    """Load a .tif file and return an RGB uint8 array of shape (H, W, 3).

    Args:
        path: Path to the .tif file.

    Returns:
        RGB image array with dtype uint8.
    """
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        max_val = img.max()
        img = (
            (img / max_val * 255).astype(np.uint8)
            if max_val > 0
            else img.astype(np.uint8)
        )
    return img


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

class RandomHorizontalFlip:
    """Randomly flip image and masks horizontally."""

    def __init__(self, prob: float = 0.5):
        """Args:
            prob: Probability of applying the flip.
        """
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, h, w = image.shape
            image = TF.hflip(image)
            target["masks"] = TF.hflip(target["masks"])
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


class RandomVerticalFlip:
    """Randomly flip image and masks vertically."""

    def __init__(self, prob: float = 0.5):
        """Args:
            prob: Probability of applying the flip.
        """
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, h, w = image.shape
            image = TF.vflip(image)
            target["masks"] = TF.vflip(target["masks"])
            boxes = target["boxes"].clone()
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            target["boxes"] = boxes
        return image, target


class RandomRotation90:
    """Randomly rotate image and masks by 0, 90, 180, or 270 degrees."""

    def __call__(self, image, target):
        k = random.randint(0, 3)
        if k == 0:
            return image, target

        image = torch.rot90(image, k, dims=[1, 2])
        target["masks"] = torch.rot90(target["masks"], k, dims=[1, 2])

        # Recompute boxes from rotated masks.
        new_boxes = []
        for mask in target["masks"]:
            m = mask.numpy()
            rows, cols = m.any(axis=1), m.any(axis=0)
            if rows.any() and cols.any():
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                new_boxes.append(
                    [float(x_min), float(y_min), float(x_max), float(y_max)]
                )
            else:
                new_boxes.append([0.0, 0.0, 1.0, 1.0])

        if new_boxes:
            target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

        return image, target


class RandomCropResize:
    """Randomly crop a region then resize back to the original size."""

    def __init__(self, min_scale: float = 0.7, prob: float = 0.5):
        """Args:
            min_scale: Minimum crop scale relative to image size.
            prob: Probability of applying the transform.
        """
        self.min_scale = min_scale
        self.prob = prob

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        _, h, w = image.shape
        scale = random.uniform(self.min_scale, 1.0)
        crop_h, crop_w = int(h * scale), int(w * scale)
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        image = image[:, top:top + crop_h, left:left + crop_w]
        image = TF.resize(image, [h, w], antialias=True)

        masks = target["masks"][:, top:top + crop_h, left:left + crop_w]
        if masks.shape[0] > 0:
            masks = TF.resize(
                masks, [h, w], interpolation=TF.InterpolationMode.NEAREST
            )
        target["masks"] = masks

        new_boxes = []
        for mask in target["masks"]:
            m = mask.numpy()
            rows, cols = m.any(axis=1), m.any(axis=0)
            if rows.any() and cols.any():
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                new_boxes.append(
                    [float(x_min), float(y_min), float(x_max), float(y_max)]
                )
            else:
                new_boxes.append([0.0, 0.0, 1.0, 1.0])
        target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)

        return image, target


class FilterDegenerateBoxes:
    """Remove instances whose bounding box has zero width or height."""

    def __call__(self, image, target):
        boxes = target["boxes"]
        if boxes.shape[0] == 0:
            return image, target
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        target["boxes"] = boxes[keep]
        target["masks"] = target["masks"][keep]
        target["labels"] = target["labels"][keep]
        return image, target


class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms):
        """Args:
            transforms: List of callable transforms.
        """
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_train_transforms():
    """Return the standard training augmentation pipeline.

    Returns:
        A Compose transform with flip, rotation, and degenerate-box filtering.
    """
    return Compose([
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.5),
        RandomRotation90(),
        FilterDegenerateBoxes(),
    ])


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class CellDataset(Dataset):
    """Training/validation dataset for cell instance segmentation.

    Each item returns ``(image_tensor, target)`` where target is a dict with:

    - ``boxes``:  FloatTensor[N, 4] — (x_min, y_min, x_max, y_max)
    - ``masks``:  UInt8Tensor[N, H, W]
    - ``labels``: Int64Tensor[N] — 1-indexed class id
    """

    def __init__(self, root_dir: str, transforms=None):
        """Args:
            root_dir: Path to the train directory.
            transforms: Optional callable applied as transforms(image, target).
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

    def __len__(self) -> int:
        return len(self.image_dirs)

    def __getitem__(self, idx: int):
        image_dir = os.path.join(self.root_dir, self.image_dirs[idx])
        image_tensor = TF.to_tensor(
            _load_tif_as_rgb_uint8(os.path.join(image_dir, "image.tif"))
        )

        boxes, masks, labels = [], [], []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            mask_path = os.path.join(image_dir, f"{class_name}.tif")
            if not os.path.exists(mask_path):
                continue

            mask = tifffile.imread(mask_path)
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            for inst_id in np.unique(mask):
                if inst_id == 0:
                    continue
                binary = (mask == inst_id).astype(np.uint8)
                rows, cols = binary.any(axis=1), binary.any(axis=0)
                if not rows.any() or not cols.any():
                    continue
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                if x_max <= x_min or y_max <= y_min:
                    continue
                boxes.append(
                    [float(x_min), float(y_min), float(x_max), float(y_max)]
                )
                masks.append(binary)
                labels.append(class_idx + 1)

        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "masks": torch.tensor(np.array(masks), dtype=torch.uint8),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        else:
            h, w = image_tensor.shape[1], image_tensor.shape[2]
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "labels": torch.zeros(0, dtype=torch.int64),
            }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)
        return image_tensor, target


class TestDataset(Dataset):
    """Test dataset that returns ``(image_tensor, file_name)`` pairs."""

    def __init__(self, test_dir: str):
        """Args:
            test_dir: Path to the test_release directory.
        """
        self.test_dir = test_dir
        self.image_files = sorted(
            f for f in os.listdir(test_dir) if f.endswith(".tif")
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        file_name = self.image_files[idx]
        image = _load_tif_as_rgb_uint8(os.path.join(self.test_dir, file_name))
        return TF.to_tensor(image), file_name