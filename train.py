"""Train Mask R-CNN for cell instance segmentation.

Usage:
    python train.py --data_dir hw3-data-release/train --batch_size 2
"""

import argparse
import contextlib
import io
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import CellDataset, get_train_transforms
from model import count_parameters, get_model

NUM_CLASSES = 4  # cell classes (labels 1-4); 0 is background


def collate_fn(batch):
    """Stack variable-length samples into parallel tuples.

    Args:
        batch: List of (image_tensor, target_dict) tuples.

    Returns:
        Tuple of (tuple-of-images, tuple-of-targets).
    """
    return tuple(zip(*batch))


def encode_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Encode a binary mask as a COCO compressed RLE dict.

    Args:
        binary_mask: Boolean or uint8 array of shape (H, W).

    Returns:
        Dict with keys ``size`` (list[int]) and ``counts`` (str).
    """
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, loader, device, epoch):
    """Run one training epoch.

    Args:
        model: Mask R-CNN model in training mode.
        optimizer: Parameter optimiser.
        loader: Training DataLoader.
        device: torch.device to move tensors to.
        epoch: Current epoch number (for logging).

    Returns:
        Mean training loss for this epoch.
    """
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += total_loss.item()
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    return running_loss / len(loader)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val_loss(model, loader, device):
    """Compute mean validation loss.

    The model is kept in train mode so Torchvision returns loss dicts
    instead of predictions.

    Args:
        model: Mask R-CNN model.
        loader: Validation DataLoader.
        device: torch.device.

    Returns:
        Mean validation loss.
    """
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="           [val loss]", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        running_loss += sum(model(images, targets).values()).item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate_ap50(model, val_dataset, device, score_threshold=0.05):
    """Compute AP50 on the validation set using pycocotools COCOeval.

    Args:
        model: Mask R-CNN model in eval mode.
        val_dataset: Validation dataset (no augmentation).
        device: torch.device.
        score_threshold: Minimum score to include a prediction.

    Returns:
        AP50 score as a float.
    """
    model.eval()

    coco_gt_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": f"class{i}"} for i in range(1, NUM_CLASSES + 1)
        ],
    }
    predictions = []
    ann_id = 1

    for img_id, (image, target) in enumerate(
        tqdm(val_dataset, desc="           [val AP50]", leave=False), start=1
    ):
        _, h, w = image.shape
        coco_gt_data["images"].append({"id": img_id, "height": h, "width": w})

        for box, label, mask in zip(
            target["boxes"], target["labels"], target["masks"]
        ):
            binary = mask.numpy().astype(np.uint8)
            x1, y1, x2, y2 = box.tolist()
            coco_gt_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "segmentation": encode_mask_to_rle(binary),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float(binary.sum()),
                "iscrowd": 0,
            })
            ann_id += 1

        output = model([image.to(device)])[0]
        for score, label, box, mask in zip(
            output["scores"].cpu(), output["labels"].cpu(),
            output["boxes"].cpu(), output["masks"].cpu(),
        ):
            if float(score) < score_threshold:
                continue
            binary = (mask[0] > 0.5).numpy().astype(np.uint8)
            if binary.sum() == 0:
                continue
            x1, y1, x2, y2 = box.tolist()
            predictions.append({
                "image_id": img_id,
                "category_id": int(label),
                "segmentation": encode_mask_to_rle(binary),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

    if not predictions:
        return 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_data
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions)
        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.params.iouThrs = np.array([0.5])
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    return float(ev.stats[0])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses, val_losses, val_ap50s, output_dir):
    """Save a dual-axis loss curve (train/val loss + val AP50).

    Args:
        train_losses: List of per-epoch train losses.
        val_losses: List of per-epoch val losses.
        val_ap50s: List of per-epoch val AP50 scores.
        output_dir: Directory to write ``loss_curve.png``.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(epochs, train_losses, label="Train loss", color="tab:blue")
    ax1.plot(epochs, val_losses, label="Val loss", color="tab:green")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_ap50s, label="Val AP50",
             color="tab:orange", linestyle="--")
    ax2.set_ylabel("Val AP50", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Train / Val Loss & Val AP50")
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()


@torch.no_grad()
def compute_confusion_matrix(model, val_dataset, device):
    """Match predictions to GT (IoU >= 0.5) and build a confusion matrix.

    Rows = GT class (0 = unmatched FN).
    Cols = Predicted class (0 = unmatched FP).

    Args:
        model: Mask R-CNN model in eval mode.
        val_dataset: Validation dataset.
        device: torch.device.

    Returns:
        Integer numpy array of shape (NUM_CLASSES+1, NUM_CLASSES+1).
    """
    model.eval()
    matrix = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)

    for image, target in tqdm(val_dataset, desc="Confusion matrix", leave=False):
        gt_labels = target["labels"].numpy()
        gt_masks = target["masks"].numpy().astype(np.uint8)
        output = model([image.to(device)])[0]
        pred_labels = output["labels"].cpu().numpy()
        pred_masks = (output["masks"].cpu().numpy()[:, 0] > 0.5).astype(np.uint8)

        matched_gt = set()
        for p_idx in range(len(pred_labels)):
            best_iou, best_g = 0.5, -1
            for g_idx in range(len(gt_labels)):
                if g_idx in matched_gt:
                    continue
                inter = (pred_masks[p_idx] & gt_masks[g_idx]).sum()
                union = (pred_masks[p_idx] | gt_masks[g_idx]).sum()
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_g = iou, g_idx
            if best_g >= 0:
                matched_gt.add(best_g)
                matrix[gt_labels[best_g], pred_labels[p_idx]] += 1
            else:
                matrix[0, pred_labels[p_idx]] += 1  # FP

        for g_idx in range(len(gt_labels)):
            if g_idx not in matched_gt:
                matrix[gt_labels[g_idx], 0] += 1  # FN

    return matrix


def plot_confusion_matrix(matrix, output_dir):
    """Save a colour-coded confusion matrix image.

    Args:
        matrix: Integer array of shape (NUM_CLASSES+1, NUM_CLASSES+1).
        output_dir: Directory to write ``confusion_matrix.png``.
    """
    labels = ["BG"] + [f"class{i}" for i in range(1, NUM_CLASSES + 1)]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES + 1))
    ax.set_yticks(range(NUM_CLASSES + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("GT class")
    ax.set_title("Confusion Matrix (IoU >= 0.5)")
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center", color=color, fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved : {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments.

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="Train Mask R-CNN")
    parser.add_argument("--data_dir", type=str,
                        default="hw3-data-release/train")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: '{args.data_dir}'")
    os.makedirs(args.output_dir, exist_ok=True)

    base_dataset = CellDataset(root_dir=args.data_dir)
    n_total = len(base_dataset)
    if n_total == 0:
        raise RuntimeError(
            f"No image folders found inside '{args.data_dir}'."
        )

    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = (
        list(s) for s in random_split(
            range(n_total), [n_train, n_val], generator=generator
        )
    )
    train_set = torch.utils.data.Subset(
        CellDataset(
            root_dir=args.data_dir, transforms=get_train_transforms()
        ),
        train_indices,
    )
    val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
    print(f"Dataset : {n_total} total  ->  {n_train} train / {n_val} val")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = get_model(num_classes=5, pretrained=True)
    model.to(device)
    print(f"Trainable parameters : {count_parameters(model):,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params, lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[15, 25], gamma=0.1
    )

    best_ap50 = 0.0
    train_losses, val_losses, val_ap50s = [], [], []
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training")

    for epoch in epoch_pbar:
        t0 = time.time()
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        val_loss = evaluate_val_loss(model, val_loader, device)
        val_ap50 = evaluate_ap50(model, val_dataset, device)
        scheduler.step()
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ap50s.append(val_ap50)

        plot_loss_curve(train_losses, val_losses, val_ap50s, args.output_dir)

        lr = optimizer.param_groups[0]["lr"]
        epoch_pbar.set_postfix(AP50=f"{val_ap50:.4f}", lr=f"{lr:.1e}")

        torch.save(
            model.state_dict(),
            os.path.join(args.output_dir, "latest_model.pth"),
        )
        is_best = val_ap50 > best_ap50
        if is_best:
            best_ap50 = val_ap50
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best_model.pth"),
            )

        tag = " (*)" if is_best else "    "
        tqdm.write(
            f"Epoch [{epoch:03d}/{args.epochs}]{tag}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_AP50={val_ap50:.4f}  lr={lr:.1e}  {elapsed:.0f}s"
        )

    print(f"Training complete. Best val AP50 : {best_ap50:.4f}")
    print(f"Loss curve : {os.path.join(args.output_dir, 'loss_curve.png')}")

    print("Computing confusion matrix on val set ...")
    model.load_state_dict(
        torch.load(
            os.path.join(args.output_dir, "best_model.pth"),
            map_location=device,
        )
    )
    matrix = compute_confusion_matrix(model, val_dataset, device)
    plot_confusion_matrix(matrix, args.output_dir)


if __name__ == "__main__":
    main()