"""Inference: Crop TTA + flip TTA with lazy full-size mask construction.

Memory strategy: crop masks are stored at their original (small) size with
offset metadata. Full-size masks are only constructed for predictions that
survive NMS, avoiding large allocations for discarded candidates.

Usage:
    python infer.py --checkpoint checkpoints/best_model.pth
    python infer.py --checkpoint checkpoints/best_model.pth --no_tta
"""

import argparse
import contextlib
import io
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, random_split
from torchvision.ops import nms
from tqdm import tqdm

from dataset import CellDataset, TestDataset
from model import get_model

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


def build_filename_to_id(json_path: str) -> dict:
    """Parse test_image_name_to_ids.json into a filename-to-id mapping.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Dict mapping file_name strings to integer image ids.
    """
    with open(json_path, "r") as fh:
        records = json.load(fh)
    return {rec["file_name"]: rec["id"] for rec in records}


def collate_fn(batch):
    """Collate variable-size samples into parallel tuples.

    Args:
        batch: List of (image_tensor, file_name) tuples.

    Returns:
        Tuple of (tuple-of-images, tuple-of-filenames).
    """
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------

def get_crop_coords(h, w, crop_ratio=0.6, stride_ratio=0.3):
    """Generate overlapping crop windows for TTA.

    Args:
        h: Image height.
        w: Image width.
        crop_ratio: Crop size as a fraction of the image dimension.
        stride_ratio: Stride as a fraction of the image dimension.

    Returns:
        List of (top, left, crop_h, crop_w) tuples.
    """
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    stride_h = int(h * stride_ratio)
    stride_w = int(w * stride_ratio)

    coords = []
    top = 0
    while True:
        top = min(top, h - crop_h)
        left = 0
        while True:
            left = min(left, w - crop_w)
            coords.append((top, left, crop_h, crop_w))
            if left + crop_w >= w:
                break
            left += stride_w
        if top + crop_h >= h:
            break
        top += stride_h
    return coords


@torch.no_grad()
def infer_single(model, image_tensor, device):
    """Run the model on one image tensor.

    Args:
        model: Mask R-CNN in eval mode.
        image_tensor: Float tensor of shape [3, H, W].
        device: torch.device.

    Returns:
        Raw model output dict for the single image.
    """
    return model([image_tensor.to(device)])[0]


def _build_full_mask(mask_info, h, w, flip_h, flip_v):
    """Build a full-size [1, H, W] binary uint8 mask from stored metadata.

    Args:
        mask_info: Tuple of (crop_mask [1, mh, mw], top, left).
                   top/left are None for full-image predictions.
        h: Original image height.
        w: Original image width.
        flip_h: Whether to flip the mask horizontally.
        flip_v: Whether to flip the mask vertically.

    Returns:
        Binary uint8 tensor of shape [1, H, W].
    """
    crop_mask, top, left = mask_info
    if top is None:
        full = (crop_mask > 0.5).to(torch.uint8)
    else:
        crop_h = crop_mask.shape[-2]
        crop_w = crop_mask.shape[-1]
        full = torch.zeros(1, h, w, dtype=torch.uint8)
        full[:, top:top + crop_h, left:left + crop_w] = (
            (crop_mask > 0.5).to(torch.uint8)
        )

    if flip_h:
        full = torch.flip(full, dims=[2])
    if flip_v:
        full = torch.flip(full, dims=[1])
    return full


def _collect_predictions(model, image_tensor, device, h, w,
                          flip_h=False, flip_v=False,
                          crop_ratio=0.6, stride_ratio=0.3):
    """Collect scores, labels, boxes, and mask metadata for one image variant.

    Masks are stored as small crop-sized tensors with (top, left) offsets.
    Full-size masks are constructed lazily after NMS.

    Args:
        model: Mask R-CNN in eval mode.
        image_tensor: Float tensor of shape [3, H, W].
        device: torch.device.
        h: Original image height.
        w: Original image width.
        flip_h: If True, flip the image horizontally before inference.
        flip_v: If True, flip the image vertically before inference.
        crop_ratio: Crop size fraction.
        stride_ratio: Crop stride fraction.

    Returns:
        Tuple of (scores, labels, boxes, mask_metas, flip_h, flip_v),
        or None if no predictions were found.
    """
    img = image_tensor
    if flip_h:
        img = torch.flip(img, dims=[2])
    if flip_v:
        img = torch.flip(img, dims=[1])

    all_scores, all_labels, all_boxes, all_mask_metas = [], [], [], []

    # Full image.
    out = infer_single(model, img, device)
    if out["scores"].numel() > 0:
        all_scores.append(out["scores"].cpu())
        all_labels.append(out["labels"].cpu())
        boxes = out["boxes"].cpu()
        if flip_h:
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        if flip_v:
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        all_boxes.append(boxes)
        for m in out["masks"].cpu():
            all_mask_metas.append((m, None, None))

    # Overlapping crops.
    for top, left, crop_h, crop_w in get_crop_coords(
        h, w, crop_ratio, stride_ratio
    ):
        crop = img[:, top:top + crop_h, left:left + crop_w]
        out = infer_single(model, crop, device)
        if out["scores"].numel() == 0:
            continue

        all_scores.append(out["scores"].cpu())
        all_labels.append(out["labels"].cpu())

        boxes = out["boxes"].cpu()
        boxes[:, [0, 2]] += left
        boxes[:, [1, 3]] += top
        if flip_h:
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        if flip_v:
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        all_boxes.append(boxes)

        for m in out["masks"].cpu():
            all_mask_metas.append((m, top, left))

    if not all_scores:
        return None

    return (
        torch.cat(all_scores),
        torch.cat(all_labels),
        torch.cat(all_boxes),
        all_mask_metas,
        flip_h,
        flip_v,
    )


def infer_with_tta(model, image_tensor, device, crop_ratio=0.6,
                   stride_ratio=0.3):
    """Run crop TTA + flip TTA and merge predictions with per-class box NMS.

    Variants: original, horizontal flip, vertical flip — each with
    overlapping crops. Full-size masks are only built for kept predictions.

    Args:
        model: Mask R-CNN in eval mode.
        image_tensor: Float tensor of shape [3, H, W].
        device: torch.device.
        crop_ratio: Crop size fraction.
        stride_ratio: Crop stride fraction.

    Returns:
        Dict with keys ``scores``, ``labels``, ``boxes``, ``masks``.
    """
    _, h, w = image_tensor.shape

    all_scores, all_labels, all_boxes = [], [], []
    all_mask_metas = []

    for flip_h, flip_v in [(False, False), (True, False), (False, True)]:
        result = _collect_predictions(
            model, image_tensor, device, h, w,
            flip_h=flip_h, flip_v=flip_v,
            crop_ratio=crop_ratio, stride_ratio=stride_ratio,
        )
        if result is None:
            continue
        scores, labels, boxes, metas, fh, fv = result
        all_scores.append(scores)
        all_labels.append(labels)
        all_boxes.append(boxes)
        for meta in metas:
            all_mask_metas.append((*meta, fh, fv))

    if not all_scores:
        return {
            "scores": torch.tensor([]),
            "labels": torch.tensor([]),
            "boxes": torch.zeros(0, 4),
            "masks": torch.zeros(0, 1, h, w),
        }

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    boxes = torch.cat(all_boxes)

    # Per-class NMS on boxes only — no full-size masks needed here.
    keep_indices = []
    for cls in labels.unique():
        cls_mask = labels == cls
        cls_idx = torch.where(cls_mask)[0]
        kept = nms(boxes[cls_mask], scores[cls_mask], iou_threshold=0.5)
        keep_indices.append(cls_idx[kept])

    if keep_indices:
        keep = torch.cat(keep_indices)
        order = scores[keep].argsort(descending=True)
        keep = keep[order]
    else:
        keep = torch.tensor([], dtype=torch.long)

    # Build full-size masks only for kept predictions.
    kept_masks = []
    for idx in keep.tolist():
        crop_mask, top, left, fh, fv = all_mask_metas[idx]
        full = _build_full_mask((crop_mask, top, left), h, w, fh, fv)
        kept_masks.append(full)

    if kept_masks:
        masks_out = torch.stack(kept_masks)
    else:
        masks_out = torch.zeros(0, 1, h, w, dtype=torch.uint8)

    return {
        "scores": scores[keep],
        "labels": labels[keep],
        "boxes": boxes[keep],
        "masks": masks_out,
    }


# ---------------------------------------------------------------------------
# Validation threshold search
# ---------------------------------------------------------------------------

@torch.no_grad()
def find_best_threshold(model, val_dataset, device, thresholds, use_tta):
    """Search for the score threshold that maximises AP50 on the val set.

    Args:
        model: Trained model in eval mode.
        val_dataset: Validation dataset (no augmentation).
        device: torch.device.
        thresholds: List of candidate score thresholds.
        use_tta: If True, use TTA inference; otherwise single-pass.

    Returns:
        Tuple of (best_threshold, best_ap50).
    """
    model.eval()

    coco_gt_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": f"class{i}"} for i in range(1, 5)
        ],
    }
    all_preds = []
    ann_id = 1

    for img_id, (image, target) in enumerate(
        tqdm(val_dataset, desc="Val inference"), start=1
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

        if use_tta:
            out = infer_with_tta(model, image, device)
        else:
            out = infer_single(model, image, device)
            out = {k: v.cpu() for k, v in out.items()}

        for score, label, box, mask in zip(
            out["scores"], out["labels"], out["boxes"], out["masks"]
        ):
            binary = (mask[0] > 0.5).numpy().astype(np.uint8)
            if binary.sum() == 0:
                continue
            x1, y1, x2, y2 = box.tolist()
            all_preds.append({
                "image_id": img_id,
                "category_id": int(label),
                "segmentation": encode_mask_to_rle(binary),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_data
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    best_threshold, best_ap50 = thresholds[0], 0.0
    print("\nThreshold search on val set:")

    for t in thresholds:
        filtered = [p for p in all_preds if p["score"] >= t]
        if not filtered:
            print(f"  threshold={t:.2f}  AP50=N/A (no predictions)")
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(filtered)
            ev = COCOeval(coco_gt, coco_dt, "segm")
            ev.params.iouThrs = np.array([0.5])
            ev.evaluate()
            ev.accumulate()
            ev.summarize()

        ap50 = float(ev.stats[0])
        marker = " <--" if ap50 > best_ap50 else ""
        print(f"  threshold={t:.2f}  AP50={ap50:.4f}{marker}")

        if ap50 > best_ap50:
            best_ap50 = ap50
            best_threshold = t

    print(f"\nBest threshold : {best_threshold:.2f}  AP50={best_ap50:.4f}")
    return best_threshold, best_ap50


# ---------------------------------------------------------------------------
# Test inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device, filename_to_id,
                  score_threshold, use_tta):
    """Run inference on all test images and collect COCO-format predictions.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader for TestDataset.
        device: torch.device.
        filename_to_id: Dict mapping tif filename to image id.
        score_threshold: Minimum score to include a prediction.
        use_tta: If True, use TTA inference; otherwise single-pass.

    Returns:
        List of prediction dicts in COCO instance-segmentation format.
    """
    model.eval()
    results = []

    for images, file_names in tqdm(loader, desc="Test inference"):
        for image, file_name in zip(images, file_names):
            image_id = filename_to_id.get(file_name)
            if image_id is None:
                tqdm.write(f"[WARN] '{file_name}' not in id mapping, skipped.")
                continue

            if use_tta:
                out = infer_with_tta(model, image, device)
            else:
                out = infer_single(model, image, device)
                out = {k: v.cpu() for k, v in out.items()}

            for score, label, box, mask in zip(
                out["scores"], out["labels"], out["boxes"], out["masks"]
            ):
                if float(score) < score_threshold:
                    continue
                binary_mask = (mask[0] > 0.5).numpy().astype(np.uint8)
                if binary_mask.sum() == 0:
                    continue
                results.append({
                    "image_id": int(image_id),
                    "bbox": [
                        float(box[0]), float(box[1]),
                        float(box[2] - box[0]), float(box[3] - box[1]),
                    ],
                    "score": float(score),
                    "category_id": int(label),
                    "segmentation": encode_mask_to_rle(binary_mask),
                })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments.

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Mask R-CNN inference with TTA"
    )
    parser.add_argument("--test_dir", type=str,
                        default="hw3-data-release/test_release")
    parser.add_argument("--train_dir", type=str,
                        default="hw3-data-release/train")
    parser.add_argument(
        "--id_json", type=str,
        default="hw3-data-release/test_image_name_to_ids.json",
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="submission")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable TTA and run single-pass inference")
    parser.add_argument(
        "--skip_val_search", action="store_true",
        help="Skip threshold search; use --score_threshold directly",
    )
    parser.add_argument("--score_threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    """Main inference entry point."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = not args.no_tta
    print(f"Device : {device}  |  TTA : {'on' if use_tta else 'off'}")
    os.makedirs(args.output_dir, exist_ok=True)

    model = get_model(num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    print(f"Loaded : {args.checkpoint}")

    if args.skip_val_search:
        best_threshold = args.score_threshold
        print(f"Skipping val search. Using threshold={best_threshold:.2f}")
    else:
        base_dataset = CellDataset(root_dir=args.train_dir)
        n_total = len(base_dataset)
        n_val = max(1, int(n_total * args.val_split))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(args.seed)
        _, val_indices = (
            list(s) for s in random_split(
                range(n_total), [n_train, n_val], generator=generator
            )
        )
        val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
        print(f"Val set : {len(val_dataset)} images")

        thresholds = [round(t, 2) for t in np.arange(0.1, 0.96, 0.05)]
        best_threshold, _ = find_best_threshold(
            model, val_dataset, device, thresholds, use_tta
        )

    test_dataset = TestDataset(test_dir=args.test_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    print(f"\nTest images : {len(test_dataset)}")
    print(f"Score threshold : {best_threshold:.2f}")

    filename_to_id = build_filename_to_id(args.id_json)
    results = run_inference(
        model, test_loader, device, filename_to_id, best_threshold, use_tta
    )
    print(f"Total predictions : {len(results)}")

    json_path = os.path.join(args.output_dir, "test-results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh)
    print(f"Saved : {json_path}")


if __name__ == "__main__":
    main()