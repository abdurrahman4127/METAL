#!/usr/bin/env python3

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from src.models import SegmentationModel, ModalityDiscriminator
from src.data import BraTSDataset
from src.config import (
    BATCH_SIZE_DF, NUM_WORKERS, BASE_FEATURES, BOTTLENECK_SCALE,
    INPUT_CHANNELS, OUTPUT_CHANNELS, ALL_MODALITY_NAMES, DISCRIMINATOR_HIDDEN_DIM
)


def evaluate(model, dataloader, device):
    """
    evaluate segmentation model on dataset.
    
    args:
        model: segmentation model
        dataloader: evaluation data loader
        device: torch device
    
    returns:
        dict of metrics
    """
    model.eval()
    metrics = {
        'WT': {'Dice': [], 'Accuracy': [], 'Precision': [],
              'Recall': [], 'F1': [], 'Sensitivity': [], 'Specificity': []},
        'TC': {'Dice': [], 'Accuracy': [], 'Precision': [],
              'Recall': [], 'F1': [], 'Sensitivity': [], 'Specificity': []},
        'ET': {'Dice': [], 'Accuracy': [], 'Precision': [],
              'Recall': [], 'F1': [], 'Sensitivity': [], 'Specificity': []}
    }

    with torch.no_grad():
        for batch_data, _, seg_gt in tqdm(dataloader, desc="Evaluating"):
            if seg_gt is None:
                continue

            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            seg_gt = seg_gt.to(device)

            seg_logits, _, _ = model(batch_data)
            seg_pred = (torch.sigmoid(seg_logits) > 0.5).float()

            if seg_pred.shape[2:] != seg_gt.shape[2:]:
                seg_gt = F.interpolate(
                    seg_gt,
                    size=seg_pred.shape[2:],
                    mode='nearest'
                )

            for i, name in enumerate(['WT', 'TC', 'ET']):
                pred = seg_pred[:, i]
                gt = seg_gt[:, i]

                intersection = (pred * gt).sum()
                union = pred.sum() + gt.sum()
                dice = (2 * intersection) / (union + 1e-7)
                metrics[name]['Dice'].append(dice.item())

                pred = pred.flatten()
                gt = gt.flatten()
                tp = (pred * gt).sum()
                fp = (pred * (1 - gt)).sum()
                tn = ((1 - pred) * (1 - gt)).sum()
                fn = ((1 - pred) * gt).sum()

                eps = 1e-7
                metrics[name]['Accuracy'].append(
                    ((tp + tn) / (tp + tn + fp + fn + eps)).item()
                )
                metrics[name]['Precision'].append(
                    (tp / (tp + fp + eps)).item()
                )
                metrics[name]['Recall'].append(
                    (tp / (tp + fn + eps)).item()
                )
                metrics[name]['F1'].append(
                    (2 * tp / (2 * tp + fp + fn + eps)).item()
                )
                metrics[name]['Sensitivity'].append(
                    (tp / (tp + fn + eps)).item()
                )
                metrics[name]['Specificity'].append(
                    (tn / (tn + fp + eps)).item()
                )

    final_metrics = {}
    for name in metrics:
        final_metrics[name] = {
            'Dice': np.mean(metrics[name]['Dice']),
            'Accuracy': np.mean(metrics[name]['Accuracy']),
            'Precision': np.mean(metrics[name]['Precision']),
            'Recall': np.mean(metrics[name]['Recall']),
            'F1': np.mean(metrics[name]['F1']),
            'Sensitivity': np.mean(metrics[name]['Sensitivity']),
            'Specificity': np.mean(metrics[name]['Specificity'])
        }

    return final_metrics


def main():
    """run evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Using device: {device}")

    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return

    dataset_dir = "/path/to/brats2020"
    if not os.path.exists(dataset_dir):
        print(f"[Error] Dataset directory not found: {dataset_dir}")
        return

    print("[Data] Loading evaluation dataset...")
    dataset = BraTSDataset(
        data_dir=dataset_dir,
        modality_names=ALL_MODALITY_NAMES,
        mode="val"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_DF,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("[Models] Initializing segmentation model...")
    segmentation_model = SegmentationModel(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        base_features=BASE_FEATURES,
        all_modality_names=ALL_MODALITY_NAMES,
        use_meta_generator=True,
        adapter_bottleneck_scale=BOTTLENECK_SCALE
    )

    print(f"[Load] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    segmentation_model.load_state_dict(checkpoint["model"])
    segmentation_model = segmentation_model.to(device)

    print("[Evaluation] Computing metrics...")
    metrics = evaluate(segmentation_model, dataloader, device)

    print("\n[Results] Segmentation metrics:")
    for region, values in metrics.items():
        print(f"\n{region}:")
        for metric_name, metric_val in values.items():
            print(f"  {metric_name}: {metric_val:.4f}")


if __name__ == "__main__":
    main()
