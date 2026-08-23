"""
preprocessing utilities for 3d medical imaging data.
"""

import numpy as np
import torch
from typing import Tuple
import nibabel as nib


def preprocess_spatial(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """
    center crop or pad 3d volume to target shape.
    
    args:
        volume: 3d input array of shape (d, h, w)
        target_shape: desired output shape (d, h, w)
    
    returns:
        preprocessed volume of target shape
    """
    current_shape = volume.shape
    result = np.zeros(target_shape, dtype=volume.dtype)
    
    slices_src = []
    slices_dst = []
    for i, (current, target) in enumerate(zip(current_shape, target_shape)):
        if current >= target:
            start = (current - target) // 2
            slices_src.append(slice(start, start + target))
            slices_dst.append(slice(0, target))
        else:
            start = (target - current) // 2
            slices_src.append(slice(0, current))
            slices_dst.append(slice(start, start + current))
    
    result[tuple(slices_dst)] = volume[tuple(slices_src)]
    return result


def create_multilabel_seg_masks(
    seg_array: np.ndarray
) -> np.ndarray:
    """
    convert brats integer segmentation labels to 3-channel binary masks.
    
    channel 0: whole tumor (label >= 1)
    channel 1: tumor core (label 1 or 3)
    channel 2: enhancing tumor (label 4)
    
    args:
        seg_array: segmentation array with labels [0, 1, 2, 3, 4]
    
    returns:
        binary mask array of shape (3, d, h, w)
    """
    wt = (seg_array >= 1).astype(np.float32)
    tc = (
        ((seg_array == 1) | (seg_array == 3)).astype(np.float32)
    )
    et = (seg_array == 4).astype(np.float32)
    
    return np.stack([wt, tc, et], axis=0)


def load_and_preprocess_nifti(
    modality_path: str,
    target_shape: Tuple[int, int, int] = (128, 128, 128)
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    load and preprocess single nifti file.
    
    args:
        modality_path: path to nifti file
        target_shape: target spatial dimensions
    
    returns:
        tuple of preprocessed tensor and original volume
    """
    nifti_obj = nib.load(modality_path)
    volume = np.asarray(nifti_obj.dataobj)
    
    volume_processed = preprocess_spatial(volume, target_shape=target_shape)
    
    v_min = np.min(volume_processed)
    v_max = np.max(volume_processed)
    if v_max > v_min:
        volume_normalized = (
            (volume_processed - v_min) / (v_max - v_min)
        )
    else:
        volume_normalized = volume_processed
    
    tensor = torch.from_numpy(
        volume_normalized[np.newaxis, ...]
    ).float()
    
    return tensor, volume
