import os
import random
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset
from typing import Tuple, Dict, Sequence

from .preprocessing import preprocess_spatial, create_multilabel_seg_masks
from ..config import ALL_MODALITY_NAMES, TARGET_SHAPE


class BraTSDataset(Dataset):
    """
    loads multi-modal mri scans and ground truth segmentation,
    applies preprocessing, and returns either full-modality
    or partial-modality samples based on mode.
    """
    def __init__(
        self,
        data_dir: str,
        modality_names: Sequence[str] = ALL_MODALITY_NAMES,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        mode: str = "train",
        partial_modality_prob: float = 0.5,
        num_partial_modalities: int = 3
    ):
        """
        args:
            data_dir: path to brats dataset directory
            modality_names: names of modalities in order
            target_shape: target spatial dimensions
            mode: 'train', 'val', or 'test'
            partial_modality_prob: probability of partial modality
            num_partial_modalities: num modalities to keep if partial
        """
        super().__init__()
        self.data_dir = data_dir
        self.modality_names = list(modality_names)
        self.target_shape = target_shape
        self.mode = mode
        self.partial_modality_prob = partial_modality_prob
        self.num_partial_modalities = num_partial_modalities

        self.scans = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

    def __len__(self) -> int:
        """return number of scans in dataset."""
        return len(self.scans)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        args:
            idx: sample index
        
        returns:
            tuple of:
            - modality_dict: dict of modality tensors
            - modality_mask: binary mask of present modalities
            - segmentation: binary segmentation mask
        """
        scan_dir = os.path.join(self.data_dir, self.scans[idx])
        
        modality_dict = {}
        for mod_name in self.modality_names:
            mod_path = os.path.join(
                scan_dir,
                f"{self.scans[idx]}_{mod_name}.nii.gz"
            )
            
            if os.path.exists(mod_path):
                nifti_obj = nib.load(mod_path)
                volume = np.asarray(nifti_obj.dataobj)
                
                volume = preprocess_spatial(
                    volume,
                    self.target_shape
                )
                
                v_min = np.min(volume)
                v_max = np.max(volume)
                if v_max > v_min:
                    volume = (volume - v_min) / (v_max - v_min)
                
                volume = torch.from_numpy(
                    volume[np.newaxis, ...]
                ).float()
                
                modality_dict[mod_name] = volume

        seg_path = os.path.join(
            scan_dir,
            f"{self.scans[idx]}_seg.nii.gz"
        )
        seg_volume = np.asarray(nib.load(seg_path).dataobj)
        seg_volume = preprocess_spatial(
            seg_volume,
            self.target_shape
        )
        seg_mask = create_multilabel_seg_masks(seg_volume)
        seg_mask = torch.from_numpy(seg_mask).float()

        num_mods = len(self.modality_names)
        modality_mask = torch.zeros(num_mods, dtype=torch.float32)
        
        for i, mod_name in enumerate(self.modality_names):
            if mod_name in modality_dict:
                modality_mask[i] = 1.0

        if random.random() < self.partial_modality_prob:
            present_mods = [
                i for i, m in enumerate(modality_mask)
                if m > 0
            ]
            
            if len(present_mods) > self.num_partial_modalities:
                keep_idxs = random.sample(
                    present_mods,
                    self.num_partial_modalities
                )
                
                for i, mod_name in enumerate(self.modality_names):
                    if i not in keep_idxs:
                        modality_dict.pop(mod_name, None)
                        modality_mask[i] = 0.0

        return modality_dict, modality_mask, seg_mask
