import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Sequence, List

from .generators import Generator
from .decoders import nnUNetDecoder
from ..config import ALL_MODALITY_NAMES


class SegmentationModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_features: int = 32,
        all_modality_names: Sequence[str] = ALL_MODALITY_NAMES,
        use_meta_generator: bool = True,
        adapter_bottleneck_scale: float = 0.25
    ):
        """
        args:
            in_channels: input image channels
            out_channels: number of output segmentation classes
            base_features: base feature dimension for encoders
            all_modality_names: names of all modalities
            use_meta_generator: enable meta-generator for adaptive adapters
            adapter_bottleneck_scale: bottleneck ratio for adapters
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features
        self.all_modality_names = list(all_modality_names)
        self.use_meta_generator = use_meta_generator

        self.generator = Generator(
            in_channels=in_channels,
            base_features=base_features,
            all_modality_names=self.all_modality_names,
            use_meta_generator=use_meta_generator,
            adapter_bottleneck_scale=adapter_bottleneck_scale
        )

        self.decoder = nnUNetDecoder(
            self.generator.encoder_feature_dims,
            out_channels
        )

    def forward(
        self,
        x_modalities: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Dict]]]:
        """
        forward pass through segmentation model.

        args:
            x_modalities: dict of modality tensors
            modality_mask: optional binary modality presence mask

        returns:
            tuple of:
            - segmentation_logits: output predictions
            - bottleneck_features: encoder bottleneck for discriminator
            - adapter_params: list of adapter dicts if meta_generator enabled
        """
        fused_skips, bottleneck_features = self.generator(
            x_modalities,
            modality_mask=modality_mask
        )

        adapter_params = None
        if self.use_meta_generator:
            adapter_params = self.generator.meta_generator(
                bottleneck_features,
                modality_mask if modality_mask is not None else (
                    self._infer_modality_mask(x_modalities)
                )
            )

        segmentation_logits = self.decoder(fused_skips)

        return segmentation_logits, bottleneck_features, adapter_params

    def _infer_modality_mask(
        self,
        x_modalities: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        infer modality presence mask from input tensors.

        args:
            x_modalities: dict of modality tensors

        returns:
            binary mask of shape (batch, num_modalities)
        """
        batch_size = next(iter(x_modalities.values())).shape[0]
        device = next(iter(x_modalities.values())).device
        num_mods = len(self.all_modality_names)

        mask = torch.zeros(
            batch_size,
            num_mods,
            device=device,
            dtype=torch.float32
        )

        for i, mod_name in enumerate(self.all_modality_names):
            if mod_name in x_modalities:
                is_present = (
                    torch.sum(
                        x_modalities[mod_name],
                        dim=[1, 2, 3, 4]
                    ) > 0
                ).float()
                mask[:, i] = is_present

        return mask
