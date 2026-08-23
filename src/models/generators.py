import torch
import torch.nn as nn
from typing import Dict, List, Sequence, Optional, Tuple

from .encoders import nnUNetEncoder
from ..config import ALL_MODALITY_NAMES


class PerModalityEncoders(nn.Module):
    """
    dedicated encoder for each modality.
    
    processes each modality independently, returning features
    and skip connections for each. handles missing modalities
    with zero tensors.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 32,
        modality_names: Sequence[str] = ALL_MODALITY_NAMES
    ):
        """
        args:
            in_channels: channels per modality
            base_features: base feature dimension
            modality_names: list of modality names
        """
        super().__init__()
        self.modality_names = list(modality_names)
        self.base_features = base_features
        
        self.encoders = nn.ModuleDict({
            mod_name: nnUNetEncoder(in_channels, base_features)
            for mod_name in self.modality_names
        })
        
        self.encoder_feature_dims = [
            base_features,
            base_features * 2,
            base_features * 4,
            base_features * 8,
            base_features * 16
        ]

    def forward(
        self,
        x_modalities: Dict[str, torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        encode each modality independently.
        
        args:
            x_modalities: dict mapping modality name to tensor
        
        returns:
            dict mapping modality name to list of skip connections
        """
        modality_features = {}
        
        for mod_name in self.modality_names:
            if mod_name in x_modalities:
                skips = self.encoders[mod_name](x_modalities[mod_name])
                modality_features[mod_name] = skips
            else:
                batch_size = (
                    next(iter(x_modalities.values())).shape[0]
                )
                device = (
                    next(iter(x_modalities.values())).device
                )
                
                skips = [
                    torch.zeros(
                        batch_size,
                        self.encoder_feature_dims[i],
                        *[(128 // (2 ** i))] * 3,
                        device=device,
                        dtype=torch.float32
                    )
                    for i in range(5)
                ]
                modality_features[mod_name] = skips
        
        return modality_features


class FeatureAggregationModule(nn.Module):
    """
    attention-based multi-modal feature fusion.
    
    learns to weight and fuse features from different modalities
    using both channel attention and spatial attention.
    """
    def __init__(self, feature_dim: int, num_modalities: int = 4):
        """
        args:
            feature_dim: feature channel dimension
            num_modalities: number of modalities to aggregate
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        self.modality_weighting = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(feature_dim * num_modalities, num_modalities),
            nn.Softmax(dim=1)
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        modality_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        aggregate multi-modal features via attention.
        
        args:
            modality_features: list of feature tensors, one per modality
        
        returns:
            aggregated feature tensor
        """
        stacked = torch.stack(modality_features, dim=1)
        batch_size = stacked.shape[0]
        num_mods = stacked.shape[1]
        
        pooled = torch.nn.functional.adaptive_avg_pool3d(
            stacked.view(batch_size, num_mods * self.feature_dim, -1),
            1
        )
        
        mod_weights = self.modality_weighting(
            pooled.view(batch_size, num_mods * self.feature_dim)
        )
        
        weighted = stacked * mod_weights.view(
            batch_size, num_mods, 1, 1, 1, 1
        )
        
        fused = weighted.sum(dim=1)
        
        spatial_att = self.spatial_attention(fused)
        fused = fused * spatial_att
        
        return fused


class Adapter(nn.Module):
    """
    lightweight per-voxel bottleneck adapter.
    
    small feed-forward mlp applied residually to feature maps.
    down-projects, applies nonlinearity, up-projects with
    residual connection.
    """
    def __init__(
        self,
        feature_dim: int,
        bottleneck_scale: float = 0.25
    ):
        """
        args:
            feature_dim: input feature dimension
            bottleneck_scale: bottleneck reduction ratio
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        bottleneck_dim = max(
            1,
            int(feature_dim * bottleneck_scale)
        )
        
        self.down_project = nn.Conv3d(
            feature_dim,
            bottleneck_dim,
            kernel_size=1
        )
        
        self.activation = nn.ReLU(inplace=True)
        
        self.up_project = nn.Conv3d(
            bottleneck_dim,
            feature_dim,
            kernel_size=1
        )
        
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        apply adapter as residual addition to input.
        
        args:
            x: input tensor of shape (b, c, d, h, w)
        
        returns:
            adapted tensor: x + adapter(x)
        """
        adapter_out = self.down_project(x)
        adapter_out = self.activation(adapter_out)
        adapter_out = self.up_project(adapter_out)
        
        return x + adapter_out


class MetaGenerator(nn.Module):
    """
    generates sample-specific adapter parameters from features.
    
    takes bottleneck features and modality mask, produces
    adapter weight tensors conditioned on sample content.
    enables meta-learning across different data distributions.
    """
    def __init__(
        self,
        bottleneck_dim: int,
        num_levels: int = 5,
        num_modalities: int = 4,
        bottleneck_scale: float = 0.25
    ):
        """
        args:
            bottleneck_dim: encoder bottleneck dimension
            num_levels: number of decoder levels needing adapters
            num_modalities: total number of modalities
            bottleneck_scale: bottleneck reduction ratio for adapters
        """
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.num_levels = num_levels
        self.bottleneck_scale = bottleneck_scale
        
        bottleneck_adapt_dim = max(
            1,
            int(bottleneck_dim * bottleneck_scale)
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(
                bottleneck_dim + num_modalities,
                256
            ),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Linear(
                    256,
                    (bottleneck_dim * bottleneck_adapt_dim +
                     bottleneck_dim * bottleneck_adapt_dim +
                     bottleneck_adapt_dim * bottleneck_dim +
                     bottleneck_dim)
                )
            )
            for _ in range(num_levels)
        ])

    def forward(
        self,
        bottleneck_features: torch.Tensor,
        modality_mask: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        args:
            bottleneck_features: encoder bottleneck output
            modality_mask: binary modality presence mask
        
        returns:
            list of dicts with adapter parameter tensors
        """
        batch_size = bottleneck_features.shape[0]
        
        bottleneck_flat = torch.nn.functional.adaptive_avg_pool3d(
            bottleneck_features,
            1
        ).view(batch_size, self.bottleneck_dim)
        
        combined = torch.cat([bottleneck_flat, modality_mask], dim=1)
        
        hidden = self.encoder(combined)
        
        adapter_params_list = []
        
        bottleneck_adapt_dim = max(
            1,
            int(self.bottleneck_dim * self.bottleneck_scale)
        )
        
        for level in range(self.num_levels):
            params = self.decoders[level](hidden)
            
            down_weight_size = (
                bottleneck_adapt_dim * self.bottleneck_dim
            )
            down_bias_size = bottleneck_adapt_dim
            up_weight_size = (
                self.bottleneck_dim * bottleneck_adapt_dim
            )
            
            down_weight = params[
                :, :down_weight_size
            ].view(
                batch_size,
                bottleneck_adapt_dim,
                self.bottleneck_dim,
                1,
                1,
                1
            )
            
            down_bias = params[
                :, down_weight_size:(
                    down_weight_size + down_bias_size
                )
            ].view(batch_size, bottleneck_adapt_dim, 1, 1, 1)
            
            up_weight = params[
                :, (down_weight_size + down_bias_size):(
                    down_weight_size + down_bias_size +
                    up_weight_size
                )
            ].view(
                batch_size,
                self.bottleneck_dim,
                bottleneck_adapt_dim,
                1,
                1,
                1
            )
            
            up_bias = params[
                :, (down_weight_size + down_bias_size +
                    up_weight_size):
            ].view(batch_size, self.bottleneck_dim, 1, 1, 1)
            
            adapter_params_list.append({
                'down_weight': down_weight,
                'down_bias': down_bias,
                'up_weight': up_weight,
                'up_bias': up_bias
            })
        
        return adapter_params_list


class Generator(nn.Module):
    """
    encoder-decoder generator with per-modality processing
    and meta-learned adaptive parameters.
    
    processes each modality through dedicated encoder,
    aggregates features, optionally generates adaptive
    parameters via meta-generator.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 32,
        all_modality_names: Sequence[str] = ALL_MODALITY_NAMES,
        use_meta_generator: bool = True,
        adapter_bottleneck_scale: float = 0.25
    ):
        """
        args:
            in_channels: channels per modality
            base_features: base feature dimension
            all_modality_names: list of all modality names
            use_meta_generator: enable meta-generator
            adapter_bottleneck_scale: bottleneck ratio
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_features = base_features
        self.all_modality_names = list(all_modality_names)
        self.use_meta_generator = use_meta_generator
        
        self.encoder_feature_dims = [
            base_features,
            base_features * 2,
            base_features * 4,
            base_features * 8,
            base_features * 16
        ]
        
        self.per_modality_encoders = PerModalityEncoders(
            in_channels,
            base_features,
            self.all_modality_names
        )
        
        self.aggregators = nn.ModuleList([
            FeatureAggregationModule(
                self.encoder_feature_dims[i],
                len(self.all_modality_names)
            )
            for i in range(5)
        ])
        
        if self.use_meta_generator:
            self.meta_generator = MetaGenerator(
                self.encoder_feature_dims[4],
                num_levels=5,
                num_modalities=len(self.all_modality_names),
                bottleneck_scale=adapter_bottleneck_scale
            )

    def forward(
        self,
        x_modalities: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        encode multi-modal input with fusion.
        
        args:
            x_modalities: dict of modality tensors
            modality_mask: optional modality presence mask
        
        returns:
            tuple of:
            - fused_skips: list of aggregated skip connections
            - bottleneck_features: encoder bottleneck
        """
        modality_features = self.per_modality_encoders(x_modalities)
        
        fused_skips = []
        for level in range(5):
            level_features = [
                modality_features[mod_name][level]
                for mod_name in self.all_modality_names
            ]
            
            fused = self.aggregators[level](level_features)
            fused_skips.append(fused)
        
        bottleneck_features = fused_skips[4]
        
        return fused_skips, bottleneck_features
