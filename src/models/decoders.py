import torch
import torch.nn as nn
from typing import List

from .encoders import nnUNetConvBlock


class nnUNetDecoder(nn.Module):
    """
    symmetric decoder that upsamples and processes features.
    
    concatenates skip connections from encoder and produces
    final segmentation output.
    """
    def __init__(
        self,
        encoder_feature_dims: List[int],
        out_channels: int = 3,
        dropout_rate: float = 0.0
    ):
        """
        args:
            encoder_feature_dims: list of encoder feature dimensions
            out_channels: number of output segmentation channels
            dropout_rate: dropout probability
        """
        super().__init__()
        self.encoder_feature_dims = encoder_feature_dims
        self.out_channels = out_channels
        
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='trilinear',
            align_corners=False
        )
        
        self.conv_blocks = nn.ModuleList()
        for i in range(4, 0, -1):
            in_feat = encoder_feature_dims[i] + encoder_feature_dims[i - 1]
            out_feat = encoder_feature_dims[i - 1]
            
            self.conv_blocks.append(
                nnUNetConvBlock(
                    in_feat,
                    out_feat,
                    kernel_size=3,
                    dropout_rate=dropout_rate
                )
            )
        
        self.final_conv = nn.Conv3d(
            encoder_feature_dims[0],
            out_channels,
            kernel_size=1
        )

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        decode skip connections to segmentation output.
        
        args:
            skips: list of 5 skip connection tensors
        
        returns:
            segmentation logits of shape (b, out_channels, d, h, w)
        """
        x = skips[4]
        
        for level in range(4, 0, -1):
            x = self.upsample(x)
            x = torch.cat([x, skips[level - 1]], dim=1)
            x = self.conv_blocks[4 - level](x)
        
        logits = self.final_conv(x)
        return logits
