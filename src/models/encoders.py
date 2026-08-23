import torch
import torch.nn as nn
from typing import List


class nnUNetConvBlock(nn.Module):
    """
    double 3d convolution with group normalization and leaky relu.
    
    implements the basic building block: conv3d -> groupnorm ->
    leakyrelu -> conv3d -> groupnorm -> leakyrelu + optional dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.0
    ):
        """
        args:
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: convolution kernel size
            dropout_rate: dropout probability
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.norm1 = nn.GroupNorm(
            num_groups=min(32, out_channels),
            num_channels=out_channels
        )
        
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.norm2 = nn.GroupNorm(
            num_groups=min(32, out_channels),
            num_channels=out_channels
        )
        
        self.dropout = (
            nn.Dropout3d(p=dropout_rate)
            if dropout_rate > 0 else None
        )
        self.activation = nn.LeakyReLU(
            negative_slope=0.01,
            inplace=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass through double conv block.
        
        args:
            x: input tensor of shape (b, c_in, d, h, w)
        
        returns:
            output tensor of shape (b, c_out, d, h, w)
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        return x


class nnUNetEncoder(nn.Module):
    """
    5-level hierarchical encoder for 3d medical images.
    
    progressively downsamples with max pooling and processes with
    conv blocks, returning skip connections at each level.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 32,
        dropout_rate: float = 0.0
    ):
        """
        args:
            in_channels: number of input channels
            base_features: base feature dimension (doubled at each level)
            dropout_rate: dropout probability for conv blocks
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_features = base_features
        
        self.level_features = [
            base_features,
            base_features * 2,
            base_features * 4,
            base_features * 8,
            base_features * 16
        ]
        
        self.conv_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        in_feat = in_channels
        for out_feat in self.level_features:
            self.conv_blocks.append(
                nnUNetConvBlock(
                    in_feat,
                    out_feat,
                    kernel_size=3,
                    dropout_rate=dropout_rate
                )
            )
            in_feat = out_feat

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        encode input through 5 levels, return skip connections.
        
        args:
            x: input tensor of shape (b, c_in, d, h, w)
        
        returns:
            list of 5 skip connection tensors at each level
        """
        skips = []
        
        for level in range(5):
            x = self.conv_blocks[level](x)
            skips.append(x)
            
            if level < 4:
                x = self.pool(x)
        
        return skips
