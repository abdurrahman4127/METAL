import torch
import torch.nn as nn


class ModalityDiscriminator(nn.Module):
    """
    adversarial discriminator predicting modality presence.
    
    takes encoder bottleneck features and predicts which
    modalities are present in the sample. encourages
    modality-invariant feature learning.
    """
    def __init__(
        self,
        bottleneck_dim: int,
        num_modalities: int = 4,
        hidden_dim: int = 48
    ):
        """
        args:
            bottleneck_dim: encoder bottleneck dimension
            num_modalities: number of modality classes
            hidden_dim: hidden layer dimension
        """
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.num_modalities = num_modalities
        
        self.discriminator = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_modalities)
        )

    def forward(
        self,
        bottleneck_features: torch.Tensor
    ) -> torch.Tensor:
        """
        predict modality presence from bottleneck.
        
        args:
            bottleneck_features: encoder bottleneck
                of shape (batch, channels, d, h, w)
        
        returns:
            logits of shape (batch, num_modalities)
        """
        return self.discriminator(bottleneck_features)
