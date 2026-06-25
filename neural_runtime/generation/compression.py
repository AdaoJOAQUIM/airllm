"""
Neural Compression Module
=====================

Advanced weight compression using learned methods.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class NeuralCompressor(nn.Module):
    """
    Learned weight compression using autoencoder.
    
    The autoencoder learns to compress weights while preserving
    information needed for inference.
    """
    
    def __init__(
        self,
        weight_shape: Tuple[int, int],
        latent_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        self.weight_shape = weight_shape
        h, w = weight_shape
        self.original_params = h * w
        self.latent_dim = latent_dim
        
        # Encoder: weight -> latent
        self.encoder = nn.Sequential(
            nn.Linear(h * w, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Decoder: latent -> weight
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, h * w),
        )
        
        self.compression_ratio = self.original_params / latent_dim
        logger.info(f"NeuralCompressor: {self.original_params:,} → {latent_dim} ({self.compression_ratio:.0f}x)")
    
    def compress(self, weight: torch.Tensor) -> torch.Tensor:
        """Compress weight to latent."""
        flat = weight.flatten().unsqueeze(0)
        return self.encoder(flat).squeeze(0)
    
    def decompress(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct weight from latent."""
        recon = self.decoder(latent.unsqueeze(0))
        return recon.squeeze(0).view(self.weight_shape)
    
    def forward(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress and reconstruct."""
        latent = self.compress(weight)
        recon = self.decompress(latent)
        return latent, recon
    
    def reconstruction_loss(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        _, recon = self.forward(weight)
        return F.mse_loss(recon, weight)
