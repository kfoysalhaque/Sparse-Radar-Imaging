# model.py
# Lightweight reconstruction model for sparse MIMO FMCW radar imaging.

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBlock(nn.Module):
    """
    Simple convolution block:
        Conv2D -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SparseRadarReconNet(nn.Module):
    """
    Mask-aware residual CNN.

    Input:
        sparse image: [B, 1, H, W]
        mask:         [B, 8]

    Output:
        reconstructed image: [B, 1, H, W]

    The mask is expanded into spatial feature maps and concatenated with
    the sparse radar image.
    """

    def __init__(
        self,
        num_virtual_channels: int = config.NUM_VIRTUAL_CHANNELS,
        base_channels: int = config.BASE_CHANNELS,
    ):
        super().__init__()

        self.num_virtual_channels = num_virtual_channels

        # Input channels = 1 sparse image + 8 mask maps
        in_channels = 1 + num_virtual_channels

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels),
            ConvBlock(base_channels, base_channels),
        )

        self.out_conv = nn.Conv2d(
            base_channels,
            1,
            kernel_size=3,
            padding=1,
        )

    def expand_mask(self, mask, height: int, width: int):
        """
        Convert mask from [B, 8] to [B, 8, H, W].
        """
        if mask.ndim != 2:
            raise ValueError(f"mask should be [B, 8], got {mask.shape}")

        mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, 8, 1, 1]
        mask = mask.expand(-1, -1, height, width)  # [B, 8, H, W]

        return mask

    def forward(self, sparse_img, mask):
        """
        Args:
            sparse_img: [B, 1, H, W]
            mask:       [B, 8]

        Returns:
            recon_img:  [B, 1, H, W]
        """
        if sparse_img.ndim != 4:
            raise ValueError(f"sparse_img should be [B,1,H,W], got {sparse_img.shape}")

        b, c, h, w = sparse_img.shape

        if c != 1:
            raise ValueError(f"sparse_img should have 1 channel, got {c}")

        mask_map = self.expand_mask(mask, h, w)

        x = torch.cat([sparse_img, mask_map], dim=1)

        feat = self.encoder(x)

        residual = self.out_conv(feat)

        # Residual reconstruction
        recon = sparse_img + residual

        # Since training images are scaled to [0, 1], keep output in [0, 1]
        recon = torch.clamp(recon, 0.0, 1.0)

        return recon


class SimpleCNNBaseline(nn.Module):
    """
    Simple CNN baseline without mask input.

    Input:
        sparse image: [B, 1, H, W]

    Output:
        reconstructed image: [B, 1, H, W]
    """

    def __init__(self, base_channels: int = config.BASE_CHANNELS):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(1, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, sparse_img, mask=None):
        residual = self.net(sparse_img)
        recon = sparse_img + residual
        recon = torch.clamp(recon, 0.0, 1.0)
        return recon


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    height = config.IMAGE_HEIGHT
    width = config.IMAGE_WIDTH

    sparse = torch.randn(batch_size, 1, height, width)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    model = SparseRadarReconNet()

    out = model(sparse, mask)

    print("Model:", model.__class__.__name__)
    print("Input sparse:", sparse.shape)
    print("Input mask:", mask.shape)
    print("Output:", out.shape)
    print("Trainable parameters:", count_parameters(model))