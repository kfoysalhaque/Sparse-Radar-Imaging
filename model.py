# radar_heatmap_model.py
# Model for low-MIMO radar heatmap enhancement:
# 1x4 / 1x2 / 1x6 radar heatmap -> 1x8 teacher radar heatmap

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Lightweight channel attention.
    Helps the model emphasize useful radar features.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResidualBlock(nn.Module):
    """
    Residual convolution block with GroupNorm.
    GroupNorm is stable even with small batch sizes.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )

        self.attn = SEBlock(out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.silu(out, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.attn(out)

        out = out + identity
        out = F.silu(out, inplace=True)

        return out


class DownBlock(nn.Module):
    """
    Downsampling block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        self.block = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block with skip connection.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.block = ResidualBlock(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = self.up_conv(x)

        x = torch.cat([x, skip], dim=1)

        x = self.block(x)

        return x


class RadarHeatmapUNet(nn.Module):
    """
    Residual Attention U-Net for radar heatmap enhancement.

    Input:
        x: [B, 1, 128, 64]

    Output:
        y: [B, 1, 128, 64]

    Recommended task:
        1x4 radar heatmap -> 1x8 radar heatmap
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        use_sigmoid=True,
    ):
        super().__init__()

        self.use_sigmoid = use_sigmoid

        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_channels)          # 128 x 64
        self.enc2 = DownBlock(base_channels, base_channels * 2)        # 64 x 32
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)    # 32 x 16
        self.enc4 = DownBlock(base_channels * 4, base_channels * 8)    # 16 x 8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8, base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 8),
        )

        # Decoder
        self.dec3 = UpBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
        )

        self.dec2 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
        )

        self.dec1 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
        )

        # Final reconstruction head
        self.head = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # [B, 32, 128, 64]
        e2 = self.enc2(e1)  # [B, 64, 64, 32]
        e3 = self.enc3(e2)  # [B, 128, 32, 16]
        e4 = self.enc4(e3)  # [B, 256, 16, 8]

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.head(d1)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out


class RadarHeatmapLoss(nn.Module):
    """
    Simple strong loss for radar heatmap regression.

    Combines:
        L1 loss      -> preserves overall heatmap intensity
        MSE loss     -> penalizes stronger pixel-level errors
        gradient loss -> preserves target structure and edges
    """

    def __init__(
        self,
        l1_weight=1.0,
        mse_weight=0.5,
        grad_weight=0.2,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def gradient_loss(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = self.l1(pred_dx, target_dx)
        loss_y = self.l1(pred_dy, target_dy)

        return loss_x + loss_y

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_mse = self.mse(pred, target)
        loss_grad = self.gradient_loss(pred, target)

        loss = (
            self.l1_weight * loss_l1
            + self.mse_weight * loss_mse
            + self.grad_weight * loss_grad
        )

        return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    model = RadarHeatmapUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        use_sigmoid=True,
    )

    x = torch.randn(4, 1, 128, 64)

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Output min/max:", y.min().item(), y.max().item())
    print("Trainable parameters:", count_parameters(model))


if __name__ == "__main__":
    main()