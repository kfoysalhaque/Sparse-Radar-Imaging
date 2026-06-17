# model.py
# Residual model for low-MIMO radar heatmap enhancement:
# 1x4 / 1x2 / 1x6 radar heatmap -> 1x8 teacher radar heatmap
#
# Main idea:
#   The model does not predict the full 1x8 heatmap from scratch.
#   It predicts only a residual correction:
#
#       output = input + learned_residual
#
# This keeps the initial output close to the input baseline.

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility
# ============================================================

def get_num_groups(channels, max_groups=8):
    """
    Return a valid number of groups for GroupNorm.
    """
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Blocks
# ============================================================

class SEBlock(nn.Module):
    """
    Lightweight channel attention.
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
    Residual convolution block with GroupNorm and SE attention.
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
            num_groups=get_num_groups(out_channels),
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
            num_groups=get_num_groups(out_channels),
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


# ============================================================
# Residual U-Net
# ============================================================

class RadarHeatmapUNet(nn.Module):
    """
    Residual Attention U-Net for radar heatmap enhancement.

    Input:
        x: [B, 1, 128, 64]

    Output:
        y: [B, 1, 128, 64]

    Task:
        1x4 radar heatmap -> 1x8 teacher heatmap

    Important:
        The model starts close to identity mapping.
        So initial val_psnr should be close to baseline_psnr.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        use_sigmoid=False,
        residual_scale=0.6,
        final_init_std=1e-4,
    ):
        super().__init__()

        # Kept for compatibility with your training script.
        # In residual mode, sigmoid is not used.
        self.use_sigmoid = use_sigmoid
        self.residual_scale = residual_scale

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

        # Residual prediction head
        self.head_block = ResidualBlock(base_channels, base_channels)

        self.head_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

        # Near-zero residual initialization.
        # This makes initial output almost equal to input.
        nn.init.normal_(self.head_conv.weight, mean=0.0, std=final_init_std)
        nn.init.zeros_(self.head_conv.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Predict residual correction
        residual_feat = self.head_block(d1)
        residual = self.head_conv(residual_feat)

        # Residual enhancement
        out = x + self.residual_scale * torch.tanh(residual)

        # Input/target are normalized to [0, 1]
        out = torch.clamp(out, 0.0, 1.0)

        return out


# ============================================================
# Loss
# ============================================================

class RadarHeatmapLoss(nn.Module):
    """
    Residual-aware foreground-weighted radar heatmap loss.

    This supports both calls:

        loss = criterion(pred, target)

    and:

        loss = criterion(pred, target, input)

    When input is given, the loss emphasizes the region where 1x8 differs from
    the low-MIMO input:

        delta_gt = target - input
        delta_pred = pred - input

    This is more useful than plain global L1/MSE because most radar heatmap
    pixels are background.
    """

    def __init__(
        self,
        # These names keep compatibility with your current train.py
        l1_weight=1.0,
        mse_weight=0.3,
        grad_weight=0.2,

        # Extra weights
        residual_l1_weight=1.0,
        residual_mse_weight=0.3,
        target_foreground_weight=4.0,
        residual_foreground_weight=8.0,
        eps=1e-8,
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight

        self.residual_l1_weight = residual_l1_weight
        self.residual_mse_weight = residual_mse_weight

        self.target_foreground_weight = target_foreground_weight
        self.residual_foreground_weight = residual_foreground_weight

        self.eps = eps

    def make_target_weight_map(self, target):
        """
        Weight bright 1x8 target regions more.
        target should be normalized to [0, 1].
        """
        return 1.0 + self.target_foreground_weight * target

    def make_residual_weight_map(self, target, inp):
        """
        Weight regions where 1x8 and input differ more.

        This focuses the model on learning the actual missing correction,
        not just background matching.
        """
        delta = torch.abs(target - inp)

        # Normalize delta per sample to keep weights stable.
        b = delta.shape[0]
        delta_flat = delta.view(b, -1)
        delta_max = delta_flat.max(dim=1)[0].view(b, 1, 1, 1)

        delta_norm = delta / (delta_max + self.eps)

        return 1.0 + self.residual_foreground_weight * delta_norm

    def weighted_l1(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))

    def weighted_mse(self, pred, target, weight):
        return torch.mean(weight * (pred - target) ** 2)

    def gradient_loss(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = torch.mean(torch.abs(pred_dx - target_dx))
        loss_y = torch.mean(torch.abs(pred_dy - target_dy))

        return loss_x + loss_y

    def forward(self, pred, target, inp=None):
        # ------------------------------------------------------------
        # 1. Direct output loss: pred should match 1x8 target
        # ------------------------------------------------------------
        target_weight = self.make_target_weight_map(target)

        out_l1 = self.weighted_l1(pred, target, target_weight)
        out_mse = self.weighted_mse(pred, target, target_weight)

        loss = (
            self.l1_weight * out_l1
            + self.mse_weight * out_mse
        )

        # ------------------------------------------------------------
        # 2. Residual correction loss:
        #    pred - input should match target - input
        # ------------------------------------------------------------
        if inp is not None:
            delta_pred = pred - inp
            delta_gt = target - inp

            residual_weight = self.make_residual_weight_map(target, inp)

            residual_l1 = self.weighted_l1(
                delta_pred,
                delta_gt,
                residual_weight,
            )

            residual_mse = self.weighted_mse(
                delta_pred,
                delta_gt,
                residual_weight,
            )

            loss = loss + (
                self.residual_l1_weight * residual_l1
                + self.residual_mse_weight * residual_mse
            )

        # ------------------------------------------------------------
        # 3. Structure / edge loss
        # ------------------------------------------------------------
        loss_grad = self.gradient_loss(pred, target)
        loss = loss + self.grad_weight * loss_grad

        return loss


# ============================================================
# Self-test
# ============================================================

def main():
    model = RadarHeatmapUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        use_sigmoid=False,
        residual_scale=0.6,
    )

    criterion = RadarHeatmapLoss()

    x = torch.rand(4, 1, 128, 64)
    target = torch.rand(4, 1, 128, 64)

    with torch.no_grad():
        y = model(x)

    loss_without_input = criterion(y, target)
    loss_with_input = criterion(y, target, x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Output min/max:", y.min().item(), y.max().item())
    print("Initial output-input MSE:", torch.mean((y - x) ** 2).item())
    print("Loss without input:", loss_without_input.item())
    print("Loss with input   :", loss_with_input.item())
    print("Trainable parameters:", count_parameters(model))


if __name__ == "__main__":
    main()