# model.py
# Residual U-Net + bottleneck self-attention for low-MIMO radar heatmap enhancement:
# 1x4 / 1x2 / 1x6 radar heatmap -> 1x8 teacher radar heatmap
#
# Main idea:
#   output = input + learned_residual
#
# Attention is added only at the bottleneck feature map, e.g., 16 x 8,
# so it is much cheaper than full-resolution Transformer attention.

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


def get_valid_num_heads(embed_dim, requested_heads=4):
    """
    Return valid attention head count for MultiheadAttention.
    """
    for h in range(min(requested_heads, embed_dim), 0, -1):
        if embed_dim % h == 0:
            return h
    return 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Basic CNN Blocks
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
    Residual convolution block with GroupNorm + SE attention.
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
# Bottleneck Self-Attention
# ============================================================

class BottleneckAttentionBlock(nn.Module):
    """
    Self-attention block applied at the bottleneck feature map.

    Input:
        x: [B, C, H, W]

    Internally:
        [B, C, H, W] -> [B, H*W, C] -> attention -> [B, C, H, W]

    This lets the model learn global range-angle structure without using
    expensive full-resolution attention.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    ):
        super().__init__()

        num_heads = get_valid_num_heads(channels, num_heads)
        hidden_dim = int(channels * mlp_ratio)

        self.norm1 = nn.LayerNorm(channels)

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(channels)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Self-attention
        tokens_norm = self.norm1(tokens)

        attn_out, _ = self.attn(
            tokens_norm,
            tokens_norm,
            tokens_norm,
            need_weights=False,
        )

        tokens = tokens + attn_out

        # Feed-forward
        tokens = tokens + self.mlp(self.norm2(tokens))

        out = tokens.transpose(1, 2).reshape(b, c, h, w)

        return out


class BottleneckAttention(nn.Module):
    """
    Stack of attention blocks at bottleneck.
    """

    def __init__(
        self,
        channels,
        num_blocks=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                BottleneckAttentionBlock(
                    channels=channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


# ============================================================
# Residual U-Net + Bottleneck Attention
# ============================================================

class RadarHeatmapUNet(nn.Module):
    """
    Residual U-Net + bottleneck self-attention.

    Input:
        x: [B, 1, 128, 64]

    Output:
        y: [B, 1, 128, 64]

    Task:
        1x4 radar heatmap -> 1x8 teacher heatmap

    Important:
        The model starts close to identity mapping:
            output ≈ input

        Then it learns the residual correction:
            output = input + correction
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        use_sigmoid=False,
        residual_scale=0.6,
        final_init_std=1e-4,
        use_bottleneck_attention=True,
        attention_blocks=2,
        attention_heads=4,
        attention_mlp_ratio=2.0,
        attention_dropout=0.0,
    ):
        super().__init__()

        # Kept for compatibility with existing train.py.
        # Sigmoid is not used in residual mode.
        self.use_sigmoid = use_sigmoid
        self.residual_scale = residual_scale

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Encoder
        self.enc1 = ResidualBlock(in_channels, c1)   # 128 x 64
        self.enc2 = DownBlock(c1, c2)                # 64 x 32
        self.enc3 = DownBlock(c2, c3)                # 32 x 16
        self.enc4 = DownBlock(c3, c4)                # 16 x 8

        # Bottleneck CNN
        self.bottleneck_cnn = nn.Sequential(
            ResidualBlock(c4, c4),
            ResidualBlock(c4, c4),
        )

        # Bottleneck attention
        if use_bottleneck_attention:
            self.bottleneck_attn = BottleneckAttention(
                channels=c4,
                num_blocks=attention_blocks,
                num_heads=attention_heads,
                mlp_ratio=attention_mlp_ratio,
                dropout=attention_dropout,
            )
        else:
            self.bottleneck_attn = nn.Identity()

        # Bottleneck fusion after attention
        self.bottleneck_fuse = ResidualBlock(c4, c4)

        # Decoder
        self.dec3 = UpBlock(
            in_channels=c4,
            skip_channels=c3,
            out_channels=c3,
        )

        self.dec2 = UpBlock(
            in_channels=c3,
            skip_channels=c2,
            out_channels=c2,
        )

        self.dec1 = UpBlock(
            in_channels=c2,
            skip_channels=c1,
            out_channels=c1,
        )

        # Residual prediction head
        self.head_block = ResidualBlock(c1, c1)

        self.head_conv = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )

        # Near-zero residual initialization
        nn.init.normal_(self.head_conv.weight, mean=0.0, std=final_init_std)
        nn.init.zeros_(self.head_conv.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck_cnn(e4)
        b = self.bottleneck_attn(b)
        b = self.bottleneck_fuse(b)

        # Decoder
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Residual correction
        residual_feat = self.head_block(d1)
        residual = self.head_conv(residual_feat)

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

    Supports both:

        loss = criterion(pred, target)

    and:

        loss = criterion(pred, target, input)

    When input is provided, it adds residual supervision:
        pred - input should match target - input
    """

    def __init__(
        self,
        # Compatible names with your train.py
        l1_weight=1.0,
        mse_weight=0.3,
        grad_weight=0.2,

        # Extra loss terms
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
        """
        return 1.0 + self.target_foreground_weight * target

    def make_residual_weight_map(self, target, inp):
        """
        Weight regions where target and input differ more.
        """
        delta = torch.abs(target - inp)

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
        # Direct prediction loss
        target_weight = self.make_target_weight_map(target)

        out_l1 = self.weighted_l1(pred, target, target_weight)
        out_mse = self.weighted_mse(pred, target, target_weight)

        loss = (
            self.l1_weight * out_l1
            + self.mse_weight * out_mse
        )

        # Residual correction loss
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

        # Structure loss
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
        use_bottleneck_attention=True,
        attention_blocks=2,
        attention_heads=4,
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