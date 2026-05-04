"""
Restormer architecture for HDR bracket merging.
Input: 9-channel tensor (3 brackets x 3 RGB channels) — under/mid/over
Output: 3-channel RGB tensor (finished edit)

Architecture:
  - Multi-Dconv Head Transposed Self-Attention (MDTA)
  - Gated-Dconv Feed-Forward Network (GDFN)
  - 4-level encoder-decoder with skip connections
  - Input projection: 9 -> dim channels
  - Output projection: dim -> 3 channels
  - Global residual connection from mid-exposure bracket (channels 3:6)

Parameters: ~7-8M (dim=32, blocks=[2,3,3,4])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# PyTorch conv kernels use int32 internally to index input/output tensors.
# Once C*H*W exceeds 2^31 (~2.15 B elements), depthwise convs raise
# "canUse32BitIndexMath" errors. The fix is to apply the same depthwise conv
# in spatial chunks, keeping each chunk under the limit. The conv must be
# 3x3 depthwise with padding=1 for the 1-pixel overlap math to be correct;
# all dw convs in MDTA and GDFN match this shape.
_INT32_BUDGET = 2 * (10 ** 9)  # 2 B elements; safely under 2^31


def _safe_depthwise_3x3(dw_module, x):
    """Apply a 3x3 stride-1 depthwise conv with int32-safe spatial chunking."""
    if x.numel() < _INT32_BUDGET:
        return dw_module(x)

    B, C, H, W = x.shape
    # Pick chunk count along H so each chunk is comfortably under the budget.
    n_chunks = 2
    while (B * C * (H // n_chunks + 2) * W) >= _INT32_BUDGET:
        n_chunks += 1

    out = torch.empty_like(x)
    base = H // n_chunks
    for i in range(n_chunks):
        h_start = i * base
        h_end = (i + 1) * base if i < n_chunks - 1 else H
        # Take chunk plus 1-pixel context on each interior side; the
        # 3x3 padding=1 conv reproduces the original boundary behavior at
        # H=0 and H=H-1 since we don't slice past the actual edges.
        in_start = max(0, h_start - 1)
        in_end = min(H, h_end + 1)
        chunk = x[:, :, in_start:in_end, :].contiguous()
        out_chunk = dw_module(chunk)
        keep_start = h_start - in_start
        keep_end = keep_start + (h_end - h_start)
        out[:, :, h_start:h_end, :] = out_chunk[:, :, keep_start:keep_end, :]
    return out


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.weight.shape,
            self.weight, self.bias, 1e-6
        ).permute(0, 3, 1, 2)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(
            dim * 3, dim * 3, 3,
            padding=1, groups=dim * 3, bias=False
        )
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = _safe_depthwise_3x3(self.qkv_dw, self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(
            q, 'b (head c) h w -> b head c (h w)',
            head=self.num_heads
        )
        k = rearrange(
            k, 'b (head c) h w -> b head c (h w)',
            head=self.num_heads
        )
        v = rearrange(
            v, 'b (head c) h w -> b head c (h w)',
            head=self.num_heads
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            'b head c (h w) -> b (head c) h w',
            head=self.num_heads, h=H, w=W
        )
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.proj_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dw = nn.Conv2d(
            hidden * 2, hidden * 2, 3,
            padding=1, groups=hidden * 2, bias=False
        )
        self.proj_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        x = self.proj_in(x)
        x1, x2 = _safe_depthwise_3x3(self.dw, x).chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, padding=1, bias=False)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


def icnr_init(conv: nn.Conv2d, upscale_factor: int = 2) -> None:
    """ICNR initialization for the Conv2d that feeds a PixelShuffle(r).

    The r² sub-pixel filters that get de-interleaved into adjacent output
    pixels are set to the SAME values at init. After the pixel shuffle the
    output starts piecewise-constant on r×r blocks (no checkerboard), and
    training can diverge from there without the structural r-pixel bias.

    Reference: Aitken et al. 2017, "Checkerboard artifact free sub-pixel
    convolution" (https://arxiv.org/abs/1707.02937).
    """
    out_c, in_c, k1, k2 = conv.weight.shape
    assert out_c % (upscale_factor ** 2) == 0, (
        f"Conv2d out_channels ({out_c}) must be divisible by upscale_factor² "
        f"({upscale_factor ** 2}) for ICNR init"
    )
    r2 = upscale_factor ** 2
    sub_c = out_c // r2
    # Draw one kernel per output channel, then expand it to r² consecutive
    # input channels (PyTorch PixelShuffle consumes channels [c*r² : c*r²+r²]
    # as output channel c's r×r sub-pixel block). This makes each output
    # channel's r² sub-pixels start equal → block-constant output, no
    # checkerboard.
    base = torch.empty(sub_c, in_c, k1, k2)
    nn.init.kaiming_normal_(base, mode="fan_out", nonlinearity="linear")
    w = base.unsqueeze(1).expand(sub_c, r2, in_c, k1, k2).reshape(out_c, in_c, k1, k2)
    with torch.no_grad():
        conv.weight.copy_(w)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class Upsample(nn.Module):
    """
    Upsample variants (all: C → C/2 channels, H×W → 2H×2W):
      - pixel_shuffle (original): Conv2d(C → 2C) → PixelShuffle(2).
        PRONE to 4-pixel checkerboard artifacts without ICNR init.
      - pixel_shuffle_icnr (default): same structure with ICNR init applied
        to the pre-shuffle Conv2d. Same capacity as pixel_shuffle, no
        structural checkerboard at init. Recommended.
      - bilinear: Upsample(2×, bilinear) → Conv2d(C → C/2). Lower capacity
        (~4.5×dim² weights vs 18×dim²); observed to diverge at the default
        3e-4 lr on this task (2026-04-21).
    """

    def __init__(self, dim, mode: str = "pixel_shuffle_icnr"):
        super().__init__()
        self.mode = mode
        if mode in ("pixel_shuffle", "pixel_shuffle_icnr"):
            conv = nn.Conv2d(dim, dim * 2, 3, padding=1, bias=False)
            if mode == "pixel_shuffle_icnr":
                icnr_init(conv, upscale_factor=2)
            self.body = nn.Sequential(conv, nn.PixelShuffle(2))
        elif mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False)
        else:
            raise ValueError(f"Unknown upsample mode: {mode}")

    def forward(self, x):
        if self.mode in ("pixel_shuffle", "pixel_shuffle_icnr"):
            return self.body(x)
        return self.conv(self.up(x))


class Restormer(nn.Module):
    def __init__(self,
                 in_channels=9,
                 out_channels=3,
                 dim=32,
                 num_blocks=[2, 3, 3, 4],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 use_residual=True,
                 residual_start=3,
                 upsample_mode="pixel_shuffle"):
        super().__init__()
        self.use_residual = use_residual
        # Channel offset for the residual base slice. Default 3 mirrors the
        # original 9-channel bracket input (mid = channels 3:6). For a
        # 3-channel V4 input, set residual_start=0 so the base is the input
        # itself (V4 + delta → target). Must satisfy
        # residual_start + out_channels <= in_channels.
        self.residual_start = residual_start
        # upsample_mode: "pixel_shuffle" (original) or "bilinear" (no
        # checkerboard artifact). See Upsample docstring.
        self.upsample_mode = upsample_mode

        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        # Encoder
        self.encoder_1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor)
            for _ in range(num_blocks[0])
        ])
        self.down_1 = Downsample(dim)

        self.encoder_2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor)
            for _ in range(num_blocks[1])
        ])
        self.down_2 = Downsample(dim * 2)

        self.encoder_3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor)
            for _ in range(num_blocks[2])
        ])
        self.down_3 = Downsample(dim * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion_factor)
            for _ in range(num_blocks[3])
        ])

        # Decoder
        self.up_3 = Upsample(dim * 8, mode=upsample_mode)
        self.reduce_3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.decoder_3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor)
            for _ in range(num_blocks[2])
        ])

        self.up_2 = Upsample(dim * 4, mode=upsample_mode)
        self.reduce_2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor)
            for _ in range(num_blocks[1])
        ])

        self.up_1 = Upsample(dim * 2, mode=upsample_mode)
        self.reduce_1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.decoder_1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor)
            for _ in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor)
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        if self.use_residual:
            s = self.residual_start
            mid_bracket = x[:, s:s + 3, :, :]

        inp = self.patch_embed(x)

        e1 = self.encoder_1(inp)
        e2 = self.encoder_2(self.down_1(e1))
        e3 = self.encoder_3(self.down_2(e2))
        b = self.bottleneck(self.down_3(e3))

        d3 = self.decoder_3(
            self.reduce_3(torch.cat([self.up_3(b), e3], dim=1))
        )
        d2 = self.decoder_2(
            self.reduce_2(torch.cat([self.up_2(d3), e2], dim=1))
        )
        d1 = self.decoder_1(
            self.reduce_1(torch.cat([self.up_1(d2), e1], dim=1))
        )

        out = self.output(self.refinement(d1))

        if self.use_residual:
            return torch.clamp(mid_bracket + out, 0, 1)
        else:
            return torch.clamp(out, 0, 1)


class RestormerGC(Restormer):
    """
    Restormer with gradient checkpointing for training at larger crop sizes.
    Reduces VRAM by recomputing activations during backward pass.
    Approx 30-40% VRAM reduction at ~30% training speed penalty.

    Usage:
        model = RestormerGC(in_channels=9, out_channels=3, dim=48,
                            num_blocks=[4,6,6,8], num_refinement_blocks=4,
                            heads=[1,2,4,8])
    """

    def forward(self, x):
        import torch.utils.checkpoint as cp

        if self.use_residual:
            s = self.residual_start
            mid_bracket = x[:, s:s + 3, :, :]

        inp = self.patch_embed(x)

        def run_seq(module, inp):
            return module(inp)

        def run_decode(up, reduce, decoder, x, skip):
            return decoder(reduce(torch.cat([up(x), skip], dim=1)))

        e1 = cp.checkpoint(run_seq, self.encoder_1, inp, use_reentrant=False)
        e2 = cp.checkpoint(run_seq, self.encoder_2, self.down_1(e1), use_reentrant=False)
        e3 = cp.checkpoint(run_seq, self.encoder_3, self.down_2(e2), use_reentrant=False)
        b = cp.checkpoint(run_seq, self.bottleneck, self.down_3(e3), use_reentrant=False)
        d3 = cp.checkpoint(run_decode, self.up_3, self.reduce_3, self.decoder_3, b, e3, use_reentrant=False)
        d2 = cp.checkpoint(run_decode, self.up_2, self.reduce_2, self.decoder_2, d3, e2, use_reentrant=False)
        d1 = cp.checkpoint(run_decode, self.up_1, self.reduce_1, self.decoder_1, d2, e1, use_reentrant=False)
        out = self.output(cp.checkpoint(run_seq, self.refinement, d1, use_reentrant=False))

        if self.use_residual:
            return torch.clamp(mid_bracket + out, 0, 1)
        else:
            return torch.clamp(out, 0, 1)
