"""
NAFNet architecture for HDR bracket merging.
Input: 9-channel tensor (3 brackets x 3 RGB)
Output: 3-channel RGB tensor

Architecture:
  - Nonlinear Activation Free blocks
  - SimpleGate activation
  - LayerNorm2d normalization
  - U-Net style encoder-decoder
  - Optional global residual from mid-exposure bracket
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        B, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        B, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (
            g - y * mean_gy - mean_g
        )
        return (gx,
                (grad_output * y).sum(dim=[0, 2, 3]),
                grad_output.sum(dim=[0, 2, 3]),
                None)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter(
            'weight', nn.Parameter(torch.ones(channels))
        )
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(channels))
        )
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(
            x, self.weight, self.bias, self.eps
        )


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_ch = int(c * dw_expand)
        ffn_ch = int(c * ffn_expand)

        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1)
        )

        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()

        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) \
            if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) \
            if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(
            torch.zeros((1, c, 1, 1)), requires_grad=True
        )
        self.gamma = nn.Parameter(
            torch.zeros((1, c, 1, 1)), requires_grad=True
        )

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self,
                 in_channels=9,
                 out_channels=3,
                 width=32,
                 middle_blk_num=12,
                 enc_blk_nums=[2, 2, 4, 8],
                 dec_blk_nums=[2, 2, 2, 2],
                 use_residual=True,
                 residual_start=3):
        super().__init__()
        self.use_residual = use_residual
        self.residual_start = residual_start

        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[
            NAFBlock(chan) for _ in range(middle_blk_num)
        ])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape

        if self.use_residual:
            mid_bracket = inp[:, self.residual_start:self.residual_start + 3, :, :]

        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(
            self.decoders, self.ups, reversed(encs)
        ):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        out = x[:, :, :H, :W]

        if self.use_residual:
            return torch.clamp(mid_bracket + out, 0, 1)
        else:
            return torch.clamp(out, 0, 1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
