"""ESRT: Efficient Super-Resolution Transformer (Lu et al., CVPRW 2022), vendored.

Source: https://github.com/luissen/ESRT (model/esrt.py, model/common.py, util/transformer.py,
util/tools.py), MIT License, Copyright (c) 2022 luissen. Architecture unchanged (32 features,
one LCB+LTB body: three high-preserving blocks, efficient multi-head attention over 3x3 unfolded
patches with the token set split in 4). Changes for this project: one input / output channel
(Sentinel-1 VV dB) instead of RGB, unused imports (thop, pdb, positional embeddings) and debug
prints removed, files merged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ model/common.py
def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, groups=groups)


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        super().__init__(*m)


# ------------------------------------------------------------------ util/tools.py
def same_padding(images, ksizes, strides, rates):
    _, _, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    ekr = (ksizes[0] - 1) * rates[0] + 1
    ekc = (ksizes[1] - 1) * rates[1] + 1
    pr = max(0, (out_rows - 1) * strides[0] + ekr - rows)
    pc = max(0, (out_cols - 1) * strides[1] + ekc - cols)
    pt, pl = int(pr / 2.), int(pc / 2.)
    return torch.nn.ZeroPad2d((pl, pc - pl, pt, pr - pt))(images)


def extract_image_patches(images, ksizes, strides, rates):
    images = same_padding(images, ksizes, strides, rates)
    return torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)(images)


def reverse_patches(images, out_size, ksizes, strides, padding):
    return torch.nn.Fold(output_size=out_size, kernel_size=ksizes, dilation=1, padding=padding,
                         stride=strides)(images)


# ------------------------------------------------------------------ util/transformer.py
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.reduce = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_all = torch.split(q, math.ceil(N // 4), dim=-2)
        k_all = torch.split(k, math.ceil(N // 4), dim=-2)
        v_all = torch.split(v, math.ceil(N // 4), dim=-2)
        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            output.append((attn @ v).transpose(1, 2))
        x = torch.cat(output, dim=1).reshape(B, N, C)
        return self.proj(x)


class MLABlock(nn.Module):
    def __init__(self, n_feat=64, dim=768, num_heads=8, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        self.dim = dim
        self.atten = EffAttention(self.dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim // 4, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x):
        x = extract_image_patches(x, ksizes=[3, 3], strides=[1, 1], rates=[1, 1])
        x = x.permute(0, 2, 1)
        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------------ model/esrt.py
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))


class one_conv(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(inchanels, growth_rate, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.flag = relu
        self.conv1 = nn.Conv2d(growth_rate, inchanels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        if relu:
            self.relu = nn.PReLU(growth_rate)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)

    def forward(self, x):
        if not self.flag:
            return self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
        return self.weight1(x) + self.weight2(self.conv1(self.relu(self.conv(x))))


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class one_module(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.layer1 = one_conv(n_feats, n_feats // 2, 3)
        self.layer2 = one_conv(n_feats, n_feats // 2, 3)
        self.layer4 = BasicConv(n_feats, n_feats, 3, 1, 1)
        self.alise = BasicConv(2 * n_feats, n_feats, 1, 1, 0)
        self.atten = CALayer(n_feats)
        self.weight1 = Scale(1)
        self.weight2 = Scale(1)
        self.weight3 = Scale(1)
        self.weight4 = Scale(1)
        self.weight5 = Scale(1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2), self.weight3(x1)], 1))))
        return self.weight4(x) + self.weight5(x4)


class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.encoder = one_module(n_feats)
        self.decoder_low = one_module(n_feats)
        self.decoder_high = one_module(n_feats)
        self.alise = one_module(n_feats)
        self.alise2 = BasicConv(2 * n_feats, n_feats, 1, 1, 0)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        for _ in range(5):
            x2 = self.decoder_low(x2)
        x3 = x2
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x


class Un(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.encoder1 = Updownblock(n_feats)
        self.encoder2 = Updownblock(n_feats)
        self.encoder3 = Updownblock(n_feats)
        self.reduce = default_conv(3 * n_feats, n_feats, 3)
        self.weight2 = Scale(1)
        self.weight1 = Scale(1)
        self.attention = MLABlock(n_feat=n_feats, dim=288)
        self.alise = default_conv(n_feats, n_feats, 3)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        _, _, h, w = x3.shape
        out = self.attention(self.reduce(torch.cat([x1, x2, x3], dim=1)))
        out = out.permute(0, 2, 1)
        out = reverse_patches(out, (h, w), (3, 3), 1, 1)
        out = self.alise(out)
        return self.weight1(x) + self.weight2(out)


class ESRT(nn.Module):
    def __init__(self, upscale=2, in_ch=1, out_ch=1, conv=default_conv):
        super().__init__()
        n_feats, kernel_size = 32, 3
        self.n_blocks = 1
        self.head = nn.Sequential(conv(in_ch, n_feats, kernel_size))
        self.body = nn.Sequential(*[Un(n_feats=n_feats) for _ in range(self.n_blocks)])
        self.tail = nn.Sequential(Upsampler(conv, upscale, n_feats), conv(n_feats, out_ch, kernel_size))
        self.up = nn.Sequential(Upsampler(conv, upscale, n_feats), BasicConv(n_feats, out_ch, 3, 1, 1))
        self.reduce = conv(self.n_blocks * n_feats, n_feats, kernel_size)

    def forward(self, x1):
        x1 = self.head(x1)
        res2 = x1
        body_out = []
        for i in range(self.n_blocks):
            x1 = self.body[i](x1)
            body_out.append(x1)
        res1 = self.reduce(torch.cat(body_out, 1))
        x1 = self.tail(res1)
        # note: the original ends with BasicConv(relu=True) on this skip path, so the skip is >= 0;
        # inputs here are min-max normalized to [0, 1], so that is kept unchanged.
        return self.up(res2) + x1
