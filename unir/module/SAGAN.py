# Implementation of self-attention layer by Christian Cosgrove
# For SAGAN implementation
# Based on Non-local Neural Networks by Wang et al. https://arxiv.org/abs/1711.07971

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelfAttentionPost(nn.Module):
    def __init__(self, input_size, attention_size):
        super(SelfAttentionPost, self).__init__()
        self.attention_size = attention_size
        self.gamma = nn.Parameter(torch.tensor(0.))
        self.h = spectral_norm(
            nn.Conv2d(input_size, self.attention_size, 1, stride=1))
        self.i = spectral_norm(
            nn.Conv2d(self.attention_size, input_size, 1, stride=1))

    def forward(self, x, att):
        width = x.size(2)
        height = x.size(3)
        m = x
        h = self.gamma * self.h(m)
        h = h.permute(0, 2, 3, 1).contiguous().view(-1,
                                                    width * height, self.attention_size)
        h = torch.bmm(att, h)
        h = h.view(-1, width, height, self.attention_size).permute(0, 3, 1, 2)
        m = self.i(h) + m
        return m


class SelfAttention(nn.Module):
    def __init__(self, input_size, attention_size):
        super(SelfAttention, self).__init__()
        self.attention_size = attention_size

        # attention layers
        self.f = spectral_norm(
            nn.Conv2d(input_size, attention_size, 1, stride=1))
        self.g = spectral_norm(
            nn.Conv2d(input_size, attention_size, 1, stride=1))
        self.input_size = input_size

    def forward(self, x):
        width = x.size(2)
        height = x.size(3)
        channels = x.size(1)
        m = x
        f = self.f(m)
        f = torch.transpose(
            f.view(-1, self.attention_size, width * height), 1, 2)
        g = self.g(m)
        g = g.view(-1, self.attention_size, width * height)
        att = torch.bmm(f, g)

        return F.softmax(att, 1)


def _upsample(x):
    h, w = x.shape[2:]
    return F.interpolate(x, scale_factor=2, mode='bilinear')


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.avg_pool2d(x, 2)


def upsample_conv(x, conv):
    return conv(_upsample(x))


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1, nl_layer=F.relu,
                 upsample=False):
        super().__init__()
        self.upsample = upsample
        self.activation = nl_layer
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.c1 = spectral_norm(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad))
        self.c2 = spectral_norm(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad))
        self.b1 = nn.BatchNorm2d(num_features=in_channels)
        self.b2 = nn.BatchNorm2d(num_features=hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
        return h + x


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1, nl_layer=F.relu,
                 downsample=False):
        super().__init__()
        self.downsample = downsample
        self.activation = nl_layer
        self.learnable_sc = in_channels != out_channels or downsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.c1 = spectral_norm(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad))
        self.c2 = spectral_norm(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = _downsample(x)
        return h + x


class OptiResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1, nl_layer=F.relu):
        super().__init__()
        self.activation = nl_layer
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.c1 = spectral_norm(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad))
        self.c2 = spectral_norm(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad))
        self.c_sc = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        x = self.c_sc(_downsample(x))
        return h + x


class ResNetDiscriminator(nn.Module):
    def __init__(self, ndf, self_attention=True, input_nc=3):
        super().__init__()
        self.self_attention = self_attention
        self.activation = F.relu
        self.block1 = OptiResBlockDown(input_nc, ndf, nl_layer=self.activation)
        self.block2 = ResBlockDown(
            1 * ndf, 2 * ndf, nl_layer=self.activation, downsample=True)
        if self_attention:
            self.attention_size = (2 * ndf) // 8
            self.att = SelfAttention(2 * ndf, self.attention_size)
            self.att_post = SelfAttentionPost(2 * ndf, self.attention_size)
        self.block3 = ResBlockDown(
            2 * ndf, 4 * ndf, nl_layer=self.activation, downsample=True)
        self.block4 = ResBlockDown(
            4 * ndf, 8 * ndf, nl_layer=self.activation, downsample=True)
        self.block5 = ResBlockDown(
            8 * ndf, 16 * ndf, nl_layer=self.activation, downsample=True)
        self.l6 = spectral_norm(nn.Linear(ndf * 16, 1))

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        if self.self_attention:
            attention_output = self.att(h)
            h = self.att_post(h, attention_output)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))  # Global pooling
        h = torch.sigmoid(self.l6(h))
        return h


class ResNetGenerator(nn.Module):
    def __init__(self, ngf, self_attention=True, add_input=False,
                 input_nc=3, output_nc=3):
        super().__init__()
        self.self_attention = self_attention

        self.activation = F.relu
        self.add_input = add_input
        self.block1 = ResBlockUp(input_nc, ngf, nl_layer=self.activation)
        self.block2 = ResBlockUp(1 * ngf, 16 * ngf, nl_layer=self.activation)
        self.block3 = ResBlockUp(16 * ngf, 8 * ngf, nl_layer=self.activation)
        self.block4 = ResBlockUp(8 * ngf, 4 * ngf, nl_layer=self.activation)
        self.block5 = ResBlockUp(4 * ngf, 2 * ngf, nl_layer=self.activation)
        if self_attention:
            self.attention_size = (2 * ngf) // 8
            self.att = SelfAttention(2 * ngf, self.attention_size)
            self.att_post = SelfAttentionPost(2 * ngf, self.attention_size)
        self.block6 = ResBlockUp(2 * ngf, ngf, nl_layer=self.activation)
        self.b7 = nn.BatchNorm2d(ngf)
        self.l8 = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x, z=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        if self.self_attention:
            self.attention_output = self.att(h)
            h = self.att_post(h, self.attention_output)
        h = self.block6(h)

        h = self.b7(h)
        h = self.activation(h)
        if self.add_input:
            return torch.tanh(self.l8(h)) + x
        else:
            return torch.tanh(self.l8(h))
