"""Convolutional audio encoder with a Transformer core and channel projection."""

from torch import nn
import torch.nn.functional as F

from .layers import FeedForward, SelfAttention, make_norm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=(1, 2))


class EncoderTransformerBlock(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.norm1 = make_norm(dim, config["norm_type"])
        self.attention = SelfAttention(dim, config)
        self.norm2 = make_norm(dim, config["norm_type"])
        self.ffn = FeedForward(dim, config)

    def forward(self, x):
        attended, _ = self.attention(self.norm1(x), causal=False)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class CompactEncoder(nn.Module):
    def __init__(self, config, input_mels=128):
        super().__init__()
        channels = [1, *config["conv_layers"]]
        self.conv = nn.Sequential(
            *[ConvBlock(a, b) for a, b in zip(channels[:-1], channels[1:])]
        )
        conv_dim = channels[-1] * (input_mels // 2 ** (len(channels) - 1))
        dim = config["dim"]
        self.input_projection = nn.Linear(conv_dim, dim)
        self.core = nn.ModuleList(
            [EncoderTransformerBlock(dim, config) for _ in range(config["core_layers"])]
        )
        self.output_channels = config["output_channels"]
        self.channel_dim = config["channel_dim"]
        self.projection = nn.Linear(dim, self.output_channels * self.channel_dim)

    def encode_features(self, mel):
        """[batch, time, mel] -> [batch, time, dim], also usable outside AMT."""
        x = self.conv(mel.unsqueeze(1))
        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.input_projection(x)
        for block in self.core:
            x = block(x)
        return x

    def forward(self, mel):
        """[batch, time, mel] -> [batch, channel, time, channel_dim]."""
        x = self.projection(self.encode_features(mel))
        batch, frames, _ = x.shape
        return x.view(batch, frames, self.output_channels, self.channel_dim).permute(
            0, 2, 1, 3
        )
