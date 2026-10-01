from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def make_norm(dim: int, norm_type: str) -> nn.Module:
    if norm_type == "rms_norm":
        return nn.RMSNorm(dim)
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unsupported norm type: {norm_type}")


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, config: dict) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        positions = torch.arange(config.get("max_positions", 8192), dtype=torch.float32)
        inv_freq = 1.0 / (
            float(config["rope_base"])
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, value: Tensor, offset: int = 0) -> Tensor:
        length = value.shape[-2]
        if offset + length > self.cos.shape[0]:
            raise ValueError(
                f"RoPE position {offset + length} exceeds {self.cos.shape[0]}"
            )
        cos = self.cos[offset : offset + length].to(dtype=value.dtype)[None, None]
        sin = self.sin[offset : offset + length].to(dtype=value.dtype)[None, None]
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)


class FeedForward(nn.Module):
    def __init__(self, dim: int, config: dict) -> None:
        super().__init__()
        hidden = dim * int(config["ffn_widening_factor"])
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(hidden, dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, config: dict) -> None:
        super().__init__()
        if dim % config["n_heads"]:
            raise ValueError("attention dimension must be divisible by n_heads")
        self.n_heads = int(config["n_heads"])
        self.head_dim = dim // self.n_heads
        self.dropout = float(config["dropout_rate"])
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.rope = (
            RotaryEmbedding(self.head_dim, config)
            if config["position_encoding_type"] == "rope"
            else None
        )

    def forward(
        self,
        value: Tensor,
        *,
        causal: bool,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, dim = value.shape
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        if self.rope is not None:
            q = self.rope(q, past_length)
            k = self.rope(k, past_length)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)
        # Incremental decoding has one new query which may attend to all cached keys.
        is_causal = causal and past_key_value is None
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).reshape(batch, length, dim)
        return self.out(output), (k, v)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, config: dict) -> None:
        super().__init__()
        if dim % config["n_heads"]:
            raise ValueError("attention dimension must be divisible by n_heads")
        self.n_heads = int(config["n_heads"])
        self.head_dim = dim // self.n_heads
        self.dropout = float(config["dropout_rate"])
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(context_dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, value: Tensor, context: Tensor) -> Tensor:
        batch, length, dim = value.shape
        context_length = context.shape[1]
        q = (
            self.q(value)
            .view(batch, length, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(batch, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, context_length, self.n_heads, self.head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(output.transpose(1, 2).reshape(batch, length, dim))
