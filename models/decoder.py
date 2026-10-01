"""Shared Transformer decoder; channels are independent members of the batch."""

from torch import nn

from .layers import CrossAttention, FeedForward, SelfAttention, make_norm


class DecoderBlock(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.norm1 = make_norm(dim, config["norm_type"])
        self.self_attention = SelfAttention(dim, config)
        self.norm2 = make_norm(dim, config["norm_type"])
        self.cross_attention = CrossAttention(dim, dim, config)
        self.norm3 = make_norm(dim, config["norm_type"])
        self.ffn = FeedForward(dim, config)

    def forward(self, x, context, past=None):
        attended, present = self.self_attention(
            self.norm1(x), causal=True, past_key_value=past
        )
        x = x + attended
        x = x + self.cross_attention(self.norm2(x), context)
        return x + self.ffn(self.norm3(x)), present


class CompactDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["dim"]
        self.context_projection = nn.Linear(config["input_dim"], dim)
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, config) for _ in range(config["layers"])]
        )
        self.final_norm = make_norm(dim, config["norm_type"])

    def forward(self, inputs_embeds, encoder_hidden_states, past_key_values=None):
        batch, channels, length, dim = inputs_embeds.shape
        x = inputs_embeds.reshape(batch * channels, length, dim)
        context = self.context_projection(encoder_hidden_states).flatten(0, 1)
        presents = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            x, present = block(x, context, past)
            presents.append(present)
        return self.final_norm(x).view(batch, channels, length, dim), tuple(presents)
