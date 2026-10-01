"""Assemble the existing compact encoder and decoder into an AMT model."""

import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

from .encoder import CompactEncoder
from .decoder import CompactDecoder


class CompactModel(nn.Module):
    def __init__(self, config, vocabulary_size, frame_loss_weight=0.0):
        super().__init__()
        self.frontend = MelSpectrogram(**config["frontend"])
        self.encoder = CompactEncoder(config["encoder"], config["frontend"]["n_mels"])
        self.decoder = CompactDecoder(config["decoder"])
        dim = config["decoder"]["dim"]
        self.embed_tokens = nn.Embedding(vocabulary_size, dim)
        self.lm_head = nn.Linear(dim, vocabulary_size, bias=False)
        self.frame_head = None
        if frame_loss_weight > 0:
            hidden = config["frame_head_dim"]
            self.frame_head = nn.Sequential(
                nn.Linear(config["encoder"]["channel_dim"], hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 128),
            )

    def encode(self, audio):
        # Preserve YourMT3's power mel -> log -> per-example min/max normalization.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            mel = self.frontend(audio.float().squeeze(1)).transpose(1, 2)
            mel = torch.log(mel + 1e-5)
            low = mel.amin(dim=(1, 2), keepdim=True)
            high = mel.amax(dim=(1, 2), keepdim=True)
            mel = torch.nan_to_num((mel - low) / (high - low + 0.008))
        return self.encoder(mel)

    def forward(self, audio, tokens):
        context = self.encode(audio)
        shifted = torch.zeros_like(tokens)
        shifted[..., 1:] = tokens[..., :-1].clamp_min(0)
        hidden, _ = self.decoder(self.embed_tokens(shifted), context)
        frame_logits = None if self.frame_head is None else self.frame_head(context)
        return {"logits": self.lm_head(hidden), "frame_logits": frame_logits}

    @torch.no_grad()
    def generate(self, audio, max_length=256):
        context = self.encode(audio)
        batch, channels = context.shape[:2]
        current = torch.zeros(
            (batch, channels, 1), device=audio.device, dtype=torch.long
        )
        finished = torch.zeros((batch, channels), device=audio.device, dtype=torch.bool)
        result, past = [], None
        for _ in range(max_length):
            hidden, past = self.decoder(self.embed_tokens(current), context, past)
            current = self.lm_head(hidden).argmax(-1)
            current = current.masked_fill(finished[..., None], 0)
            result.append(current)
            finished |= current[..., 0].eq(1)
            if finished.all():
                break
        return torch.cat(result, dim=-1)
