"""Token supervision and the optional auxiliary frame objective."""

import numpy as np
import torch
import torch.nn.functional as F

from tokenization.vocabulary import NUM_CHANNELS, PROGRAM_TO_CHANNEL


def token_loss(logits, tokens, allowed_ids):
    """Return sum and count, so DDP and uneven evaluation batches reduce correctly.

    A coarse program label supervises the *sum* of its allowed probabilities.
    Exact labels use ordinary cross entropy. PAD=0 never contributes.
    """
    logits = logits.float().reshape(-1, logits.shape[-1])
    targets = tokens.reshape(-1)
    valid = targets.ne(0)
    losses = F.cross_entropy(logits, targets, ignore_index=0, reduction="none")
    allowed = allowed_ids.reshape(-1, allowed_ids.shape[-1])
    partial = allowed.ge(0).any(-1) & valid
    if partial.any():
        choices = allowed[partial]
        log_probs = F.log_softmax(logits[partial], dim=-1).gather(
            -1, choices.clamp_min(0)
        )
        losses[partial] = -torch.logsumexp(
            log_probs.masked_fill(choices.lt(0), -torch.inf), dim=-1
        )
    return losses.sum(), valid.sum()


def frame_targets(samples, frames, fps):
    targets = np.zeros((len(samples), NUM_CHANNELS, frames, 128), dtype=np.float32)
    mask = np.ones_like(targets, dtype=bool)
    for row, sample in enumerate(samples):
        if sample["has_unannotated"]:
            mask[row] = False
        active = {(e.program, e.pitch): 0 for e in sample["tie_note_events"]}

        def fill(program, pitch, first, last):
            first = max(0, min(frames, first))
            last = max(first + 1, min(frames, last))
            targets[row, PROGRAM_TO_CHANNEL[program], first:last, pitch] = 1

        for event in sorted(
            sample["note_events"],
            key=lambda e: (e.time, e.is_drum, e.program, e.pitch, e.velocity),
        ):
            position = int(np.floor(event.time * fps + 1e-8))
            key = event.program, event.pitch
            if event.is_drum:
                if event.velocity:
                    fill(128, event.pitch, position, position + 1)
            elif event.velocity:
                if key in active:
                    fill(*key, active[key], position)
                active[key] = position
            elif key in active:
                fill(*key, active.pop(key), position)
        for (program, pitch), first in active.items():
            fill(program, pitch, first, frames)
    return targets, mask


def make_batch(samples, tokenizer, config, device):
    tokens, allowed = tokenizer.encode(samples)
    batch = {
        "audio": torch.from_numpy(
            np.stack([s["audio"].sum(axis=0, keepdims=True) for s in samples])
        ),
        "tokens": torch.from_numpy(tokens),
        "allowed_ids": torch.from_numpy(allowed),
    }
    if config["training"]["frame_loss_weight"] > 0:
        frontend = config["model"]["frontend"]
        values, mask = frame_targets(
            samples,
            config["audio"]["input_frames"] // frontend["hop_length"] + 1,
            frontend["sample_rate"] / frontend["hop_length"],
        )
        batch.update(
            frame_targets=torch.from_numpy(values), frame_mask=torch.from_numpy(mask)
        )
    return {key: value.to(device) for key, value in batch.items()}


def loss_parts(output, batch):
    total, count = token_loss(output["logits"], batch["tokens"], batch["allowed_ids"])
    result = {"token": (total, count)}
    if "frame_targets" in batch:
        mask = batch["frame_mask"]
        values = F.binary_cross_entropy_with_logits(
            output["frame_logits"].float(), batch["frame_targets"], reduction="none"
        )
        result["frame"] = ((values * mask).sum(), mask.sum())
    return result
