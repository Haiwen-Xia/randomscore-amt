"""Native PyTorch checkpoints; all ranks participate, only rank zero writes."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def save_checkpoint(
    path, model, optimizer, scheduler, step, config, cache, augmentation_rng
):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "cache_rng": cache.rng.bit_generator.state,
        "augmentation_rng": augmentation_rng.bit_generator.state,
        "producer_index": cache.produced_items,
    }
    ranks = [state]
    if dist.is_initialized():
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, state)
    if not dist.is_initialized() or dist.get_rank() == 0:
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "format_version": 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "config": config,
                "ranks": ranks,
            },
            temporary,
        )
        os.replace(temporary, path)
        last = path.parent / "last.pt"
        if path != last:
            # Atomic link replacement avoids writing a second multi-GB checkpoint.
            link = path.parent / "last.tmp"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(path.name)
            os.replace(link, last)
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(path, model, optimizer=None, scheduler=None, config=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("expected a randomscore-amt checkpoint")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        # Changing these silently would change the meaning of optimizer continuation.
        for key in ("model", "audio", "tokenization"):
            if state["config"][key] != config[key]:
                raise ValueError(
                    f"resume requires unchanged {key}; use run.weights for finetuning"
                )
        for key in (
            "learning_rate",
            "min_learning_rate",
            "warmup_steps",
            "max_steps",
            "frame_loss_weight",
        ):
            if state["config"]["training"][key] != config["training"][key]:
                raise ValueError(
                    f"resume requires unchanged training.{key}; use run.weights for finetuning"
                )
        world = dist.get_world_size() if dist.is_initialized() else 1
        if len(state["ranks"]) != world:
            raise ValueError(
                "resume requires the original world size; use run.weights to change it"
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    return state


def restore_random_state(state, cache, augmentation_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])
    cache.rng.bit_generator.state = state["cache_rng"]
    augmentation_rng.bit_generator.state = state["augmentation_rng"]
