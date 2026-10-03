"""AMT: offline/online producers -> cache -> cross-aug -> tokens -> model -> Adam."""

import json
import os
import random
import logging
import sys
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel

from data.datasets import OfflineDataset
from data.cache import ClipCache, allocate_counts
from data.augment import augment_batch
from models.model import CompactModel
from tokenization.targets import TargetTokenizer
from training.loss import make_batch, loss_parts
from training.evaluation import autocast, evaluate
from training.checkpoint import load_checkpoint, restore_random_state, save_checkpoint
from training.logging import capture_output


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=60)
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return device, rank, world


def validate_config(config):
    t, cache = config["training"], config["cache"]
    if t["precision"] not in {"bf16", "float32"}:
        raise ValueError("training.precision must be bf16 or float32")
    if not 0 < t["warmup_steps"] < t["max_steps"]:
        raise ValueError("require 0 < warmup_steps < max_steps")
    if t["frame_loss_weight"] < 0:
        raise ValueError("frame_loss_weight must be nonnegative")
    if t["save_every"] <= 0 or t["log_every"] <= 0 or t["eval_every"] < 0:
        raise ValueError(
            "save_every/log_every must be positive; eval_every may be zero"
        )
    if t["stop_after_steps"] is not None and t["stop_after_steps"] <= 0:
        raise ValueError("stop_after_steps must be positive or null")
    if cache["segments_per_source"] <= 0 or cache["producer_workers"] < 0:
        raise ValueError("segments_per_source must be positive and producer_workers nonnegative")
    aug = config["augmentation"]
    if (
        not 0 <= aug["stem_keep_probability"] <= 1
        or aug["max_k"] < 0
        or aug["tau"] < 0
        or aug["alpha"] <= 0
        or aug["max_stems"] <= 0
    ):
        raise ValueError("invalid augmentation probabilities or limits")
    ev = config["evaluation"]
    if ev["batch_size"] <= 0 or ev["train_clips_per_dataset"] < 0:
        raise ValueError("invalid evaluation batch size or train clip count")
    if any(
        ev[k] is not None and ev[k] <= 0 for k in ("max_files", "max_segments_per_file")
    ):
        raise ValueError("evaluation limits must be positive or null")
    if (
        not 0 < t["batch_size"] <= cache["capacity"]
        or not 0 < cache["refresh_clips"] <= cache["capacity"]
    ):
        raise ValueError("batch_size and refresh_clips must fit in cache.capacity")
    if config["run"]["resume"] and config["run"]["weights"]:
        raise ValueError("choose run.resume or run.weights, not both")
    if config["model"]["encoder"]["output_channels"] != 13:
        raise ValueError("the fixed mc13 tokenizer needs 13 encoder output channels")
    if (
        config["model"]["decoder"]["input_dim"]
        != config["model"]["encoder"]["channel_dim"]
    ):
        raise ValueError("decoder input_dim must match encoder channel_dim")
    online = config.get("online")
    if online is not None:
        if (
            online["producer_workers"] < 0
            or online["refresh_bank_every"] <= 0
            or online["refresh_bank_sources"] < 0
        ):
            raise ValueError("invalid online worker or bank refresh settings")
        sr = config["audio"]["sample_rate"]
        stride = round(online["segment_seconds"] * sr)
        frames = round(online["render_seconds"] * sr)
        if (
            stride < config["audio"]["input_frames"]
            or frames < stride
            or frames % stride
        ):
            raise ValueError(
                "online render duration must contain full training segments"
            )
        names = {s["name"] for s in config["data"]["train"]}
        if names.intersection(online["routes"]):
            raise ValueError(
                "online route names must differ from offline dataset names"
            )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    with capture_output(config["run"]["output_dir"]):
        train(config)


def train(config):
    validate_config(config)
    device, rank, world = setup_distributed()
    torch.set_num_threads(4)
    seed = config["run"]["seed"] + rank * 1000003
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        logging.info(
            f"Starting {config['run']['name']} on {device} (world_size={world})",
        )
    t = config["training"]
    tokenizer = TargetTokenizer(**config["tokenization"])
    model = CompactModel(
        config["model"], tokenizer.num_tokens, t["frame_loss_weight"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.5, total_iters=t["warmup_steps"]
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t["max_steps"] - t["warmup_steps"],
        eta_min=t["min_learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[t["warmup_steps"]]
    )
    step, resume_state = 0, None
    if config["run"]["resume"]:
        checkpoint = load_checkpoint(
            config["run"]["resume"], model, optimizer, scheduler, config
        )
        step, resume_state = checkpoint["step"], checkpoint["ranks"][rank]
    elif config["run"]["weights"]:
        load_checkpoint(config["run"]["weights"], model)
    dataset = OfflineDataset(
        config["data_root"],
        config["data"]["train"],
        config["audio"]["input_frames"],
        config["audio"]["sample_rate"],
        config["cache"]["segments_per_source"],
        seed,
    )
    producers = {"offline": dataset}
    online, banks, producer_options = None, [], {}
    if config.get("online") is not None:
        from data.build import build_online
        from data.sources import MidiSource

        sources = {
            "training": MidiSource(
                dataset.entries, dataset.probabilities, config["audio"]["sample_rate"]
            )
        }
        online, banks = build_online(config, sources, seed + 500003)
        producers["online"] = online
        producer_options["online"] = {"producer_workers": config["online"]["producer_workers"]}
    weights = config["cache"]["partition_weights"] or {name: 1 for name in producers}
    allocate_counts(t["batch_size"], weights)
    cache, wandb_run = None, None
    try:
        if rank == 0:
            OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml")
            logging.info(
                "Dataset sampling probabilities: %s",
                dict(zip(dataset.entries, dataset.probabilities)),
            )
            logging.info(
                f"Model: {sum(p.numel() for p in model.parameters()):,} parameters; device={device}; world_size={world}",
            )
            if config["wandb"]["mode"] != "disabled":
                import wandb

                wandb_run = wandb.init(
                    project=config["wandb"]["project"],
                    name=config["run"]["name"],
                    dir=str(output_dir.resolve()),
                    mode=config["wandb"]["mode"],
                    config=config,
                )
        cache = ClipCache(
            producers,
            config["cache"],
            seed,
            None if resume_state is None else resume_state["producer_index"],
            producer_options,
        )
        if resume_state is not None:
            restore_random_state(resume_state, cache, rng)
        train_model = torch.compile(model) if t["compile"] else model
        if dist.is_initialized():
            train_model = DistributedDataParallel(
                train_model,
                device_ids=[device.index] if device.type == "cuda" else None,
            )

        def log(values):
            if rank != 0:
                return
            record = {"step": step, **values}
            logging.info("metrics %s", json.dumps(record, sort_keys=True))
            with (output_dir / "metrics.log").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if wandb_run is not None:
                wandb_run.log(values, step=step)

        stop = (
            t["max_steps"]
            if t["stop_after_steps"] is None
            else min(t["max_steps"], step + t["stop_after_steps"])
        )
        model.train()
        last_evaluation = -1
        while step < stop:
            samples = augment_batch(
                cache.sample(t["batch_size"]), cache, config["augmentation"], rng
            )
            batch = make_batch(samples, tokenizer, config, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, t["precision"]):
                parts = loss_parts(train_model(batch["audio"], batch["tokens"]), batch)
                loss = 0
                for name, (total, count) in parts.items():
                    count = count.clone()
                    if dist.is_initialized():
                        dist.all_reduce(count)
                    weight = t["frame_loss_weight"] if name == "frame" else 1
                    # DDP averages gradients: compensate to obtain a global token mean.
                    loss = loss + total * (world * weight) / count.clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), t["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            step += 1
            if step == 1 or step % t["log_every"] == 0 or step == stop:
                logged = loss.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(logged)
                values = {
                    "train/loss": float(logged) / world,
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if "frame" in parts:
                    frame = torch.stack([v.detach().double() for v in parts["frame"]])
                    if dist.is_initialized():
                        dist.all_reduce(frame)
                    values["train/frame_loss"] = float(frame[0] / frame[1].clamp_min(1))
                log(values)
            # Save before evaluation so an evaluation failure cannot discard
            # the training progress at a checkpoint step.
            if step % t["save_every"] == 0 or step == stop:
                save_checkpoint(
                    output_dir / f"step-{step:07d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    config,
                    cache,
                    rng,
                )
            if t["eval_every"] and step % t["eval_every"] == 0:
                log(evaluate(model, tokenizer, config, device, dataset.entries, online))
                last_evaluation = step
            if step < stop:
                cache.refill()
                if (
                    online is not None
                    and step % config["online"]["refresh_bank_every"] == 0
                ):
                    for bank in banks:
                        bank.refresh(config["online"]["refresh_bank_sources"])
        if config["evaluation"]["at_end"] and last_evaluation != step:
            log(evaluate(model, tokenizer, config, device, dataset.entries, online))
    except BaseException:
        logging.exception("Training or evaluation failed")
        raise
    finally:
        if cache is not None:
            cache.close()
        if wandb_run is not None:
            wandb_run.finish()
        # Let torchrun terminate peers on failure; a collective teardown here
        # can hide the original exception behind another rank's pending reduce.
        if dist.is_initialized() and sys.exc_info()[0] is None:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
