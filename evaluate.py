"""Evaluate a native checkpoint with Hydra data/evaluation overrides."""

import json
from pathlib import Path

import hydra
import torch
import logging
import sys
import torch.distributed as dist
from omegaconf import OmegaConf

from models.model import CompactModel
from tokenization.targets import TargetTokenizer
from training.checkpoint import load_checkpoint
from training.evaluation import evaluate
from train import setup_distributed, validate_config
from training.logging import capture_output


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    with capture_output(config["run"]["output_dir"]):
        run_evaluation(config)


def run_evaluation(config):
    validate_config(config)
    torch.set_num_threads(4)
    path = config["run"]["weights"] or config["run"]["resume"]
    if not path:
        raise ValueError("set run.weights to the checkpoint to evaluate")
    device, rank, _ = setup_distributed()
    try:
        tokenizer = TargetTokenizer(**config["tokenization"])
        model = CompactModel(
            config["model"],
            tokenizer.num_tokens,
            config["training"]["frame_loss_weight"],
        ).to(device)
        load_checkpoint(path, model)
        metrics = evaluate(model, tokenizer, config, device)
        if rank == 0:
            output = Path(config["run"]["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(config), output / "evaluation_config.yaml")
            (output / "evaluation.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            logging.info("evaluation metrics %s", json.dumps(metrics, sort_keys=True))
    finally:
        if dist.is_initialized() and sys.exc_info()[0] is None:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
