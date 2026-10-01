# RandomScore AMT

An independent AMT implementation with the compact encoder/decoder, YourMT3
note tokenization, offline audio, and online sampler–renderer generation.
All runtime code lives in this directory.

## Run

Use Python 3.10 or newer. Install a matching PyTorch/torchaudio pair for your CUDA
runtime, then install `requirements.txt`. Run commands from this directory.
`DATA_ROOT` points to the prepared data tree containing `yourmt3_indexes/` and
the audio/annotation directories. Existing YourMT3 `.npy` annotations are read
with a local class mapping; the old source tree is not needed.

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/music_data
python train.py experiment=0901_offline
torchrun --standalone --nproc_per_node=2 train.py experiment=0901_offline
```

On this workspace, `.venv/bin/python` uses the installed music-transcription
environment plus a project-local Hydra installation. `.venv/` is not released.

Hydra composes `configs/config.yaml` with `model/`, `data/`, and an optional
`experiment/` override. `experiment=0901_offline` selects the offline recipe;
`model=compact_small` and `data=slakh` can be selected independently. Every run
saves its resolved `config.yaml`. Modules receive ordinary dictionaries.

For a short real-data run without changing the learning-rate schedule:

```bash
python train.py experiment=0901_offline run.name=smoke \
  training.compile=false training.batch_size=2 training.stop_after_steps=2 \
  cache.capacity=64 cache.refresh_clips=8 cache.workers=2 \
  evaluation.train_clips_per_dataset=1 evaluation.max_files=1 \
  evaluation.max_segments_per_file=1 wandb.mode=disabled
```

The small evaluation limits above are only for checking the pipeline. Remove
them for actual evaluation. The offline recipe preserves its historical split
selection: Slakh/MAESTRO training includes their validation splits; MultiTpop
evaluation uses dev, and MAPS/Slakh evaluation uses test.

## Read and modify

Follow `train.py` from `OfflineDataset` through `ClipCache`, `augment_batch`,
`TargetTokenizer`, and `CompactModel`. There is one explicit training loop,
Adam optimizer, linear warmup from half the peak learning rate, and cosine decay.

* `data/datasets.py`: JSON indexes, audio crops, boundary TIEs, and source sampling.
  Add a combination by editing the config's dataset list, splits, and weights.
* `data/cache.py`: per-rank CPU clip storage with bounded FIFO replacement;
  fresh clips are preferred as bases and distinct source files are used as donors.
  DataLoader workers only read sources; mixing/tokenization happen in the rank
  process. No shared mutable cache or sample-dictionary wrapper is needed.
* `data/augment.py`: retain stems, select cross-source donors, exclude overlapping
  instruments/drums, bound event count, regroup stems, and mix with random gains.
* `tokenization/`: YourMT3 token order, event codec, 13 channel groups, and inverse
  conversion. The default vocabulary has 596 tokens; each channel has length 256.
* `models/encoder.py` and `models/decoder.py`: independent PyTorch modules.
  `CompactEncoder.encode_features(mel)` exposes unprojected temporal features;
  `forward(mel)` includes the existing channel projection. `models/model.py`
  assembles the frontend, encoder, decoder, embeddings, and prediction head.
* `training/loss.py`: exact-label CE and optional partial-label CE. An ordinary
  `label_constraint` dictionary specifies allowed programs or drum pitches.
  Piano/guitar datasets may identify a family but not its fine instrument class;
  their loss is `-log(sum(probability of allowed tokens))`. Precise labels use CE.

The numeric dataset weights follow the historical YourMT3 rule: each file in
dataset i receives `weight_i * (1 - n_i / N)`, then weights are normalized.
They are not direct dataset probabilities. Inspect the resulting probabilities
and split counts with `python -m scripts.stat_dataset`.

## Online composition

`MidiSource` preloads each indexed MIDI once into compact shared note arrays.
`CorruptionSampler` samples a clip and corrupts its attributes using measured
marginals. The source has no eviction, refresh, or computed-property wrappers.

`data/build.py` selects implementations with explicit `if/elif` branches and
assembles a dictionary of named sampler–renderer routes. All combinations use
the same `SeededRenderedDataset` and `AudioMidiSampleBuilder`. To add a
combination, add a route in Hydra config; do not add a Dataset subclass.
Sources are passed in as a dictionary of objects; each route's `source` selects
one of those objects. The training entrypoint supplies the `training` MIDI source.

```text
preloaded MidiSource -> CorruptionSampler -> selected renderer
    -> actual rendered notes + stems + label sets
    -> AudioMidiSampleBuilder -> seeded training segments
```

NSynth uses `renderers/prune_audio.py` (the v3 algorithm only). Its bounded
waveform bank is separate from the preloaded MIDI symbols. The renderer returns
actual fallback pitches and drops silent notes from supervision. Pianoteq and
Organteq share process restart, timeout and error handling in `renderers/process.py`;
measured capability dictionaries determine usable preset/pitch pairs. Permanent
configuration errors propagate; only explicitly unrenderable samples are retried.
NSynth keeps the sampled fine program for teacher forcing while its loss allows
the whole family. A family stem containing several fine programs is treated as
inseparable during augmentation; Synth Pad targets use the Synth Lead channel.

Online audio is rendered for 8.192 seconds and split into four 2.048-second
segments; each model input contains 32767 samples, matching the existing frontend.
TIEs come from the rendered note intervals. Each route has a visible config name,
which is also its evaluation metric name.

```bash
export CORRUPTION_STATISTICS=/path/to/slakh_train_8.192s_marginals.json
export NSYNTH_ROOT=/path/to/Nsynth
python train.py online=nsynth run.name=offline_nsynth

export PIANOTEQ_BIN=/path/to/Pianoteq
export PIANOTEQ_CAPABILITIES=/path/to/pianoteq_capabilities.json
python train.py experiment=0912_mixed
```

To measure new marginals, use `python -m scripts.stat_midi --root "$DATA_ROOT"
--dataset slakh --split train --output artifacts/slakh_marginals.json`. This
samples bounded MIDI windows and bins times to 1 ms; it does not claim to
recreate a historical statistics artifact. Use that original artifact when
reproducing the 0912 configuration.

The 0912 recipe selects the same eight offline datasets, a 1:1 offline/online
base quota, equal NSynth/Pianoteq route weights, four clips per online render,
and MultiTpop test. Its historical 64 GiB NSynth bank needs that much available
shared memory; override `online.nsynth.bank.size_gib` for smaller machines.
NSynth bank refresh changes the resident source pool, so a seed alone does not
freeze timbre selection across refreshes or checkpoint restarts.

`ClipCache` stores audio in explicit offline/online partitions. The weights set
base-batch, prefill and refresh quotas, which must be integral. Cross-augmentation
draws donors from all partitions, so final mixed audio need not have that ratio.

The model and token semantics are retained; the data pipeline has been simplified.
FIFO eviction, independently seeded source reads, and the local augmentation RNG
do not reproduce the old asynchronous pipeline's exact sample sequence.

## Metrics and checkpoints

`metrics.log` and optional W&B contain `train/loss`, `train/lr`,
`eval/train/<dataset-or-route>/loss`, `eval/test/<dataset>/loss`, and
`eval/test/<dataset>/note_f1`. The latter is the per-file mean non-drum,
instrument-agnostic onset F1 (50 ms / 50 cent; offsets ignored). Files without
pitched reference notes are excluded from that mean. Training summaries use
fixed unaugmented clips; test losses cover the selected files' segments. Online
training summaries evaluate each configured route separately with fixed seeds.
Python logging, warnings and exceptions use the existing `FileHandler` in
`run.log`. This is not a universal stdout/stderr capture layer.
Evaluation loss excludes programs outside the fixed decoder vocabulary (for
example program 118 in MultiTpop), with a logged warning. The complete reference
is retained for instrument-agnostic Note F1. Training target validation stays strict.

Frame supervision is optional. At `training.frame_loss_weight=0`, no frame head,
targets, loss, or frame metrics are created. A positive scalar enables them and
adds `train/frame_loss`. Old metric names are not double-written.

```bash
# Resume optimizer, scheduler, global step and per-rank RNG states.
python train.py experiment=0901_offline run.resume=outputs/0901_offline/last.pt

# Start a new optimization run from model weights.
python train.py experiment=0901_offline run.name=finetune \
  run.weights=outputs/0901_offline/last.pt

# Evaluate a native checkpoint using the same model config as training.
python evaluate.py experiment=0901_offline run.name=evaluation \
  run.weights=outputs/0901_offline/last.pt
```

Checkpoints are written atomically at `training.save_every` steps and when a
short run ends. `last.pt` points to the latest periodic file. Resume requires
the same architecture, world size, and learning-rate schedule. It rebuilds the
cache from the saved cursor of each producer; queued/cache samples are not checkpointed,
so continuation is not a bitwise replay. This stage does not load Lightning
checkpoints.

DDP reduces token sums/counts before backward and shards evaluation files without
padding duplicates. Only rank zero writes logs and checkpoints. Batch size,
cache capacity, and producer workers are **per rank**.

YourMT3-derived code retains its Apache 2.0 attribution; see `NOTICE` and `LICENSE`.
