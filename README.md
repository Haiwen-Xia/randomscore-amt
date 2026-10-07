# RandomScore AMT

This repository is a research-oriented automatic music transcription (AMT)
codebase related to [*Randomized Scores and Diverse Timbres: Augmenting
Automatic Music Transcription with Online-Generated Data*](paper.pdf). It is
designed for experiments on how symbolic score variation and timbral coverage
affect transcription beyond the training datasets.

The central idea is simple: a **sampler** chooses the notes, and a **renderer**
turns them into labeled audio during training. Offline recordings can be mixed
with these newly generated pairs. The model is a compact, 13-channel
autoregressive transcriber with a shared decoder. The implementation favors
readable components and inspectable outputs so that data-generation choices can
be studied and changed directly. It also simplifies much of YourMT3's training
logic into an explicit PyTorch loop with a smaller set of data, augmentation,
evaluation, and checkpoint modules.

## How it is organized

```text
Prepared audio + notes ------> offline dataset --+
                                                +--> clip cache --> augmentation
MIDI --> sampler-renderer group --> online data -+       --> tokens --> AMT model
```

| Area | What it contains |
| --- | --- |
| `data/` | Prepared dataset indexes, MIDI sources, score corruption, offline/online sampling, clip cache, and audio augmentation. |
| `renderers/` | NSynth note rendering and Pianoteq/Organteq stem rendering. |
| `tokenization/` | Note-event vocabulary, 13-channel targets, and decoding back to notes. |
| `models/` | Log-mel frontend, convolutional/Transformer encoder, channel split, and shared autoregressive decoder. |
| `training/` | Loss, validation, logging, and native PyTorch checkpoints. |
| `configs/` | Composable model, dataset, sampler, renderer, and experiment settings. |

`train.py` connects these pieces in one training loop. Offline and online
examples use the same augmentation, tokenizer, and model. Each named online
sampler-renderer group pairs a symbolic sampler with an audio renderer and has
its own sampling weight. Groups are combined in configuration.

## Set up from a fresh machine

Use Linux, Python 3.10 or newer, and a GPU with a compatible PyTorch build for
training. Create an environment and install PyTorch and torchaudio for your CUDA
version using the [official PyTorch installer](https://pytorch.org/get-started/locally/).
Then install this project's dependencies:

```bash
git clone https://github.com/Haiwen-Xia/randomscore-amt.git
cd randomscore-amt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the matching torch + torchaudio pair for your machine first.
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

CPU execution is supported for inspection and small checks, but full training
and benchmark inference are intended for a GPU. Commands below assume the
environment is activated and the current directory is this repository.

### Prepare offline data

Set `DATA_ROOT` to a directory where you want to keep datasets. The loader
expects 16 kHz audio, note annotations, and JSON indexes in
`$DATA_ROOT/yourmt3_indexes/`. These can be produced with the
[YourMT3 dataset installer](https://github.com/mimbres/YourMT3), which is needed
for preparation only; training does not import YourMT3.

```bash
export DATA_ROOT="$HOME/music_data"
git clone https://github.com/mimbres/YourMT3.git ../YourMT3
cd ../YourMT3/amt/src
python -m pip install -r requirements.txt
python install_dataset.py "$DATA_ROOT"
# Enter 1,2,4,5,6,7,12,13 when prompted for the default training mixture.
# Include 3 for MAPS evaluation. Use --nodown if source files are already present.
cd ../../../randomscore-amt
```

The eight training datasets are Slakh, MusicNet, MAESTRO, GuitarSet, ENST-drums,
EGMD, URMP, and IDMT-SMT-Bass. The default offline recipe also evaluates MAPS
and MultiTpop, so prepare both before using it. Check that, for example,
`$DATA_ROOT/yourmt3_indexes/slakh_train_file_list.json` exists before running.
For a smaller first run, prepare only Slakh (installer choice `1`) and select
`data=slakh` below. The complete dataset selection and evaluation splits are
in `configs/data/`.

MultiTpop and RWC require their own prepared audio, aligned MIDI, and indexes.
They are not installed by the eight-dataset command above. Obtain MultiTpop's
metadata and aligned MIDI from its [dataset release](https://gclef-cmu.org/multtipop/)
and follow the dataset's instructions for sourcing audio. MultiTpop uses
`multtipop_dev_file_list.json` for the default offline recipe and
`multtipop_test_file_list.json` for the mixed recipe and benchmark. Prepare
those splits as YourMT3-style 16 kHz audio and note annotations, and put their
indexes under `$DATA_ROOT/yourmt3_indexes/`. MAPS is installer choice `3`; RWC
requires separate access to its source data.

## Train

Start with the offline recipe once the default data mixture is prepared:

```bash
python train.py experiment=0901_offline run.name=my_offline_run
```

For a short Slakh check with modest cache and evaluation limits:

```bash
python train.py data=slakh run.name=slakh_check \
  training.compile=false training.batch_size=2 training.stop_after_steps=2 \
  cache.capacity=64 cache.refresh_clips=8 cache.producer_workers=2 \
  evaluation.train_clips_per_dataset=1 evaluation.max_files=1 \
  evaluation.max_segments_per_file=1 wandb.mode=disabled
```

The defaults write a resolved config, `metrics.log`, `run.log`, and checkpoints
to `outputs/<run.name>/`. W&B is optional: set `wandb.mode=disabled` to avoid
creating a run. For distributed training, launch the same entry point with
`torchrun --standalone --nproc_per_node=<GPU_COUNT> train.py ...`.

### Add online-rendered audio

The sampler starts from indexed training MIDI and can retain source notes or
corrupt their attributes using the included
`artifacts/slakh_train_8.192s_marginals.json`. The renderer supplies the timbre.
Each 8.192-second rendering yields four 2.048-second training segments.

For NSynth, download the **train JSON/WAV archive** from the
[NSynth dataset page](https://magenta.tensorflow.org/datasets/nsynth), extract
it, and point `NSYNTH_ROOT` to the extraction directory or its train directory.
The TFRecord archive does not supply the individual WAV files this renderer
reads.

```bash
export NSYNTH_ROOT=/path/to/nsynth-train
python train.py online=nsynth run.name=offline_nsynth
```

The mixed recipe uses offline datasets, NSynth, and Pianoteq. It supports the
**free Pianoteq 9.1.2 version** used in the related paper; a paid license is not
required by the renderer. Download Pianoteq from
[Modartt](https://www.modartt.com/pianoteq_overview) and point `PIANOTEQ_BIN`
to its Linux standalone executable. The checked-in
`artifacts/pianoteq_capabilities.json` records measured playable preset/pitch
pairs for Pianoteq 9.1.2, including its unavailable pitches. If using a
different version or preset collection, measure a new catalog and set
`PIANOTEQ_CAPABILITIES` to its path; the bundled catalog should not be assumed
to describe another release.

```bash
export PIANOTEQ_BIN='/path/to/Pianoteq 9'
export NSYNTH_ROOT=/path/to/nsynth-train
python train.py experiment=0912_mixed run.name=my_mixed_run
```

The mixed recipe uses a 64 GiB NSynth waveform bank by default. On a machine
with less shared memory, set `online.nsynth.bank.size_gib=<available_size>` and
adjust the batch/cache settings as needed. Pianoteq and Organteq binaries are
not distributed here. The free Pianoteq executable can restart between render
attempts; only successfully rendered notes are used as targets.

## Inspect and evaluate

Export examples from the actual training pipeline before a long experiment:

```bash
python -m scripts.export_inspect data=slakh run.name=inspection
```

`outputs/inspection/inspect/` contains 30 numbered WAV/MIDI/JSON sets and a
manifest. The JSON records source files or online routes, crop positions, and
the retained components of augmented mixtures. Change `inspection.count` to
export fewer or more examples.

To compare a native checkpoint on RWC, MAPS, and MultiTpop after those datasets
are prepared, run:

```bash
python benchmark.py outputs/my_mixed_run/step-0320000.pt --cache-notes
```

The benchmark uses the training evaluator's instrument-agnostic onset note F1
(50 ms onset tolerance, offsets ignored). It writes per-track scores and a
dataset summary under `outputs/benchmarks/<run.name>/`; `--cache-notes` saves
predicted notes for resuming or rescoring. `--limit N` selects at most N tracks
per dataset or RWC subset, and `--max-segments N` evaluates an audio prefix for
short checks. Omit both limits for the full suite. The dataset selection is
editable in `configs/benchmark/rwc_maps_multtipop.yaml`.

Training also logs evaluation losses and test note F1 at the configured
interval. Resume an interrupted run with `run.resume=<checkpoint>`; use
`run.weights=<checkpoint>` to initialize a new run from model weights. The
standalone `evaluate.py` uses the same Hydra training configuration for
validation.

## Citation and license

If this repository supports your research, please cite the accompanying
[paper](paper.pdf). YourMT3-derived components retain their Apache 2.0
attribution in [NOTICE](NOTICE) and [LICENSE](LICENSE). Dataset and renderer
downloads follow their respective providers' terms.
