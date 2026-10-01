"""Explicit construction; adding a combination needs config, not a Dataset subclass."""

from .corruption import CorruptionSampler
from .rendered import AudioMidiSampleBuilder, SeededRenderedDataset
from renderers.modartt import ModarttRenderer
from renderers.nsynth import NSynthRenderer
from renderers.nsynth_bank import NSynthBank


def build_online(config, sources, seed):
    online = config["online"]
    sr = config["audio"]["sample_rate"]
    # Source objects are assembled by the caller and may be shared across routes.
    routes, banks = {}, []
    for name, spec in online["routes"].items():
        sampler_config = spec["sampler"]
        source = sources[spec["source"]]
        if sampler_config["type"] == "corruption":
            sampler = CorruptionSampler(
                source,
                sampler_config["statistics"],
                sampler_config,
                online["render_seconds"],
            )
        else:
            raise ValueError(f"unknown sampler: {sampler_config['type']}")
        renderer_config = spec["renderer"]
        fallback = None
        if renderer_config["type"] == "nsynth":
            bank = NSynthBank(renderer_config["root"], sr, **renderer_config["bank"])
            renderer = NSynthRenderer(bank, renderer_config)
            banks.append(bank)
        elif renderer_config["type"] in {"pianoteq", "organteq"}:
            renderer = ModarttRenderer(renderer_config, sr)
            if renderer_config["type"] == "pianoteq":
                fallback = renderer.piano_fallback
        else:
            raise ValueError(f"unknown renderer: {renderer_config['type']}")
        routes[name] = {
            "sampler": sampler,
            "renderer": renderer,
            "weight": spec["weight"],
            "fallback": fallback,
        }
    builder = AudioMidiSampleBuilder(
        sr, config["audio"]["input_frames"], online["segment_seconds"]
    )
    dataset = SeededRenderedDataset(routes, builder, seed)
    return dataset, banks
