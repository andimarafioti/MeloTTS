import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


class FakeSynthesizer:
    def __init__(self, *args, **kwargs):
        self.device = "cpu"
        self.state_dict = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def load_state_dict(self, state_dict, strict=True):
        self.state_dict = state_dict

    def infer(
        self,
        x_tst,
        x_tst_lengths,
        speakers,
        tones,
        lang_ids,
        bert,
        ja_bert,
        sdp_ratio=0.2,
        noise_scale=0.6,
        noise_scale_w=0.8,
        length_scale=1.0,
    ):
        assert x_tst.device.type == self.device
        assert tones.device.type == self.device
        assert lang_ids.device.type == self.device
        assert bert.device.type == self.device
        assert ja_bert.device.type == self.device
        assert x_tst_lengths.device.type == self.device
        assert speakers.device.type == self.device

        audio = torch.linspace(-0.25, 0.25, 32, device=x_tst.device, dtype=torch.float32)
        return audio.view(1, 1, -1), None


def fake_hps():
    return SimpleNamespace(
        num_languages=1,
        num_tones=1,
        symbols=["_", "a", "b", "c"],
        data=SimpleNamespace(
            filter_length=8,
            hop_length=2,
            sampling_rate=44100,
            n_speakers=1,
            spk2id={"EN-Default": 0},
        ),
        train=SimpleNamespace(segment_size=8),
        model={},
    )


def import_api_module_with_fakes():
    utils_module = ModuleType("melo.utils")
    utils_module.get_text_for_tts_infer = lambda text, language, hps, device, symbol_to_id: (
        torch.zeros(1024, 3),
        torch.zeros(768, 3),
        torch.tensor([1, 2, 3]),
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 0, 0]),
    )

    models_module = ModuleType("melo.models")
    models_module.SynthesizerTrn = FakeSynthesizer

    split_module = ModuleType("melo.split_utils")
    split_module.split_sentence = lambda text, language_str: [text]

    mel_processing_module = ModuleType("melo.mel_processing")
    mel_processing_module.spectrogram_torch = lambda *args, **kwargs: None
    mel_processing_module.spectrogram_torch_conv = lambda *args, **kwargs: None

    download_utils_module = ModuleType("melo.download_utils")
    download_utils_module.load_or_download_config = lambda *args, **kwargs: fake_hps()
    download_utils_module.load_or_download_model = (
        lambda *args, **kwargs: {"model": {"weights": torch.tensor([1.0])}}
    )

    originals = {}
    fake_modules = {
        "melo.utils": utils_module,
        "melo.models": models_module,
        "melo.split_utils": split_module,
        "melo.mel_processing": mel_processing_module,
        "melo.download_utils": download_utils_module,
    }

    for name, module in fake_modules.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    sys.modules.pop("melo.api", None)
    try:
        return importlib.import_module("melo.api")
    finally:
        sys.modules.pop("melo.api", None)
        for name, module in fake_modules.items():
            original = originals[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_tts_to_file_runs_end_to_end_on_modern_torch():
    api_module = import_api_module_with_fakes()

    tts = api_module.TTS(language="EN", device="cpu")
    audio = tts.tts_to_file("This works on current torch.", speaker_id=0, quiet=True)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.shape[0] > 0
    assert np.isfinite(audio).all()


def test_tts_auto_device_still_runs_with_modern_torch(monkeypatch):
    api_module = import_api_module_with_fakes()

    monkeypatch.setattr(api_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(api_module.torch.backends, "mps", SimpleNamespace(is_available=lambda: False))

    tts = api_module.TTS(language="EN", device="auto")
    audio = tts.tts_to_file("Auto device also works.", speaker_id=0, quiet=True)

    assert tts.device == "cpu"
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
