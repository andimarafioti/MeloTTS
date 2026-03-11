import importlib
import os
import sys
import types

import numpy as np
import pytest


def _distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def _stub_non_english_language_modules():
    stubbed_names = [
        "melo.text.chinese",
        "melo.text.japanese",
        "melo.text.chinese_mix",
        "melo.text.korean",
        "melo.text.french",
        "melo.text.spanish",
    ]
    originals = {name: sys.modules.get(name) for name in stubbed_names}

    for name in stubbed_names:
        module = types.ModuleType(name)
        module.text_normalize = lambda text: text
        module.g2p = lambda text: (["a"], [0], [1])
        module.get_bert_feature = lambda text, word2ph, device=None: None
        module.distribute_phone = _distribute_phone
        sys.modules[name] = module

    return originals


def _restore_modules(originals):
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.mark.skipif(
    os.getenv("MELO_RUN_REAL_MODEL_TESTS") != "1",
    reason="Set MELO_RUN_REAL_MODEL_TESTS=1 to run real checkpoint synthesis tests.",
)
def test_real_english_model_synthesizes_on_current_torch():
    originals = _stub_non_english_language_modules()
    sys.modules.pop("melo.api", None)
    try:
        api_module = importlib.import_module("melo.api")
        tts = api_module.TTS(language="EN", device="cpu")
        audio = tts.tts_to_file(
            "This is a real synthesis smoke test on modern PyTorch.",
            speaker_id=0,
            quiet=True,
        )
    finally:
        sys.modules.pop("melo.api", None)
        _restore_modules(originals)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.shape[0] > 1000
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 0.0
