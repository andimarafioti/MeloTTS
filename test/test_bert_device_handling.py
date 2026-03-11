import importlib
from types import SimpleNamespace

import torch


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        return FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})


class FakeModel:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, **kwargs):
        hidden = torch.ones((1, 3, 4))
        return {"hidden_states": [hidden, hidden, hidden, hidden]}


def test_explicit_cpu_does_not_promote_to_mps(monkeypatch):
    module = importlib.import_module("melo.text.english_bert")
    bert_utils = importlib.import_module("melo.text.bert_utils")
    fake_model = FakeModel()

    monkeypatch.setattr(module, "tokenizer", FakeTokenizer())
    monkeypatch.setattr(module, "model", None)
    monkeypatch.setattr(module, "load_hidden_state_model", lambda model_id, device: fake_model.to(device))
    monkeypatch.setattr(bert_utils.torch.backends, "mps", SimpleNamespace(is_available=lambda: True))
    monkeypatch.setattr(bert_utils.sys, "platform", "darwin")

    features = module.get_bert_feature("abc", [1, 1, 1], device="cpu")

    assert features.shape == (4, 3)
    assert fake_model.devices == ["cpu"]


def test_cached_model_moves_to_requested_device(monkeypatch):
    module = importlib.import_module("melo.text.english_bert")
    fake_model = FakeModel()

    monkeypatch.setattr(module, "tokenizer", FakeTokenizer())
    monkeypatch.setattr(module, "model", fake_model)

    features = module.get_bert_feature("abc", [1, 1, 1], device="cpu")

    assert features.shape == (4, 3)
    assert fake_model.devices == ["cpu"]
