import sys

import torch
from transformers import AutoModel


def resolve_bert_device(device):
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    return device


def load_hidden_state_model(model_id, device):
    return AutoModel.from_pretrained(model_id).to(device)
