import torch
from transformers import AutoTokenizer

from .bert_utils import load_hidden_state_model, resolve_bert_device

model_id = 'bert-base-uncased'
tokenizer = None
model = None

def get_bert_feature(text, word2ph, device=None):
    global model, tokenizer
    device = resolve_bert_device(device)
    if model is None:
        model = load_hidden_state_model(model_id, device)
    else:
        model = model.to(device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
