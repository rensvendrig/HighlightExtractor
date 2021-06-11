import torch
import numpy as np


class Encoder(object):
    def __init__(self,sent_filter,measurement):
        self.measurement=measurement
        self.sent_filter=sent_filter
        return

""" This is copied from the 'A Visual Guide to Using BERT for the First Time' (https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) '"""
def get_embedding(model, tokenizer, input_text):
    if isinstance(input_text, list):
        tokenized = input_text.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    elif isinstance(input_text, str):
        tokenized = tokenizer.encode(input_text, add_special_tokens=True)

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    input_ids = torch.tensor(padded)
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        embedding = model(input_ids, attention_mask=attention_mask)

    return embedding
