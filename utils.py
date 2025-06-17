import re
import torch
import numpy as np
from collections import Counter

def tokenize(sentence):
    # Simple whitespace tokenizer, can be replaced with more language-specific tokenizers
    return sentence.strip().split()

def build_vocab(lines, min_freq=1):
    counter = Counter()
    for line in lines:
        tokens = tokenize(line)
        counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def encode_sentence(tokens, vocab):
    return [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]

def cosine_similarity_topk(m_embeds, h_embeds, m_lines, h_lines, top_k=1):
    # m_embeds: (N, D)
    # h_embeds: (M, D)
    m_norm = m_embeds / m_embeds.norm(dim=1, keepdim=True)
    h_norm = h_embeds / h_embeds.norm(dim=1, keepdim=True)
    sim_matrix = torch.mm(m_norm, h_norm.t())  # (N, M)
    results = []
    for i, sims in enumerate(sim_matrix):
        topk_vals, topk_idx = torch.topk(sims, k=top_k)
        for val, idx in zip(topk_vals, topk_idx):
            results.append((m_lines[i], h_lines[idx], float(val)))
    return results