import re
import torch
import numpy as np
from collections import Counter

def tokenize(sentence):
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

def align_top1(m_embeds, h_embeds, m_lines, h_lines):
    # Normalize embeddings
    m_norm = m_embeds / m_embeds.norm(dim=1, keepdim=True)
    h_norm = h_embeds / h_embeds.norm(dim=1, keepdim=True)
    sim_matrix = torch.mm(m_norm, h_norm.t())  # (N, M)
    results = []
    for i, sims in enumerate(sim_matrix):
        top_val, top_idx = torch.max(sims, 0)
        results.append({
            "marathi": m_lines[i],
            "hindi": h_lines[top_idx],
            "cosine_similarity": float(top_val)
        })
    return results
