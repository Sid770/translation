import torch
import json
import re
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------- CONFIG ----------
MARATHI_PDF = "B:\\OneDrive - Amity University\\Desktop\\Indus\\sid_work\\The Holy Bible in Marathi-1-4.pdf"
HINDI_PDF = "B:\\OneDrive - Amity University\\Desktop\\Indus\\sid_work\\The Holy Hindi 4 pages.pdf"
SIMILARITY_THRESHOLD = 0.6  # Only save pairs above this similarity
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------- PDF READING & SENTENCE SPLITTING ----------
import pytesseract
from pdf2image import convert_from_path

def extract_text_from_pdf_with_ocr(pdf_path, lang='hin'):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang=lang) + "\n"
    return text



def split_into_sentences(text):
    # Basic sentence splitter; adjust for your language/script as needed
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

marathi_text = extract_text_from_pdf_with_ocr(MARATHI_PDF)
hindi_text = extract_text_from_pdf_with_ocr(HINDI_PDF)

marathi_lines = split_into_sentences(marathi_text)
hindi_lines = split_into_sentences(hindi_text)

# ---------- EMBEDDING MODEL ----------
print("Loading SentenceTransformer model...")
model = SentenceTransformer(MODEL_NAME)

# ---------- EMBED SENTENCES ----------
print("Embedding Marathi sentences...")
marathi_embeds = model.encode(marathi_lines, convert_to_tensor=True, show_progress_bar=True)
print("Embedding Hindi sentences...")
hindi_embeds = model.encode(hindi_lines, convert_to_tensor=True, show_progress_bar=True)

# ---------- ALIGNMENT ----------
def cosine_similarity_topk(m_embeds, h_embeds, m_lines, h_lines, top_k=1):
    m_norm = m_embeds / m_embeds.norm(dim=1, keepdim=True)
    h_norm = h_embeds / h_embeds.norm(dim=1, keepdim=True)
    sim_matrix = torch.mm(m_norm, h_norm.t())  # (N, M)
    results = []
    for i, sims in enumerate(sim_matrix):
        topk_vals, topk_idx = torch.topk(sims, k=top_k)
        for val, idx in zip(topk_vals, topk_idx):
            results.append((m_lines[i], h_lines[idx], float(val)))
    return results

aligned_pairs = cosine_similarity_topk(marathi_embeds, hindi_embeds, marathi_lines, hindi_lines, top_k=1)

# ---------- FILTER SIMILAR PAIRS ----------
def filter_high_similarity_pairs(aligned_pairs, threshold=0.6):
    return [(m, h, sim) for (m, h, sim) in aligned_pairs if sim >= threshold]

filtered_pairs = filter_high_similarity_pairs(aligned_pairs, threshold=SIMILARITY_THRESHOLD)

# ---------- OUTPUT TO JSON ----------
aligned_json = []
for marathi, hindi, sim in filtered_pairs:
    aligned_json.append({
        "marathi": marathi,
        "hindi": hindi,
        "cosine_similarity": sim
    })

with open("aligned_sentence_pairs2.json", "w", encoding="utf-8") as f:
    json.dump(aligned_json, f, ensure_ascii=False, indent=2)

print(f"Aligned sentence pairs with similarity >= {SIMILARITY_THRESHOLD} saved to aligned_sentence_pairs2.json")

# ---------- NOTE ----------
# This code uses a SentenceTransformer multilingual model for proper cross-language semantic similarity.
# For best results, ensure you have 'sentence-transformers' and 'torch' installed.
