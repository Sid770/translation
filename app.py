import torch
import json
import re
import pdfplumber
from model import EncoderGRU
from utils import build_vocab, tokenize, encode_sentence, cosine_similarity_topk

# ---------- CONFIG ----------
MARATHI_PDF = "B:\OneDrive - Amity University\Desktop\Indus\sid_work\The Holy Bible in Marathi-1-4.pdf"
HINDI_PDF = "B:\OneDrive - Amity University\Desktop\Indus\sid_work\The Holy Hindi 4 pages.pdf"
EMBEDDING_DIM = 256
HIDDEN_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- PDF READING & SENTENCE SPLITTING ----------
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def split_into_sentences(text):
    # Basic sentence splitter; adjust for your language/script as needed
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

marathi_text = extract_text_from_pdf(MARATHI_PDF)
hindi_text = extract_text_from_pdf(HINDI_PDF)

marathi_lines = split_into_sentences(marathi_text)
hindi_lines = split_into_sentences(hindi_text)

# ---------- VOCAB ----------
marathi_vocab = build_vocab(marathi_lines)
hindi_vocab = build_vocab(hindi_lines)

# ---------- MODELS ----------
encoder_marathi = EncoderGRU(len(marathi_vocab), EMBEDDING_DIM, HIDDEN_SIZE).to(DEVICE)
encoder_hindi = EncoderGRU(len(hindi_vocab), EMBEDDING_DIM, HIDDEN_SIZE).to(DEVICE)

encoder_marathi.eval()
encoder_hindi.eval()

# ---------- EMBED SENTENCES ----------
def embed_sentences(sentences, encoder, vocab):
    encoder.eval()
    all_embeds = []
    with torch.no_grad():
        for sent in sentences:
            ids = encode_sentence(tokenize(sent), vocab)
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
            _, h = encoder(ids_tensor)
            all_embeds.append(h.squeeze(0).cpu())
    return torch.stack(all_embeds)

marathi_embeds = embed_sentences(marathi_lines, encoder_marathi, marathi_vocab)
hindi_embeds = embed_sentences(hindi_lines, encoder_hindi, hindi_vocab)

# ---------- ALIGNMENT ----------
aligned_pairs = cosine_similarity_topk(marathi_embeds, hindi_embeds, marathi_lines, hindi_lines, top_k=1)

# ---------- OUTPUT TO JSON ----------
aligned_json = []
for marathi, hindi, sim in aligned_pairs:
    aligned_json.append({
        "marathi": marathi,
        "hindi": hindi,
        "cosine_similarity": sim
    })

with open("aligned_sentence_pairs.json", "w", encoding="utf-8") as f:
    json.dump(aligned_json, f, ensure_ascii=False, indent=2)

print("Aligned sentence pairs saved to aligned_sentence_pairs.json")