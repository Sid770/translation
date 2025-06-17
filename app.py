import torch
import json
import re
import pytesseract
from pdf2image import convert_from_path
from model import EncoderGRU
from utils import build_vocab, tokenize, encode_sentence, align_top1

MARATHI_PDF = "B:\OneDrive - Amity University\Desktop\Indus\sid_work\The Holy Bible in Marathi-1-4.pdf"
HINDI_PDF = "B:\OneDrive - Amity University\Desktop\Indus\sid_work\The Holy Hindi 4 pages.pdf"
EMBEDDING_DIM = 256
HIDDEN_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_text_from_pdf_ocr(pdf_path, lang="eng"):
    pages = convert_from_path(pdf_path)
    text = ""
    for img in pages:
        text += pytesseract.image_to_string(img, lang=lang) + "\n"
    return text

def split_into_sentences(text):
    sentences = re.split(r'(?<=[\u0964.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

marathi_text = extract_text_from_pdf_ocr(MARATHI_PDF, lang="mar")
hindi_text = extract_text_from_pdf_ocr(HINDI_PDF, lang="hin")

marathi_lines = split_into_sentences(marathi_text)
hindi_lines = split_into_sentences(hindi_text)

marathi_vocab = build_vocab(marathi_lines)
hindi_vocab = build_vocab(hindi_lines)

encoder_marathi = EncoderGRU(len(marathi_vocab), EMBEDDING_DIM, HIDDEN_SIZE).to(DEVICE)
encoder_hindi = EncoderGRU(len(hindi_vocab), EMBEDDING_DIM, HIDDEN_SIZE).to(DEVICE)

encoder_marathi.eval()
encoder_hindi.eval()

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

aligned_pairs = align_top1(marathi_embeds, hindi_embeds, marathi_lines, hindi_lines)

with open("aligned_sentence_pairs.json", "w", encoding="utf-8") as f:
    json.dump(aligned_pairs, f, ensure_ascii=False, indent=2)

print("Aligned sentence pairs saved to aligned_sentence_pairs.json")
