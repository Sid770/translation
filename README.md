# Sentence Alignment using GRU Encoders

This repository aligns Marathi and Hindi sentences (e.g., from the Bhagwat Geeta) using GRU-based sentence encoders and cosine similarity.

## Files

- `main.py`: Main script to process, embed, and align sentences.
- `model.py`: GRU encoder definition.
- `utils.py`: Tokenization, vocabulary, and utility functions.
- `requirements.txt`: Python dependencies.
- `data/marathi.txt`: Marathi input sentences (one per line).
- `data/hindi.txt`: Hindi input sentences (one per line).

## Usage

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**

   Fill `data/marathi.txt` and `data/hindi.txt` with sentences (one sentence per line, aligned or not).

3. **Run alignment**

   ```bash
   python main.py
   ```

   This will output an `aligned_sentence_pairs.csv` with the best-aligned sentence pairs and their cosine similarities.

## Notes
- This code assumes random initialized encoders. For better alignment, you may want to pretrain encoders or use cross-lingual pretrained embeddings.
- The process is unsupervised and based only on sentence embeddings and cosine similarity.
