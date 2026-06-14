# Retrieval-Augmented Machine Translation (kNN-MT)

CENG 467 — Natural Language Understanding and Generation
Izmir Institute of Technology | Spring 2026
Student: Zehra Eda Cesur (310201089)

## Project Description
Implementation of kNN-MT (Khandelwal et al., 2021) for English-to-German
translation on IWSLT 2017, with two datastores (IWSLT and Europarl)
and beam search integration via output logit injection.

## Results

| System | BLEU | chrF | TER | METEOR | ROUGE-L | BERTScore |
|--------|------|------|-----|--------|---------|-----------|
| Baseline 1 (beam=5) | 32.10 | 57.71 | 54.98 | 0.5216 | 0.5937 | 0.8721 |
| Baseline 1 (greedy) | 13.74 | 40.92 | 115.76 | 0.3437 | 0.3900 | 0.7654 |
| kNN-MT + IWSLT (greedy) | 13.04 | 39.48 | 118.13 | 0.3310 | 0.3785 | 0.7582 |
| kNN-MT + Europarl (greedy) | 14.22 | 41.36 | 111.45 | 0.3461 | 0.3929 | 0.7671 |
| kNN-MT + Europarl (beam=5) | 29.24 | 57.31 | 61.74 | 0.5128 | 0.5636 | 0.8556 |

## Requirements
pip install fairseq sacrebleu sentencepiece datasets bert-score nltk faiss-cpu rouge-score

## Reproduction Steps
1. Open `CENG467_kNN_MT.ipynb` in Google Colab
2. Set runtime to GPU (T4)
3. Mount Google Drive
4. Run all cells in order
5. Datasets: IWSLT 2017 En-De (HuggingFace), Europarl En-De (statmt.org)

## Repository Structure
- `CENG467_kNN_MT.ipynb` — Full pipeline notebook
- `results/` — All metric CSVs and hypothesis files
- `scripts/` — Preprocessing and training scripts
- `data/` — Dataset download scripts

## References
- Khandelwal et al. (2021). Nearest Neighbor Machine Translation. ICLR 2021.
- Vaswani et al. (2017). Attention Is All You Need. NeurIPS 2017.
- Ott et al. (2019). fairseq. NAACL 2019.
- Koehn (2005). Europarl. MT Summit 2005.
