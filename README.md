# Retrieval-Augmented Machine Translation (kNN-MT)

CENG 467 — Natural Language Understanding and Generation  
Izmir Institute of Technology | Spring 2026  
Student: Zehra Eda Cesur (310201089)

## Project Description
Implementation of kNN-MT (Khandelwal et al., 2021) for English-to-German translation on the IWSLT 2017 dataset.

## Results
| System | BLEU | METEOR | BERTScore |
|--------|------|--------|-----------|
| Baseline 1: IWSLT-trained NMT (beam=5) | 32.10 | 0.5216 | 0.8721 |
| kNN-MT (λ=0.05, K=8, T=10, greedy) | 13.04 | 0.3310 | 0.7582 |

## Requirements
pip install fairseq sacrebleu sentencepiece datasets bert-score nltk faiss-cpu

## Reproduction Steps
1. Open `CENG467_kNN_MT.ipynb` in Google Colab
2. Set runtime to GPU (T4)
3. Run all cells in order
4. Dataset: IWSLT 2017 En-De (auto-downloaded via HuggingFace)

## References
- Khandelwal et al. (2021). Nearest Neighbor Machine Translation. ICLR 2021.
- Vaswani et al. (2017). Attention Is All You Need. NeurIPS 2017.
- Ott et al. (2019). fairseq. NAACL 2019.
