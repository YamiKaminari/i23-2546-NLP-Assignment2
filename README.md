# CS-4063 NLP Assignment 2 — Neural NLP Pipeline
## FAST NUCES | BBC Urdu Corpus

---

## Repository Structure

```
i23-XXXX-NLP-Assignment2/
├── NLP_Assignment2.ipynb       ← Main notebook (all cells must be executed)
├── README.md                   ← This file
├── embeddings/
│   ├── tfidf_matrix.npy        ← TF-IDF term-document matrix (N × 10000)
│   ├── ppmi_matrix.npy         ← PPMI co-occurrence matrix (5000 × 5000)
│   ├── embeddings_w2v.npy      ← Averaged Skip-gram embeddings ½(V+U)
│   └── word2idx.json           ← Vocabulary token→index mapping
├── models/
│   ├── bilstm_pos.pt           ← BiLSTM POS tagger (fine-tuned embeddings)
│   ├── bilstm_ner.pt           ← BiLSTM+CRF NER tagger (fine-tuned)
│   └── transformer_cls.pt      ← Transformer topic classifier
├── data/
│   ├── pos_train.conll         ← POS CoNLL training set
│   ├── pos_test.conll          ← POS CoNLL test set
│   ├── ner_train.conll         ← NER CoNLL training set
│   └── ner_test.conll          ← NER CoNLL test set
└── figures/                    ← All generated plots (auto-created)
    ├── tsne_ppmi.png
    ├── w2v_loss.png
    ├── pos_training.png
    ├── ner_training.png
    ├── pos_confusion.png
    ├── transformer_training.png
    ├── transformer_confusion.png
    └── attn_heatmap_ex{1,2,3}.png
```

---

## Prerequisites

```bash
pip install torch numpy matplotlib scikit-learn
```

Python 3.8+ and PyTorch 1.12+ recommended. CUDA optional but speeds up training.

---

## Required Input Files

Place these files in the **same directory as the notebook**:

| File | Used In | Purpose |
|------|---------|---------|
| `cleaned.txt` | All parts | Primary training corpus (one doc per line) |
| `raw.txt` | Parts 1 & 2 | Ablation baseline |
| `Metadata.json` | Part 3 | Topic labels for classification |

---

## How to Reproduce

### Option A — Jupyter Notebook (recommended)

```bash
jupyter notebook NLP_Assignment2.ipynb
# Run All Cells: Kernel → Restart & Run All
```

### Option B — Command line (nbconvert)

```bash
pip install nbconvert
jupyter nbconvert --to notebook --execute NLP_Assignment2.ipynb \
    --output NLP_Assignment2_executed.ipynb
```

**Expected runtime:** ~30–60 minutes on CPU, ~10–20 minutes on GPU.

---

## Part Overview

### Part 1 — Word Embeddings [25 marks]
- **Section 1.1**: TF-IDF weighted term-document matrix → `tfidf_matrix.npy`
- **Section 1.2**: PPMI co-occurrence matrix + t-SNE plot → `ppmi_matrix.npy`
- **Section 2.1**: Skip-gram Word2Vec (from scratch, PyTorch) → `embeddings_w2v.npy`
- **Section 2.2**: Nearest-neighbour eval, analogy tests, 4-condition MRR comparison

### Part 2 — BiLSTM Sequence Labeling [25 marks]
- **Section 3**: 500 sentence annotation (rule-based POS + NER BIO), CoNLL files
- **Section 4**: 2-layer BiLSTM, CRF+Viterbi for NER, frozen vs fine-tuned embeddings
- **Section 5**: POS accuracy/F1/confusion matrix, NER conlleval scores, ablations A1–A4

### Part 3 — Transformer Classification [20 marks]
- **Section 6**: 5-class topic dataset from `Metadata.json`
- **Section 7**: From-scratch Transformer (scaled dot-product attn, multi-head, FFN, sinusoidal PE, 4 Pre-LN encoder blocks, CLS token)
- **Section 8**: Test metrics, confusion matrix, attention heatmaps, BiLSTM vs Transformer analysis

---

## Key Design Decisions

| Restriction | Implementation |
|-------------|---------------|
| No HuggingFace / Gensim | Pure PyTorch throughout |
| No `nn.Transformer` | Custom `EncoderBlock` with Pre-LN |
| No `nn.MultiheadAttention` | Custom `MultiHeadSelfAttention` with per-head projections |
| No pretrained models | W2V trained from scratch on corpus |

---

## GitHub Commit Checklist

- [ ] Initial project structure
- [ ] Part 1: TF-IDF and PPMI implementation
- [ ] Part 1: Skip-gram Word2Vec training
- [ ] Part 2: Annotation pipeline and BiLSTM model
- [ ] Part 3: Transformer encoder implementation and evaluation
