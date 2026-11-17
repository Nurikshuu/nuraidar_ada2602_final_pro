# Nuraidar Final Programming

Final project: Tiny Transformer for legal act classification (Kazakh text).

## Description

The project classifies Kazakh legal acts by their form (`act_form`, e.g. *Шешім* = decision, *Қаулы* = resolution, *Бұйрық* = order) using only the **title** of the act.
Dataset: [KazakhLawCorpus-clean](https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus-clean) (about 215.9K documents).
It is downloaded automatically from Hugging Face inside the notebook with `load_dataset`, so no data files are stored in this repository.

Three approaches are compared on a temporal split (train on older acts, test on the newest ones):

1. TF-IDF (word and character n-grams) + Logistic Regression (baseline)
2. A small Transformer trained from scratch with a custom BPE tokenizer (up to 10M parameters; the actual model has 4.7M)
3. Fine-tuned `intfloat/multilingual-e5-small` (118M parameters)

Author: Mambetaly Nuraidar, ADA-2602M.

## Results

7 classes, main metric macro-F1. Test = acts adopted after 2024-08-28 (cleaned of exact duplicates).

| Model | Params | Size, MB | Val F1-macro | Test F1-macro | Test accuracy |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | 1.7M | 24.5 | 0.902 | **0.905** | 0.958 |
| Tiny Transformer (from scratch) | 4.7M | 18.6 | 0.847 | **0.852** | 0.951 |
| multilingual-e5-small (fine-tuned) | 117.7M | 465.2 | 0.915 | **0.894** | 0.956 |

The TF-IDF baseline is the best on test and is also the fastest. See the notebook for details, strict-test scores and error analysis.

## Notebook structure

0. Setup
1. Data loading
2. EDA (distribution, text length, duplicates, class balance)
3. Preprocessing and temporal train/val/test split
4. Data leakage checks (exact duplicates, near-duplicates via cosine similarity on character n-grams)
5. SQLite experiment log
6. Baseline: TF-IDF + Logistic Regression
7. Training loop for transformers
8. Transformer from scratch with BPE tokenizer
9. Fine-tuning multilingual-e5-small
10. Model comparison
11. Error analysis
12. Predictions for custom input text
13. Interactive input

## How to run

The notebook runs in Google Colab, VS Code or Jupyter. A GPU is recommended for section 9 (training on CPU is much slower).

```
pip install datasets tokenizers transformers scikit-learn
```

Other libraries used: numpy, pandas, matplotlib, joblib, torch. Then open `Nuraidar_FINAL.ipynb` and run all cells.

## Files

- `Nuraidar_FINAL.ipynb` - full project with code, outputs and results
- `README.md` - this file
- `results/` - small result files of the training runs (experiment database, learning curve, metadata)
