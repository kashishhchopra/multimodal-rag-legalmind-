# Notebooks

Use this folder for exploratory work that doesn't belong in `src/`:

- EDA over the sample corpus (page counts, native vs. scanned split, clause
  numbering styles you need `structure_splitter.py` to handle).
- Ablations — e.g. OpenAI vs. BGE embeddings, with/without reranking,
  chunk-size sweeps — feeding into `eval/report.md`.
- One-off eval runs and result inspection before promoting anything stable
  into `eval/run_ragas.py`.

Keep notebooks numbered and named for what they answer, e.g.
`01_corpus_eda.ipynb`, `02_embedding_ablation.ipynb`.
