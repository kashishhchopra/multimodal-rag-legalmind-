# Dataset card: LegalMind training pairs (`data_lake/pairs/pairs.jsonl`)

Built by `backend/scripts/build_training_pairs.py`. Not published; this card documents how it is made.

| Source | Query | Positive | Licence of source |
|---|---|---|---|
| issue | "Issue for Consideration" text of a Supreme Court judgment | best-matching reasoning chunk of the same judgment (kept only if BM25 over the pool returns it in the top 20) | CC-BY-4.0 (AWS Open Data, Dattam Labs) |
| citation | passage that cites "YYYY INSC N" | headnote/issue chunk of the cited judgment, when that judgment is in the corpus | CC-BY-4.0 |
| cuad | clause category + description | annotated clause span | CC-BY-4.0 (CUAD, train split only) |
| ledgar | "<provision type> clause" | provision text | LexGLUE/LEDGAR, SEC filings; verify before redistribution |

- **Splits:** judgments by year (train <= 2022, validation 2023, test >= 2024); a judgment never appears in two splits (asserted in the builder). CUAD/LEDGAR contribute train splits only, so their test sets stay sealed.
- **Hard negatives:** BM25 top non-gold chunks plus the neighbouring chunk of the same judgment (issue pairs only).
- **Known biases:** only Supreme Court of India, 2019-2024, English, native-text PDFs (scanned PDFs are queued for OCR, not yet included). Issue statements share vocabulary with their judgments, which makes issue-pair retrieval easier than real user queries. Judgments contain personal data of private individuals; no redaction is applied to the corpus, only to memo exports (identifiers only).
- **Non-commercial data is NOT mixed in:** IL-TUR (CC-BY-NC-SA), ILDC and HLDC (CC-BY-NC) are catalogued in `DATASETS.md` but excluded from these pairs.
