# LegalMind Evaluation Report

Not yet run. Populate `eval/testset.jsonl` with real `doc_id`s and gold
`chunk_id`s from an ingested corpus, then run:

```bash
python eval/run_ragas.py
```

This file is overwritten by that script with the latest metrics: RAGAS
faithfulness and context precision, hand-labelled citation accuracy, P95
latency, and a breakdown of citation accuracy by ingestion path (native vs.
OCR vs. vision) to check the OCR-doc answer parity target.
