# Sample corpus

- `native/` — clean, text-layer PDFs (contracts exported from Word, e-signed
  agreements, etc.). These exercise the deterministic PyMuPDF/pdfplumber path.
- `scanned/` — scanned or photographed pages (signed hard copies, notices,
  judgments). These exercise the OCR/vision fallback path.

Keep a roughly even split across both folders — the "OCR-doc answer parity"
target in `docs/ARCHITECTURE.md` is measured by comparing accuracy between
them, so a corpus that's all-native or all-scanned can't validate it.

Do not commit real client/personal documents here — use public contract
templates, sample NDAs, or synthetic documents you generate yourself.
