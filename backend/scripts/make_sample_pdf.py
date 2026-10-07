"""CLI wrapper to generate the demo PDF (electronics_manual.pdf).

Run from the backend/ directory:
    python scripts/make_sample_pdf.py [output_path]

The actual builder lives in app/sample_pdf.py so tests can reuse it.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the app package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sample_pdf import build_sample_pdf  # noqa: E402


def main() -> None:
    if len(sys.argv) > 1:
        dest = Path(sys.argv[1])
    else:
        dest = Path(__file__).resolve().parent.parent / "data" / "electronics_manual.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = build_sample_pdf(dest)
    print(f"Wrote sample PDF -> {path}")


if __name__ == "__main__":
    main()
