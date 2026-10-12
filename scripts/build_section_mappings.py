"""Build full old->new section mapping CSVs (IPC->BNS, CrPC->BNSS, Evidence Act->BSA).

    backend/.venv/bin/python scripts/build_section_mappings.py

Source: the "Correspondence Table and Comparison Summary" PDFs (BNS/BNSS/BSA each vs the old code) hosted by the
Kerala Prisons Department, authored by Anil Kishore Yadav IPS (CAPT Bhopal). They are a secondary compilation, NOT the
MHA's own table; verify any section you rely on against the Gazette text. Output CSVs go in data_lake/bns_mapping/ and are
picked up by backend/app/legal/section_mapper.py automatically (they add alternates after the curated entries).
"""
import csv
import re
import sys
from collections import OrderedDict
from pathlib import Path

import fitz
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_lake" / "bns_mapping"
BASE = "https://www.keralaprisons.gov.in/userfiles/act-and-rules/"
# file, old code label in CSV, new code label, (title column, old-section column). The BSA table lists the old
# Evidence Act section BEFORE the subject, unlike the BNS/BNSS tables.
SOURCES = [
    ("comparison_summary_BNS_to_IPC.pdf", "IPC", "BNS", (1, 2)),
    ("comparison_summary_BNSS_to_CrPC.pdf", "CRPC", "BNSS", (1, 2)),
    ("comparison_summary_BSA_to_IEA.pdf", "IEA", "BSA", (2, 1)),
]


def parse(pdf: Path, cols=(1, 2)):
    pairs = OrderedDict()
    for page in fitz.open(pdf):
        for t in page.find_tables().tables:
            for r in t.extract():
                r = [(c or "").replace("\n", " ").strip() for c in r]
                if len(r) >= 3 and re.match(r"^\d{1,3}[A-Z]?(\s*\(\w+\))*$", r[0]):
                    new = re.sub(r"\s+", "", r[0])
                    for tok in re.findall(r"\b\d{1,3}[A-Z]{0,2}\b", r[cols[1]]):
                        pairs.setdefault((tok, new), r[cols[0]])
    return pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, old, new, cols in SOURCES:
        pdf = OUT / fname
        if not pdf.exists():
            r = requests.get(BASE + fname, timeout=90, headers={"User-Agent": "LegalMind research"})
            r.raise_for_status()
            pdf.write_bytes(r.content)
        pairs = parse(pdf, cols)
        dest = OUT / f"{old.lower()}_{new.lower()}_from_correspondence_table.csv"
        with dest.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["old_code", "old_section", "new_code", "new_section", "title"])
            for (o, n), title in pairs.items():
                w.writerow([old, o, new, n, title[:200]])
        print(f"{old}->{new}: {len(pairs)} pairs, {len({o for o, _ in pairs})} old sections -> {dest.name}")


if __name__ == "__main__":
    sys.exit(main())
