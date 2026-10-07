"""Train a YOLO stamp/signature detector on the synthetic dataset and report mAP@0.5 per class.

    python scripts/vision/make_synthetic_stamps.py -n 2000
    <python-with-ultralytics> scripts/vision/train_detector.py --data data_lake/synthetic_stamps/data.yaml --epochs 10

Requires `pip install ultralytics`. Weights download on first use. Reports metrics on the VAL split, which is
synthetic: expect optimistic numbers. The honest test is the real scanned-page set (Tobacco-800 or hand-annotated
SC scans); pass it with --real-data <data.yaml> to evaluate the same weights there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data_lake/synthetic_stamps/data.yaml"))
    ap.add_argument("--real-data", type=Path, default=None, help="data.yaml of REAL annotated pages for a sealed evaluation")
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("backend/app/models/stamp_detector"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=str(args.out.resolve().parent), name=args.out.name, exist_ok=True, workers=2, verbose=False)
    best = args.out / "weights" / "best.pt"
    report = {}
    for tag, data in (("synthetic_val", args.data), ("real", args.real_data)):
        if not data:
            continue
        m = YOLO(str(best)).val(data=str(data), device=args.device, imgsz=args.imgsz, verbose=False)
        names = m.names
        report[tag] = {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
                       "per_class_mAP50": {names[int(c)]: round(float(v), 4) for c, v in zip(m.box.ap_class_index, m.box.ap50)}}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
