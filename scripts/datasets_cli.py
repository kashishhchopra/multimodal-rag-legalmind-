"""Dataset registry CLI: list, fetch (resumable), status, and DATASETS.md generation.

    python3 scripts/datasets_cli.py list [--priority P0]
    python3 scripts/datasets_cli.py fetch cuad
    python3 scripts/datasets_cli.py fetch sc_judgments --years 2023 2024 --meta-only
    python3 scripts/datasets_cli.py fetch-all --priority P0 --max-gb 3
    python3 scripts/datasets_cli.py manifest      # rewrites DATASETS.md

Everything lands in data_lake/ (git-ignored). Every download is recorded in
data_lake/manifest.json with source, date, size and sha256 for DATASETS.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
LAKE = ROOT / "data_lake"
REGISTRY = ROOT / "datasets" / "registry.yaml"
MANIFEST = LAKE / "manifest.json"
HF_API = "https://huggingface.co/api/datasets/"


def load_registry() -> list[dict]:
    return yaml.safe_load(REGISTRY.read_text())["datasets"]


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def record(ds_id: str, **entry) -> None:
    LAKE.mkdir(exist_ok=True)
    m = load_manifest()
    m.setdefault(ds_id, []).append({"date": dt.date.today().isoformat(), **entry})
    MANIFEST.write_text(json.dumps(m, indent=2))


def sha256(path: Path, limit_mb: int = 512) -> str:
    """Checksum; for files above limit_mb only the first limit_mb are hashed (flagged in the value)."""
    h, remaining = hashlib.sha256(), limit_mb * 1024 * 1024
    with path.open("rb") as f:
        while remaining > 0 and (chunk := f.read(min(1 << 20, remaining))):
            h.update(chunk)
            remaining -= len(chunk)
    partial = path.stat().st_size > limit_mb * 1024 * 1024
    return h.hexdigest() + (f" (first {limit_mb}MB)" if partial else "")


# ----------------------------------------------------------------- S3 (public, unsigned)
def s3_list(bucket: str, prefix: str) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get(f"https://{bucket}.s3.amazonaws.com/", params=params, timeout=60)
        r.raise_for_status()
        out += [(k, int(s)) for k, s in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", r.text, re.S)]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", r.text)
        if not m:
            return out
        token = m.group(1)


def s3_download(bucket: str, key: str, dest: Path) -> Path:
    """Resumable download using HTTP Range."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    total = int(requests.head(url, timeout=60).headers["Content-Length"])
    have = dest.stat().st_size if dest.exists() else 0
    if have == total:
        return dest
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("ab" if have else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def fetch_s3(ds: dict, years: list[int], meta_only: bool, max_gb: float) -> None:
    a = ds["access"]
    bucket = a["bucket"]
    if not years:
        sys.exit("S3 datasets need --years (e.g. --years 2023 2024). Big: ~200-400MB/year English.")
    budget = max_gb * 1e9
    for y in years:
        meta_dir = LAKE / ds["id"] / f"year={y}" / "metadata"
        if meta_only and a.get("meta_prefix") and "{year}" in a["meta_prefix"]:
            from concurrent.futures import ThreadPoolExecutor

            keys = [k for k, _ in s3_list(bucket, a["meta_prefix"].format(year=y))]
            with ThreadPoolExecutor(16) as ex:  # thousands of tiny files: parallelise
                list(ex.map(lambda k: s3_download(bucket, k, meta_dir / Path(k).name), keys))
            print(f"metadata year={y}: {len(keys)} files")
        if meta_only or "{year}" not in a["prefix"]:
            continue
        key = a["prefix"].format(year=y)
        size = next((s for k, s in s3_list(bucket, key) if k == key), 0)
        if size > budget:
            print(f"skip {key}: {size/1e9:.2f}GB exceeds remaining budget {budget/1e9:.2f}GB")
            continue
        dest = s3_download(bucket, key, LAKE / ds["id"] / f"year={y}" / Path(key).name)
        budget -= size
        record(ds["id"], source=f"s3://{bucket}/{key}", path=str(dest.relative_to(ROOT)), bytes=dest.stat().st_size, sha256=sha256(dest))
        print(f"ok {dest}")


# ----------------------------------------------------------------- Hugging Face
def hf_size(repo: str, revision: str | None = None) -> int:
    r = requests.get(HF_API + repo, params={"blobs": "true", **({"revision": revision} if revision else {})}, timeout=60)
    r.raise_for_status()
    return sum(s.get("size") or 0 for s in r.json().get("siblings", []))


def fetch_hf(ds: dict, max_gb: float, sample_rows: int) -> None:
    from huggingface_hub import snapshot_download

    a = ds["access"]
    repo = a["repo"]
    dest = LAKE / "hf" / repo.replace("/", "__")
    if a.get("streaming_only"):
        # Too large to mirror: save a small sample through the datasets-server API.
        splits = requests.get("https://datasets-server.huggingface.co/splits", params={"dataset": repo}, timeout=60).json().get("splits", [])
        if not splits:
            print(f"{ds['id']}: no preview splits available for {repo}; stream it manually with datasets.load_dataset(streaming=True)")
            return
        s = splits[0]
        rows = requests.get("https://datasets-server.huggingface.co/first-rows", params={"dataset": repo, "config": s["config"], "split": s["split"]}, timeout=120).json()
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / f"sample_{s['config']}_{s['split']}.json"
        out.write_text(json.dumps(rows.get("rows", [])[:sample_rows], ensure_ascii=False))
        record(ds["id"], source=f"hf:{repo} (sample)", path=str(out.relative_to(ROOT)), bytes=out.stat().st_size, sha256=sha256(out))
        print(f"sample ok {out}")
        return
    # Prefer the auto-converted parquet branch: script-only datasets (e.g. CUAD) have no data files on main.
    for rev in ("refs/convert/parquet", None):
        try:
            size = hf_size(repo, rev)
        except requests.HTTPError:
            continue
        if size == 0 and rev is None:
            break
        if a.get("allow_patterns"):
            size = 0  # size guard applies to the whole repo; patterns select a subset
        if size > max_gb * 1e9:
            print(f"{ds['id']}: {size/1e9:.1f}GB exceeds --max-gb {max_gb}; skipped (raise --max-gb to fetch)")
            return
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=dest, allow_patterns=a.get("allow_patterns"))
        except Exception as e:  # noqa: BLE001 - report and try next revision
            print(f"{ds['id']}: revision {rev} failed: {e}")
            continue
        n = sum(1 for p in dest.rglob("*") if p.is_file())
        total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
        record(ds["id"], source=f"hf:{repo}@{rev or 'main'}", path=str(dest.relative_to(ROOT)), files=n, bytes=total)
        print(f"ok {repo} -> {dest} ({n} files, {total/1e6:.1f}MB)")
        return
    print(f"{ds['id']}: could not fetch {repo} (gated, renamed or empty). Check the dataset card and accept terms if gated.")


# ----------------------------------------------------------------- plain URL (zip or file)
def fetch_url(ds: dict) -> None:
    import zipfile

    a = ds["access"]
    dest_dir = LAKE / "manual" / ds["id"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = dest_dir / a.get("filename", "download.bin")
    if not fname.exists():
        with requests.get(a["url"], stream=True, timeout=120, allow_redirects=True, headers={"User-Agent": "LegalMind research"}) as r:
            r.raise_for_status()
            with fname.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    if fname.suffix == ".zip":
        z = zipfile.ZipFile(fname)
        names = [n for n in z.namelist() if n.strip("/")]  # a bare "/" root entry is harmless noise
        if any(n.startswith("/") or ".." in Path(n).parts for n in names):
            sys.exit(f"{fname}: unsafe paths in zip; not extracting")
        z.extractall(dest_dir / "extracted", members=names)
    record(ds["id"], source=a["url"].split("?")[0], path=str(fname.relative_to(ROOT)), bytes=fname.stat().st_size, sha256=sha256(fname))
    print(f"ok {fname}")


# ----------------------------------------------------------------- CLI
def fetch_one(ds: dict, args) -> None:
    kind = ds["access"]["kind"]
    if kind == "s3":
        fetch_s3(ds, args.years or [], args.meta_only, args.max_gb)
    elif kind == "hf":
        fetch_hf(ds, args.max_gb, args.sample_rows)
    elif kind == "url":
        fetch_url(ds)
    else:
        print(f"[manual] {ds['id']} ({ds['priority']}): {ds['name']}\n         -> {ds['access'].get('url')}")
        for fb in ds["access"].get("mteb_fallback", []):
            print(f"         fallback HF mirror: {fb}")


def write_datasets_md() -> None:
    m, rows = load_manifest(), load_registry()
    lines = ["# DATASETS.md", "", "Generated by `scripts/datasets_cli.py manifest`. Licences marked *verify* have not been checked against the dataset card.", "",
             "| ID | Dataset | Pri | Licence | Access | Local status | Last download | Attribution/notes |", "|---|---|---|---|---|---|---|---|"]
    for d in rows:
        e = m.get(d["id"], [])
        a = d["access"]
        where = a.get("repo") or a.get("bucket") or a.get("url", "")
        status = f"{len(e)} download(s), {sum(x.get('bytes', 0) for x in e)/1e6:.1f}MB" if e else ("manual" if a["kind"] == "manual" else "not fetched")
        lines.append(f"| {d['id']} | {d['name']} | {d['priority']} | {d['license']} | {a['kind']}: {where} | {status} | {e[-1]['date'] if e else '-'} | {d.get('attribution', d['use'])} |")
    (ROOT / "DATASETS.md").write_text("\n".join(lines) + "\n")
    print("wrote DATASETS.md")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("list", "fetch", "fetch-all", "manifest"):
        sp = sub.add_parser(name)
        if name in ("list", "fetch-all"):
            sp.add_argument("--priority", choices=["P0", "P1", "P2"])
        if name == "fetch":
            sp.add_argument("id")
        if name in ("fetch", "fetch-all"):
            sp.add_argument("--years", type=int, nargs="*")
            sp.add_argument("--meta-only", action="store_true")
            sp.add_argument("--max-gb", type=float, default=2.0)
            sp.add_argument("--sample-rows", type=int, default=200)
    args = p.parse_args()
    reg = load_registry()
    if args.cmd == "list":
        for d in reg:
            if not args.priority or d["priority"] == args.priority:
                print(f"{d['priority']} {d['id']:<18} {d['access']['kind']:<7} {d['name']}")
    elif args.cmd == "fetch":
        ds = next((d for d in reg if d["id"] == args.id), None) or sys.exit(f"unknown id {args.id}")
        fetch_one(ds, args)
        write_datasets_md()
    elif args.cmd == "fetch-all":
        for d in reg:
            if (not args.priority or d["priority"] == args.priority) and not (d["access"]["kind"] == "s3" and not args.years):
                try:
                    fetch_one(d, args)
                except Exception as e:  # noqa: BLE001 - keep going, report at the end
                    print(f"FAILED {d['id']}: {e}")
        write_datasets_md()
    else:
        write_datasets_md()


if __name__ == "__main__":
    main()
