"""Workarounds for torch/macOS quirks.

de_mmap: on this Apple-silicon setup (torch 2.5.1 CPU), Linear layers whose weights are memory-mapped straight from a
safetensors file returned NaN for some models (cross-encoder/ms-marco-MiniLM-L-6-v2), while a cloned copy of the same
weights was fine. Cloning every parameter after load is cheap and removes the problem; always run it on freshly loaded
models and fail loudly if outputs are still non-finite.
"""
from __future__ import annotations

import numpy as np
import torch


def de_mmap(model) -> None:
    mod = getattr(model, "model", model)  # CrossEncoder wraps the HF model in .model
    with torch.no_grad():
        for p in mod.parameters():
            p.data = p.data.clone()
        for b in mod.buffers():
            b.data = b.data.clone()


def assert_finite(scores, what: str = "scores") -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"non-finite {what}: the model produced NaN/inf (see app/models/torch_fixes.py)")
    return arr
