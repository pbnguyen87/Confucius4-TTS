"""Export a Lightning T2S checkpoint (.ckpt) to the safetensors file inference expects.

Keeps only ``t2s_model.*`` tensors (drops the w2v-BERT extractor, optimizer state, EMA, etc.) and strips
the prefix so keys match ``checkpoints/t2s_model.safetensors``. The repo has no such tool of its own.

    python vistral_finetune/export_t2s.py logs/t2s_vi_vistral_stageA/ckpt/last.ckpt \\
        checkpoints/t2s_model_vistral_stageA.safetensors
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--dtype", choices=["keep", "fp32", "bf16", "fp16"], default="keep")
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    out = {}
    for k, v in sd.items():
        if not k.startswith("t2s_model."):
            continue
        t = v.detach().contiguous()
        if a.dtype != "keep":
            t = t.to({"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype])
        out[k[len("t2s_model."):]] = t
    if not out:
        raise SystemExit("no t2s_model.* tensors found in checkpoint")
    emb = out.get("text_projector.embed.weight")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, a.out, metadata={"format": "pt", "source": os.path.basename(a.ckpt)})
    print(f"wrote {a.out}: {len(out)} tensors, {sum(t.numel() for t in out.values()) / 1e6:.1f} M params, "
          f"{os.path.getsize(a.out) / 1e9:.2f} GB, text table {tuple(emb.shape) if emb is not None else 'n/a'}")


if __name__ == "__main__":
    main()
