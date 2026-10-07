"""Build the Mistral + Vistral hybrid text embedding and write a T2S checkpoint that uses it.

What it does (see PLAN.md §3):
  1. downloads only Vistral's tokenizer files and its ``model.embed_tokens.weight`` tensor
     (~314 MB, fetched with an HTTP range request so the 14 GB of transformer weights are never pulled);
  2. verifies that Vistral's first 32,000 pieces are Mistral's pieces, id for id;
  3. keeps Confucius4's existing 32,000 Mistral rows byte-identical and appends the 6,369 new Vietnamese
     rows, initialised as  0.5 * mean(Mistral rows of the piece's Mistral sub-pieces)
                         + 0.5 * scale * Vistral row            (scale brings new-row norm to the old mean),
     then each new row is put on the old-row mean input norm and one global factor is bisected so the
     projector's mean OUTPUT norm for new rows equals that for old rows (see PLAN.md §3);
  4. writes ``checkpoints/t2s_model_vistral_hybrid.safetensors`` = original T2S checkpoint with the
     enlarged table, plus ``vistral_finetune/report.json`` with every number used.

Usage::

    python vistral_finetune/build_hybrid_embedding.py                 # full build
    python vistral_finetune/build_hybrid_embedding.py --dry-run       # everything except the 2.6 GB write
    python vistral_finetune/build_hybrid_embedding.py --init vistral  # new rows = rescaled Vistral only
    python vistral_finetune/build_hybrid_embedding.py --init subpiece # new rows = Mistral sub-piece mean only
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # a stale ~/.cache/huggingface/token breaks public downloads
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

VISTRAL_REPO = "jan-hq/Vistral-7B-Chat-DPO"      # ungated mirror of Viet-Mistral/Vistral-7B-Chat (same tokenizer + table)
MISTRAL_ROWS = 32000
TOKENIZER_FILES = ["tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vistral-repo", default=VISTRAL_REPO)
    ap.add_argument("--t2s-checkpoint", default=str(REPO_ROOT / "checkpoints" / "t2s_model.safetensors"))
    ap.add_argument("--mistral-tokenizer", default=str(REPO_ROOT / "checkpoints"))
    ap.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / "t2s_model_vistral_hybrid.safetensors"))
    ap.add_argument("--tokenizer-out", default=str(HERE / "tokenizer"))
    ap.add_argument("--assets", default=str(HERE / "assets"))
    ap.add_argument("--init", choices=["blend", "vistral", "subpiece"], default="blend")
    ap.add_argument("--blend", type=float, default=0.5, help="weight of the sub-piece mean in 'blend' init")
    ap.add_argument("--no-renorm", action="store_true", help="do not rescale new rows to the old-row mean norm")
    ap.add_argument("--dry-run", action="store_true", help="do everything except writing the new checkpoint")
    return ap.parse_args()


# --------------------------------------------------------------------------- download helpers
def fetch_tokenizer(repo: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    dest.mkdir(parents=True, exist_ok=True)
    if Path(repo).is_dir():  # local export, e.g. mistral_lora_finetune/checkpoints/mistral-7b-vi-final
        import shutil
        for f in TOKENIZER_FILES + ["tokenizer.json"]:
            if (Path(repo) / f).is_file():
                shutil.copy(Path(repo) / f, dest / f)
        return dest
    have = set(list_repo_files(repo, token=False))
    got = []
    for f in TOKENIZER_FILES:
        if f in have:
            hf_hub_download(repo, f, local_dir=str(dest), token=False)
            got.append(f)
    if "tokenizer.model" not in got and "tokenizer.json" not in got:
        raise RuntimeError(f"{repo} has no tokenizer files")
    return dest


def fetch_embed_tokens(repo: str, cache: Path) -> "torch.Tensor":
    """Read only ``model.embed_tokens.weight`` from the sharded safetensors via HTTP range requests."""
    import requests
    import torch

    cache.mkdir(parents=True, exist_ok=True)
    if Path(repo).is_dir():  # local checkpoint: read the embedding straight from its safetensors
        from safetensors import safe_open
        d = Path(repo); idx = d / "model.safetensors.index.json"
        key = "model.embed_tokens.weight"
        shard = d / json.loads(idx.read_text())["weight_map"][key] if idx.is_file() else d / "model.safetensors"
        with safe_open(str(shard), "pt") as f:
            return f.get_tensor(key).float().clone()
    cached = cache / "vistral_embed_tokens.pt"
    if cached.is_file():
        return torch.load(cached, map_location="cpu").float()
    base = f"https://huggingface.co/{repo}/resolve/main/"
    idx = requests.get(base + "model.safetensors.index.json", timeout=60).json()
    shard = idx["weight_map"]["model.embed_tokens.weight"]
    url = base + shard
    hlen = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60).content)[0]
    header = json.loads(requests.get(url, headers={"Range": f"bytes=8-{8 + hlen - 1}"}, timeout=60).content)
    meta = header["model.embed_tokens.weight"]
    s, e = meta["data_offsets"]
    print(f"[build] fetching {meta['dtype']} {meta['shape']} ({(e - s) / 1e6:.0f} MB) from {shard}")
    t0 = time.time()
    raw = requests.get(url, headers={"Range": f"bytes={8 + hlen + s}-{8 + hlen + e - 1}"}, timeout=1800).content
    if len(raw) != e - s:
        raise RuntimeError(f"short read: {len(raw)} of {e - s} bytes")
    dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[meta["dtype"]]
    emb = torch.frombuffer(bytearray(raw), dtype=dt).view(*meta["shape"]).float().clone()
    torch.save(emb.to(torch.bfloat16), cached)  # bf16 on disk, 314 MB
    print(f"[build] done in {time.time() - t0:.0f}s, cached at {cached}")
    return emb


# --------------------------------------------------------------------------- core
def verify_shared_vocab(mis_tok, vis_tok) -> dict:
    mv, vv = mis_tok.get_vocab(), vis_tok.get_vocab()
    inv_v = {i: t for t, i in vv.items()}
    mismatched = [(t, i, inv_v.get(i)) for t, i in mv.items() if inv_v.get(i) != t]
    return {"mistral_vocab": len(mv), "vistral_vocab": len(vv), "shared_rows": MISTRAL_ROWS,
            "mismatched_shared_ids": len(mismatched), "examples": mismatched[:5]}


def build_table(mis_table, vis_table, mis_tok, vis_tok, init: str, blend: float, renorm: bool = True, proj_weights: dict | None = None):
    import torch

    n_new = vis_table.shape[0] - MISTRAL_ROWS
    old_norm = mis_table.norm(dim=1).mean()
    new_norm = vis_table[MISTRAL_ROWS:].norm(dim=1).mean()
    scale = float(old_norm / new_norm)
    new = torch.empty(n_new, mis_table.shape[1], dtype=mis_table.dtype)
    n_subpieces = []
    for j in range(n_new):
        tid = MISTRAL_ROWS + j
        piece = vis_tok.convert_ids_to_tokens(tid)
        # word-initial pieces: encode the bare word (tokenizer adds the "▁"); word-internal pieces: drop the
        # spurious leading "▁" the tokenizer inserts, so the word-boundary row is not averaged into every row
        initial = piece.startswith("▁")
        sub = mis_tok.encode(piece[1:] if initial else piece, add_special_tokens=False)
        if not initial and sub:
            toks = mis_tok.convert_ids_to_tokens(sub)
            if toks[0] == "▁":
                sub = sub[1:]
            elif toks[0].startswith("▁") and toks[0][1:] in mis_tok.get_vocab():
                sub[0] = mis_tok.get_vocab()[toks[0][1:]]
        sub = [s for s in sub if s < MISTRAL_ROWS] or [mis_tok.unk_token_id or 0]
        n_subpieces.append(len(sub))
        sub_mean = mis_table[sub].mean(dim=0)
        vis_row = scale * vis_table[tid].to(mis_table.dtype)
        if init == "subpiece":
            new[j] = sub_mean
        elif init == "vistral":
            new[j] = vis_row
        else:
            new[j] = blend * sub_mean + (1.0 - blend) * vis_row
    # Blending two non-parallel vectors (and averaging sub-pieces) shrinks the norm; put every new row on the
    # old-row mean norm so fc1+SiLU see the magnitude they were trained on (checked by check_projection.py).
    if renorm:
        new = new * (old_norm / new.norm(dim=1, keepdim=True).clamp_min(1e-8))
    # Input norm alone is not the right target: fc1+SiLU+fc2 is nonlinear and the new rows' *direction* excites it
    # more than the old rows do (+22 % output norm at equal input norm). Calibrate one global scalar so the mean
    # projector OUTPUT norm of the new rows matches that of the old rows -- that is what the transformer sees.
    calib = {}
    if proj_weights is not None:
        def project(x):
            h = torch.nn.functional.silu(x @ proj_weights["fc1_w"].T + proj_weights["fc1_b"])
            return h @ proj_weights["fc2_w"].T + proj_weights["fc2_b"]
        with torch.no_grad():
            target = project(mis_table).norm(dim=1).mean()
            lo, hi = 0.3, 2.0
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if project(new * mid).norm(dim=1).mean() > target: hi = mid
                else: lo = mid
            factor = 0.5 * (lo + hi)
            new = new * factor
            calib = {"output_norm_calibration_factor": round(factor, 4),
                     "projector_out_norm_old_rows": round(float(target), 4),
                     "projector_out_norm_new_rows": round(float(project(new).norm(dim=1).mean()), 4)}
    hybrid = torch.cat([mis_table, new], dim=0)
    stats = {**calib, "n_new_rows": n_new, "scale_applied_to_vistral_rows": round(scale, 4),
             "old_row_mean_norm": round(float(old_norm), 4), "vistral_new_row_mean_norm": round(float(new_norm), 4),
             "hybrid_new_row_mean_norm": round(float(new.norm(dim=1).mean()), 4),
             "mistral_subpieces_per_new_piece_mean": round(sum(n_subpieces) / n_new, 2), "init": init, "blend": blend, "renorm_to_old_mean": renorm}
    return hybrid, stats


def tokens_per_syllable(tok, text: str) -> float:
    import re

    syl = [w for w in re.sub(r"[.,!?;:]", " ", text).split() if w]
    return len(tok.encode(text, add_special_tokens=False)) / max(len(syl), 1)


def main() -> None:
    a = parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    report: dict = {"vistral_repo": a.vistral_repo, "t2s_checkpoint": a.t2s_checkpoint}

    print("[build] 1/4 tokenizer")
    vis_dir = fetch_tokenizer(a.vistral_repo, Path(a.tokenizer_out))
    mis_tok = AutoTokenizer.from_pretrained(a.mistral_tokenizer, token=False)
    vis_tok = AutoTokenizer.from_pretrained(str(vis_dir), token=False)
    report["vocab_check"] = verify_shared_vocab(mis_tok, vis_tok)
    if report["vocab_check"]["mismatched_shared_ids"]:
        sys.exit(f"[build] ABORT: {report['vocab_check']['mismatched_shared_ids']} shared ids differ between tokenizers: "
                 f"{report['vocab_check']['examples']}")
    print(f"        Mistral {report['vocab_check']['mistral_vocab']} pieces, Vistral {report['vocab_check']['vistral_vocab']}, "
          f"first {MISTRAL_ROWS} identical")

    print("[build] 2/4 embedding tensor")
    vis_table = fetch_embed_tokens(a.vistral_repo, Path(a.assets))
    if vis_table.shape[0] != len(vis_tok) or vis_table.shape[0] <= MISTRAL_ROWS:
        sys.exit(f"[build] ABORT: embedding rows {vis_table.shape[0]} vs tokenizer size {len(vis_tok)}")

    print("[build] 3/4 hybrid table")
    tensors = {}
    with safe_open(a.t2s_checkpoint, "pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    mis_table = tensors["text_projector.embed.weight"]
    if tuple(mis_table.shape) != (MISTRAL_ROWS, vis_table.shape[1]):
        sys.exit(f"[build] ABORT: unexpected Mistral table shape {tuple(mis_table.shape)}")
    cos = torch.nn.functional.cosine_similarity(mis_table.float(), vis_table[:MISTRAL_ROWS], dim=1)
    report["shared_row_drift_cosine"] = {"mean": round(float(cos.mean()), 4), "p10": round(float(cos.quantile(0.1)), 4),
                                         "median": round(float(cos.median()), 4)}
    pw = {"fc1_w": tensors["text_projector.text_projection_fc1.weight"].float(), "fc1_b": tensors["text_projector.text_projection_fc1.bias"].float(),
          "fc2_w": tensors["text_projector.text_projection_fc2.weight"].float(), "fc2_b": tensors["text_projector.text_projection_fc2.bias"].float()}
    hybrid, stats = build_table(mis_table.float(), vis_table, mis_tok, vis_tok, a.init, a.blend, renorm=not a.no_renorm, proj_weights=pw)
    report["table"] = stats
    assert torch.equal(hybrid[:MISTRAL_ROWS], mis_table.float()), "old rows must be untouched"
    tensors["text_projector.embed.weight"] = hybrid.to(mis_table.dtype).contiguous()

    sample = ("Hôm nay trời đẹp, chúng ta đi dạo một vòng quanh hồ nhé. Mô hình mới có thể tạo giọng nói tiếng Việt "
              "tự nhiên hơn, kể cả khi câu có chèn từ tiếng Anh như deep learning, server, hay deadline.")
    report["tokens_per_syllable"] = {"mistral": round(tokens_per_syllable(mis_tok, sample), 3),
                                     "vistral": round(tokens_per_syllable(vis_tok, sample), 3)}
    print(f"        projector output norm old {stats.get('projector_out_norm_old_rows')} new {stats.get('projector_out_norm_new_rows')} "
          f"(calibration factor {stats.get('output_norm_calibration_factor')})")
    print(f"        new rows {stats['n_new_rows']}, scale {stats['scale_applied_to_vistral_rows']}, "
          f"tokens/syllable Mistral {report['tokens_per_syllable']['mistral']} -> Vistral {report['tokens_per_syllable']['vistral']}")

    print("[build] 4/4 write")
    report["out"] = a.out
    report["config_changes"] = {"t2s_model.vocab_size": int(hybrid.shape[0]), "paths.tokenizer_path": str(vis_dir),
                                "paths.t2s_checkpoint": a.out}
    Path(HERE / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.dry_run:
        print(f"        dry run: checkpoint not written. Report -> {HERE / 'report.json'}")
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, a.out, metadata={"format": "pt", "note": f"Confucius4 T2S with Mistral+Vistral hybrid text table ({a.init})"})
    print(f"        wrote {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB). Report -> {HERE / 'report.json'}")


if __name__ == "__main__":
    main()
