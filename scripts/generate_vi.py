"""Vietnamese zero-shot / fine-tuned voice generation with Confucius4-TTS.

Differences from the stock ``example.py``:

* **Vietnamese text normalization.** The repo's ``TextNormalizer`` has no Vietnamese
  branch and would send ``vi`` text through the English ``wetext`` normalizer ("2" ->
  "two"). This script normalizes numbers, percentages, decimals and a few abbreviations
  into Vietnamese words itself (or via ``vinorm`` when installed) and calls
  ``generate(raw=True)`` so the model receives the text unchanged.
* **Local checkpoints.** ``ConfuciusTTS`` resolves every checkpoint through
  ``hf_hub_download`` and treats ``t2s_checkpoint`` as a file name inside the HF repo.
  This script patches that resolver so files already present under ``checkpoints/``,
  ``pretrained/`` or given as absolute paths are used directly. That is what makes a
  fine-tuned T2S/S2A checkpoint loadable with ``--t2s-checkpoint`` / ``--s2a-checkpoint``.
* CPU-friendly defaults and timing output.

Examples::

    # zero-shot, pretrained weights, reference clip from the pipeline
    python scripts/generate_vi.py \\
        --prompt-wav ../speech_dataset/audio-pipeline/work_5min/s7_loudnorm/audio/5e7117f231ca_00001400_00011700.wav \\
        --text "Hôm nay trời đẹp, chúng ta đi dạo một vòng quanh hồ nhé." \\
        --out data/test_vi.wav --device cpu

    # fine-tuned T2S
    python scripts/generate_vi.py --prompt-wav ref.wav --text-file input.txt \\
        --t2s-checkpoint logs/t2s_vi_podcast/ckpt/exported/t2s_model.safetensors --out out.wav
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # inference_config.yaml uses paths relative to the repo root

# --------------------------------------------------------------------------- Vietnamese text normalization
_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

ABBREVIATIONS = {
    r"\bTP\.?\s*HCM\b": "thành phố Hồ Chí Minh",
    r"\bTP\.": "thành phố",
    r"\bHN\b": "Hà Nội",
    r"\bUBND\b": "ủy ban nhân dân",
    r"\bVN\b": "Việt Nam",
    r"\bTS\.": "tiến sĩ",
    r"\bThS\.": "thạc sĩ",
    r"\bGS\.": "giáo sư",
    r"\bPGS\.": "phó giáo sư",
    r"\bBS\.": "bác sĩ",
    r"\bv\.v\.": "vân vân",
    r"\bvs\b": "với",
    r"%": " phần trăm",
    r"\bkm\b": "ki lô mét",
    r"\bkg\b": "ki lô gam",
    r"\bcm\b": "xen ti mét",
    r"\bmm\b": "mi li mét",
    r"\bUSD\b": "đô la Mỹ",
    r"\bVNĐ\b|\bVND\b|\bđ\b": "đồng",
}


def _read_three(n: int, full: bool) -> str:
    """Read a 0..999 group. ``full`` forces reading leading zeros ("không trăm ...")."""
    hundreds, rest = divmod(n, 100)
    tens, units = divmod(rest, 10)
    words = []
    if hundreds or full:
        words += [_DIGITS[hundreds], "trăm"]
    if tens == 0:
        if units:
            if hundreds or full:
                words.append("lẻ")
            words.append(_DIGITS[units])
    elif tens == 1:
        words.append("mười")
        if units == 5:
            words.append("lăm")
        elif units:
            words.append(_DIGITS[units])
    else:
        words += [_DIGITS[tens], "mươi"]
        if units == 1:
            words.append("mốt")
        elif units == 4:
            words.append("tư")
        elif units == 5:
            words.append("lăm")
        elif units:
            words.append(_DIGITS[units])
    return " ".join(words)


def number_to_vietnamese(n: int) -> str:
    if n == 0:
        return "không"
    if n < 0:
        return "âm " + number_to_vietnamese(-n)
    groups = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    scales = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]
    parts = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        full = i != len(groups) - 1  # leading zeros are read inside lower groups
        text = _read_three(g, full)
        if scales[i]:
            text += " " + scales[i]
        parts.append(text)
    return " ".join(parts)


def _replace_number(match: re.Match) -> str:
    raw = match.group(0)
    s = raw.replace(".", "").replace(" ", "") if re.fullmatch(r"\d{1,3}(\.\d{3})+", raw) else raw
    if "," in s:  # Vietnamese decimal comma
        int_part, frac = s.split(",", 1)
        return number_to_vietnamese(int(int_part)) + " phẩy " + " ".join(_DIGITS[int(c)] for c in frac if c.isdigit())
    return number_to_vietnamese(int(s))


def normalize_vietnamese(text: str) -> str:
    """Lightweight Vietnamese text normalization for TTS input."""
    try:  # prefer the full normalizer when available (pip install vinorm)
        from vinorm import TTSnorm  # type: ignore

        text = TTSnorm(text, punc=False, unknown=False, lower=False, rule=False)
    except Exception:
        pass

    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    for pat, rep in ABBREVIATIONS.items():
        text = re.sub(pat, rep, text)
    # dates dd/mm/yyyy -> "ngày d tháng m năm y"
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", lambda m: f"ngày {m.group(1)} tháng {m.group(2)} năm {m.group(3)}", text)
    # numbers: thousands with dots (1.234.567), decimals with comma (3,5), plain integers
    text = re.sub(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+,\d+|\d+", _replace_number, text)
    text = text.replace('"', "").replace("“", "").replace("”", "")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- local checkpoint resolver
def _patch_local_checkpoints(explicit: dict[str, str]) -> None:
    """Make ConfuciusTTS use local files instead of downloading.

    ``explicit`` maps HF file names (as they appear in inference_config.yaml) to local paths.
    Files found under checkpoints/ or pretrained/campplus/ are also picked up automatically.
    """
    import huggingface_hub
    import confuciustts.cli.inference as inf

    real = huggingface_hub.hf_hub_download

    def resolver(repo_id: str, filename: str, **kw):
        cands = []
        if filename in explicit:
            cands.append(Path(explicit[filename]))
        p = Path(filename)
        if p.is_absolute():
            cands.append(p)
        cands += [REPO_ROOT / "checkpoints" / p.name, REPO_ROOT / "pretrained" / "campplus" / p.name]
        for c in cands:
            if c.is_file():
                return str(c)
        return real(repo_id, filename=filename, **kw)

    inf.hf_hub_download = resolver


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"  # MPS is skipped on purpose (unreliable on Intel Macs)


# --------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Vietnamese generation with Confucius4-TTS", formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="text to synthesize")
    src.add_argument("--text-file", help="UTF-8 text file to synthesize")
    ap.add_argument("--prompt-wav", required=True, help="reference voice clip (3-15 s, single speaker)")
    ap.add_argument("--out", default="data/output_vi.wav")
    ap.add_argument("--config", default="config/inference_config.yaml")
    ap.add_argument("--w2v-bert-path", default="pretrained/w2v-bert-2.0" if (REPO_ROOT / "pretrained/w2v-bert-2.0").is_dir() else None,
                    help="local w2v-BERT 2.0 dir; default uses pretrained/w2v-bert-2.0 if present, else the HF id from config")
    ap.add_argument("--t2s-checkpoint", default=None, help="local T2S safetensors (e.g. a fine-tuned export); default checkpoints/t2s_model.safetensors")
    ap.add_argument("--s2a-checkpoint", default=None, help="local S2A .pt; default checkpoints/s2a_model.pt or HF download")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    ap.add_argument("--no-normalize", action="store_true", help="send text as-is (already normalized)")
    ap.add_argument("--print-normalized", action="store_true", help="only print the normalized text and exit")

    g = ap.add_argument_group("sampling")
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-p", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=30)
    g.add_argument("--num-beams", type=int, default=3)
    g.add_argument("--repetition-penalty", type=float, default=10.0)
    g.add_argument("--n-timesteps", type=int, default=25, help="S2A ODE steps")
    g.add_argument("--cfg-rate", type=float, default=0.7)
    g.add_argument("--max-text-tokens-per-segment", type=int, default=80)
    g.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    text = a.text if a.text is not None else Path(a.text_file).read_text(encoding="utf-8")
    norm = text.strip() if a.no_normalize else normalize_vietnamese(text)
    print(f"[generate_vi] text      : {text.strip()[:200]}")
    print(f"[generate_vi] normalized: {norm[:200]}")
    if a.print_normalized:
        return

    import torch
    import torchaudio
    import yaml

    if a.seed is not None:
        torch.manual_seed(a.seed)

    # config: optionally point w2v-BERT to the local copy
    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if a.w2v_bert_path:
        cfg["paths"]["w2v_bert_path"] = a.w2v_bert_path
    explicit = {}
    if a.t2s_checkpoint:
        explicit[cfg["paths"]["t2s_checkpoint"]] = str(Path(a.t2s_checkpoint).resolve())
    if a.s2a_checkpoint:
        explicit[cfg["paths"]["s2a_checkpoint"]] = str(Path(a.s2a_checkpoint).resolve())
    _patch_local_checkpoints(explicit)

    tmp_cfg = REPO_ROOT / "config" / ".inference_config_runtime.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    from confuciustts.cli.inference import ConfuciusTTS

    device = resolve_device(a.device)
    t0 = time.time()
    model = ConfuciusTTS(config_path=str(tmp_cfg), device=device)
    print(f"[generate_vi] models loaded on {device} in {time.time() - t0:.1f}s, sample_rate={model.sample_rate}")

    t1 = time.time()
    audio = model.generate(
        norm, "vi", a.prompt_wav, raw=True,
        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, num_beams=a.num_beams,
        repetition_penalty=a.repetition_penalty, n_timesteps=a.n_timesteps,
        inference_cfg_rate=a.cfg_rate, max_text_tokens_per_segment=a.max_text_tokens_per_segment,
        verbose=True,
    )
    dur = audio.shape[-1] / model.sample_rate
    elapsed = time.time() - t1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), audio.cpu(), model.sample_rate)
    print(f"[generate_vi] saved {out} : {dur:.2f}s audio, generated in {elapsed:.1f}s (RTF {elapsed / max(dur, 1e-6):.2f})")


if __name__ == "__main__":
    main()
