"""Generate the same Vietnamese text in both cloning modes of Confucius4-TTS from one reference clip.

For every input sentence two files are written with the same seed and sampling settings:

    <out-dir>/<name>.reference.wav      stock reference mode (speaker vector + mel prompt)
    <out-dir>/<name>.continuation.wav   continuation mode (prompt transcript + prompt semantic tokens)

plus ``<out-dir>/pairs.jsonl`` with one row per sentence (texts, files, durations, generation
time), so the pairs can be scored or listened to side by side. Models are loaded once.

Examples::

    # one sentence
    python scripts/generate_vi_both_modes.py \\
        --prompt-wav audio_samples/5e7117f231ca_00123600_00136600.wav \\
        --prompt-text-file audio_samples/5e7117f231ca_00123600_00136600.txt \\
        --text "Hôm nay trời đẹp, chúng ta đi dạo một vòng quanh hồ nhé." --out-dir data/both_modes

    # several sentences, one per line, fine-tuned T2S, auto transcript of the prompt
    python scripts/generate_vi_both_modes.py --prompt-wav ref.wav --auto-transcribe \\
        --text-file sentences.txt --t2s-checkpoint logs/t2s_vi/exported/t2s_model.safetensors \\
        --out-dir data/both_modes_ft --seed 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_vi as gv  # noqa: E402  (chdir to repo root, normalizer, checkpoint patching)
import generate_vi_continuation as gc  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Reference vs continuation mode, same text, same seed",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="one sentence")
    src.add_argument("--text-file", help="UTF-8 file, one sentence per line (blank lines and # comments skipped)")
    ap.add_argument("--prompt-wav", required=True, help="reference clip, 3-15 s")
    pt = ap.add_mutually_exclusive_group(required=True)
    pt.add_argument("--prompt-text", help="exact transcript of the reference clip")
    pt.add_argument("--prompt-text-file")
    pt.add_argument("--auto-transcribe", action="store_true", help="transcribe the clip with --asr-model")
    ap.add_argument("--asr-model", default="vinai/PhoWhisper-large")
    ap.add_argument("--out-dir", default="data/both_modes")
    ap.add_argument("--name", default=None, help="base file name; default s001, s002, ... (or 'sample' for a single --text)")
    ap.add_argument("--modes", default="reference,continuation", help="subset to run, e.g. 'continuation'")
    ap.add_argument("--config", default="config/inference_config.yaml")
    ap.add_argument("--w2v-bert-path", default="pretrained/w2v-bert-2.0" if (REPO_ROOT / "pretrained/w2v-bert-2.0").is_dir() else None)
    ap.add_argument("--t2s-checkpoint", default=None)
    ap.add_argument("--s2a-checkpoint", default=None)
    ap.add_argument("--codec-path", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--join", default=" ", help="continuation: string between prompt text and target text")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and normalized texts, load nothing")
    g = ap.add_argument_group("sampling (shared by both modes)")
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-p", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=30)
    g.add_argument("--num-beams", type=int, default=3)
    g.add_argument("--repetition-penalty", type=float, default=10.0)
    g.add_argument("--penalize-prompt", action="store_true", help="continuation: HF penalty over prompt + generated tokens")
    g.add_argument("--max-new-tokens", type=int, default=600)
    g.add_argument("--n-timesteps", type=int, default=25)
    g.add_argument("--cfg-rate", type=float, default=0.7)
    g.add_argument("--max-text-tokens-per-segment", type=int, default=80)
    g.add_argument("--cross-fade-duration", type=float, default=0.3)
    g.add_argument("--seed", type=int, default=1000)
    return ap.parse_args()


def read_sentences(a: argparse.Namespace) -> list[str]:
    if a.text is not None:
        return [a.text.strip()]
    lines = Path(a.text_file).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.lstrip().startswith("#")]


def main() -> None:
    a = parse_args()
    for k in ("prompt_wav", "out_dir", "text_file", "prompt_text_file", "t2s_checkpoint", "s2a_checkpoint", "codec_path"):
        setattr(a, k, gv._user_path(getattr(a, k)))
    if not Path(a.prompt_wav).is_file():
        sys.exit(f"[both] reference audio not found: {a.prompt_wav}")
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    bad = set(modes) - {"reference", "continuation"}
    if bad:
        sys.exit(f"[both] unknown mode(s): {sorted(bad)}")

    sentences = read_sentences(a)
    if not sentences:
        sys.exit("[both] no sentences to synthesize")
    norm_sentences = [s if a.no_normalize else gv.normalize_vietnamese(s) for s in sentences]
    names = [a.name or ("sample" if len(sentences) == 1 else f"s{i + 1:03d}") for i in range(len(sentences))]
    if a.name and len(sentences) > 1:
        names = [f"{a.name}_{i + 1:03d}" for i in range(len(sentences))]

    prompt_text = None
    if not a.auto_transcribe:
        prompt_text = a.prompt_text if a.prompt_text is not None else Path(a.prompt_text_file).read_text(encoding="utf-8")
        prompt_text = prompt_text.strip() if a.no_normalize else gv.normalize_vietnamese(prompt_text)

    print(f"[both] prompt     : {a.prompt_wav}")
    print(f"[both] prompt text: {prompt_text if prompt_text else '(ASR: ' + a.asr_model + ')'}")
    print(f"[both] modes      : {modes}   seed {a.seed}   {len(sentences)} sentence(s) -> {a.out_dir}")
    for n, s in zip(names, norm_sentences):
        print(f"  {n}: {s[:160]}")
    if a.dry_run:
        return

    import torch
    import torchaudio
    import yaml

    device = gv.resolve_device(a.device)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.w2v_bert_path:
        cfg["paths"]["w2v_bert_path"] = a.w2v_bert_path
    explicit = {}
    if a.t2s_checkpoint:
        explicit[cfg["paths"]["t2s_checkpoint"]] = str(Path(a.t2s_checkpoint).resolve())
    if a.s2a_checkpoint:
        explicit[cfg["paths"]["s2a_checkpoint"]] = str(Path(a.s2a_checkpoint).resolve())
    gv._patch_local_checkpoints(explicit)
    tmp_cfg = REPO_ROOT / "config" / ".inference_config_runtime.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    from confuciustts.cli.inference import ConfuciusTTS

    t0 = time.time()
    model = ConfuciusTTS(config_path=str(tmp_cfg), device=device)
    codec = gc.load_repcodec(model.device, a.codec_path) if "continuation" in modes else None
    print(f"[both] models loaded on {device} in {time.time() - t0:.1f}s")

    if a.auto_transcribe and "continuation" in modes:
        prompt_text = gc.transcribe(a.prompt_wav, a.asr_model, device)
        if not a.no_normalize:
            prompt_text = gv.normalize_vietnamese(prompt_text)
        print(f"[both] ASR transcript: {prompt_text}")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pairs.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        for name, raw, text in zip(names, sentences, norm_sentences):
            row = {"name": name, "text": raw, "normalized": text, "prompt_wav": a.prompt_wav, "prompt_text": prompt_text,
                   "seed": a.seed, "t2s_checkpoint": a.t2s_checkpoint, "s2a_checkpoint": a.s2a_checkpoint}
            for mode in modes:
                torch.manual_seed(a.seed)
                t1 = time.time()
                try:
                    if mode == "reference":
                        audio = model.generate(
                            text, "vi", a.prompt_wav, raw=True, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                            num_beams=a.num_beams, repetition_penalty=a.repetition_penalty, n_timesteps=a.n_timesteps,
                            inference_cfg_rate=a.cfg_rate, max_text_tokens_per_segment=a.max_text_tokens_per_segment,
                            cross_fade_duration=a.cross_fade_duration,
                        )
                    else:
                        audio = gc.synth_continuation(model, codec, a.prompt_wav, prompt_text, text, "vi", a)
                except Exception as exc:  # noqa: BLE001
                    print(f"[both] {name} {mode}: FAILED {exc}")
                    row[f"{mode}_error"] = str(exc)
                    continue
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                path = out_dir / f"{name}.{mode}.wav"
                torchaudio.save(str(path), audio.cpu(), model.sample_rate)
                dur = audio.shape[-1] / model.sample_rate
                dt = time.time() - t1
                row[f"{mode}_wav"] = str(path)
                row[f"{mode}_seconds"] = round(dur, 3)
                row[f"{mode}_gen_time"] = round(dt, 2)
                print(f"[both] {name} {mode:12s}: {dur:5.2f}s audio in {dt:5.1f}s -> {path}")
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[both] done, log appended to {log_path}")


if __name__ == "__main__":
    main()
