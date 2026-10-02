"""Vietnamese generation with Confucius4-TTS in *continuation* (in-context) mode.

The released code only implements *reference* mode: the prompt clip is compressed into a
speaker vector (T2S) and a mel prompt (S2A), and the whole target text is generated from
scratch. The paper also reports a *continuation* mode (README table row
"Confucius4-TTS (Continuation)"): the prompt's transcript is prepended to the target text and
the prompt's semantic tokens are prepended to the output sequence, so T2S literally continues
the prompt utterance. That needs the prompt transcript, which is why it was left out of the
public inference code. This script adds it on top of the released weights.

T2S input layout (from ``Text2Semantic._prepare_embed_inputs``)::

    [speaker slot] [text tokens ...] [BOS] [semantic tokens ...] [EOS]

Continuation mode fills it as::

    [speaker slot] [prefix + prompt_text + " " + target_text] [BOS] [prompt_codes] -> model continues

where ``prompt_codes`` are MaskGCT RepCodec tokens of the prompt clip (same tokenizer used to
build the fine-tuning data, see ``scripts/semantic_tokenizer.py``). The generated tokens
after the prompt are cut out and sent to S2A exactly as in reference mode (mel prompt + CAM++
style vector), so S2A is unchanged.

Caveats
-------
* The exact training format of continuation mode is not documented. We join prompt and
  target text with a space (adding a period to the prompt text if it has none). Change it
  with ``--join``.
* HF's repetition penalty would punish every token that appears in the prompt. We replace it
  with a penalty over the *generated* tokens only (``--penalize-prompt`` restores HF
  behaviour for comparison).
* Prompt clips longer than 15 s exceed what T2S saw in training; keep them at 3-15 s.

Examples::

    python scripts/generate_vi_continuation.py \\
        --prompt-wav ref.wav --prompt-text "Xin chào các bạn, hôm nay chúng ta nói về trí tuệ nhân tạo." \\
        --text "Mô hình mới có thể tạo giọng nói tiếng Việt tự nhiên hơn." --out data/cont_vi.wav

    # transcribe the prompt automatically with PhoWhisper, also write the reference-mode output for A/B
    python scripts/generate_vi_continuation.py --prompt-wav ref.wav --auto-transcribe \\
        --text-file input.txt --out data/cont_vi.wav --also-reference
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_vi as gv  # noqa: E402  (sets cwd to REPO_ROOT, provides normalizer + checkpoint patching)
from semantic_tokenizer import DEFAULT_CODEC_FILE, DEFAULT_CODEC_REPO, SemanticTokenizer  # noqa: E402

LOCAL_CODEC = REPO_ROOT / "pretrained" / "MaskGCT" / "semantic_codec" / "model.safetensors"
MAX_PROMPT_SECONDS = 15.0  # MAX_PROMPT_AUDIO_DURATION_SEC in the training dataset
_END_PUNCT = set(".?!;:。？！")


# --------------------------------------------------------------------------- helpers
def load_repcodec(device, local_path: Optional[str] = None):
    """MaskGCT RepCodec only (w2v-BERT is reused from ConfuciusTTS, so it is not loaded twice)."""
    tok = SemanticTokenizer.__new__(SemanticTokenizer)  # skip __init__, we only need _build_codec
    tok.device = device
    local = local_path or (str(LOCAL_CODEC) if LOCAL_CODEC.is_file() else None)
    return tok._build_codec(DEFAULT_CODEC_REPO, DEFAULT_CODEC_FILE, local, None)


def join_prompt_and_target(prompt_text: str, target: str, join: str = " ") -> str:
    p = prompt_text.strip()
    if p and p[-1] not in _END_PUNCT:
        p += "."
    return f"{p}{join}{target.strip()}"


def make_generated_only_penalty(penalty: float, start_index: int):
    """Repetition penalty applied to tokens generated after ``start_index`` only."""
    import torch
    from transformers import LogitsProcessor

    class _Penalty(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            gen = input_ids[:, start_index:]
            if gen.shape[1] == 0:
                return scores
            score = torch.gather(scores, 1, gen)
            score = torch.where(score < 0, score * penalty, score / penalty)
            return scores.scatter(1, gen, score)

    return _Penalty()


# --------------------------------------------------------------------------- T2S continuation
def t2s_continue(
    t2s,
    text_ids,
    condition_vector,
    prompt_codes,
    *,
    max_new_tokens: int = 600,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 30,
    num_beams: int = 3,
    repetition_penalty: float = 10.0,
    penalize_prompt: bool = False,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Continue ``prompt_codes`` given the joined text. Returns (codes (1,N), latent (1,N,D))."""
    import torch
    from transformers import LogitsProcessorList
    from confuciustts.llm.llm import Text2Semantic

    cfg = t2s.config
    device = text_ids.device
    bos, eos = cfg.start_semantic_token, cfg.stop_semantic_token
    prompt = prompt_codes.reshape(1, -1).to(device=device, dtype=torch.long)
    M = prompt.shape[1]
    max_sem = getattr(cfg, "max_semantic_seq_lens", 1520)
    if M + 2 >= max_sem:
        raise ValueError(f"prompt has {M} semantic tokens, limit is {max_sem - 2}; use a shorter prompt clip")
    max_new = min(max_new_tokens, max_sem - (M + 2))

    t2s.store_conditioning(condition_vector, text_ids)
    prefix_len = 1 + text_ids.shape[1]  # speaker slot + text
    # placeholders for the prefix positions (ignored by _prepare_embed_inputs), then BOS + prompt codes
    start = torch.cat([torch.full((1, prefix_len + 1), bos, dtype=torch.long, device=device), prompt], dim=1)
    attention_mask = torch.ones_like(start)

    gen_kwargs = dict(
        attention_mask=attention_mask,
        max_length=start.shape[1] + max_new,
        do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p,
        num_beams=num_beams, **({"early_stopping": True} if num_beams > 1 else {}),
        eos_token_id=eos, pad_token_id=eos,
    )
    if penalize_prompt:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    elif repetition_penalty != 1.0:
        gen_kwargs["logits_processor"] = LogitsProcessorList([make_generated_only_penalty(repetition_penalty, start.shape[1])])

    with torch.no_grad():
        generated = super(Text2Semantic, t2s).generate(start, **gen_kwargs)

    gen_part = generated[0, start.shape[1]:]
    eos_pos = (gen_part == eos).nonzero()
    gen_part = gen_part[: int(eos_pos[0])] if len(eos_pos) else gen_part
    if gen_part.numel() == 0:
        raise RuntimeError("T2S produced no tokens after the prompt (EOS immediately); try another seed or a shorter prompt")
    N = gen_part.shape[0]

    # latent for S2A: hidden state at the position *before* each generated token
    seq = torch.cat([torch.tensor([bos], device=device), prompt[0], gen_part, torch.tensor([eos], device=device)]).unsqueeze(0)
    with torch.no_grad():
        emb = t2s._prepare_embed_inputs(text_inputs=text_ids, semantic_codes=seq, condition_vector=condition_vector)
        h = t2s.transformer(inputs_embeds=emb, use_cache=False, return_dict=True).last_hidden_state
    latent = h[:, prefix_len + M: -2]  # positions p_M, g_1 .. g_{N-1}
    assert latent.shape[1] == N, (latent.shape, N)
    return gen_part.unsqueeze(0), latent


def encode_text(tokenizer, lang: str, text: str, device):
    from confuciustts.utils.text_utils import LANGUAGE_TOKEN_MAP

    lang_token = LANGUAGE_TOKEN_MAP.get(lang, f"请用{lang}朗读接下来的文字")
    return tokenizer.encode(f"You are a helpful assistant. {lang_token}:{text}", return_tensors="pt").to(device)


def synth_continuation(model, codec, prompt_wav: str, prompt_text: str, text: str, lang: str, a) -> "torch.Tensor":
    """Full pipeline: prompt tokens -> T2S continuation per segment -> S2A -> BigVGAN."""
    import torch
    from confuciustts.utils.audio_post import cross_fade_concat

    wav_16k, wav_tgt = model._load_prompt(prompt_wav)
    dur = wav_16k.shape[-1] / 16000
    if dur > MAX_PROMPT_SECONDS:
        print(f"[continuation] WARNING prompt is {dur:.1f}s, training used <= {MAX_PROMPT_SECONDS:.0f}s; expect degraded output")
    semantic_features = model._extract_semantic(wav_16k)            # (1, T, 1024) normalized w2v-BERT L17
    style_embedding = model._extract_style(wav_16k)
    reference_mel = model._ref_mel(wav_tgt)
    with torch.no_grad():
        idx, _ = codec.quantize(semantic_features)
    prompt_codes = idx.reshape(-1)
    print(f"[continuation] prompt: {dur:.1f}s, {prompt_codes.numel()} semantic tokens, text {len(prompt_text.split())} words")

    segments = model.normalizer.segment_text(text, tokenize_fn=model.tokenizer.tokenize, language=lang,
                                             max_tokens=a.max_text_tokens_per_segment) or [text]
    chunks = []
    for i, seg in enumerate(segments):
        joined = join_prompt_and_target(prompt_text, seg, a.join)
        text_ids = encode_text(model.tokenizer, lang, joined, model.device)
        max_text = getattr(model.t2s_model.config, "max_text_seq_lens", 520)
        if text_ids.shape[1] >= max_text:
            raise ValueError(f"segment {i}: {text_ids.shape[1]} text tokens >= {max_text}; shorten the prompt text or the segment")
        t0 = time.time()
        codes, latent = t2s_continue(
            model.t2s_model, text_ids, semantic_features, prompt_codes,
            max_new_tokens=a.max_new_tokens, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
            num_beams=a.num_beams, repetition_penalty=a.repetition_penalty, penalize_prompt=a.penalize_prompt,
        )
        N = codes.shape[1]
        print(f"[continuation] segment {i + 1}/{len(segments)}: {text_ids.shape[1]} text tokens -> {N} new semantic tokens (~{N / 50:.1f}s) in {time.time() - t0:.1f}s")
        with torch.no_grad():
            mel = model.s2a_model.inference(
                semantic_token=codes, lm_latent=latent, prompt_feat=reference_mel, embedding=style_embedding,
                target_feat_len=torch.tensor([int(N * 1.72)], device=model.device),
                n_timesteps=a.n_timesteps, inference_cfg_rate=a.cfg_rate,
            )
            audio = model.bigvgan(mel.float().to(model.device)).squeeze(1)
        chunks.append(audio if audio.dim() == 2 else audio.unsqueeze(0))
    return cross_fade_concat(chunks, model.sample_rate, silence_duration=a.cross_fade_duration)


def transcribe(prompt_wav: str, asr_model: str, device: str) -> str:
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model=asr_model, device=0 if device.startswith("cuda") else -1)
    out = pipe(prompt_wav, generate_kwargs={"language": "vietnamese", "task": "transcribe"})
    return out["text"].strip()


# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Vietnamese continuation-mode generation with Confucius4-TTS",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--text-file")
    ap.add_argument("--prompt-wav", required=True, help="reference clip, 3-15 s, single speaker")
    pt = ap.add_mutually_exclusive_group(required=True)
    pt.add_argument("--prompt-text", help="exact transcript of the reference clip")
    pt.add_argument("--prompt-text-file")
    pt.add_argument("--auto-transcribe", action="store_true", help="transcribe the clip with --asr-model (PhoWhisper)")
    ap.add_argument("--asr-model", default="vinai/PhoWhisper-large")
    ap.add_argument("--out", default="data/output_vi_continuation.wav")
    ap.add_argument("--also-reference", action="store_true", help="also write <out>.reference.wav with the stock reference mode, same seed")
    ap.add_argument("--config", default="config/inference_config.yaml")
    ap.add_argument("--w2v-bert-path", default="pretrained/w2v-bert-2.0" if (REPO_ROOT / "pretrained/w2v-bert-2.0").is_dir() else None)
    ap.add_argument("--t2s-checkpoint", default=None)
    ap.add_argument("--s2a-checkpoint", default=None)
    ap.add_argument("--codec-path", default=None, help="MaskGCT semantic_codec safetensors; default pretrained/MaskGCT/... or HF download")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-normalize", action="store_true", help="texts are already normalized")
    ap.add_argument("--join", default=" ", help="string between prompt text and target text")
    g = ap.add_argument_group("sampling")
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-p", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=30)
    g.add_argument("--num-beams", type=int, default=3)
    g.add_argument("--repetition-penalty", type=float, default=10.0, help="applied to generated tokens only unless --penalize-prompt")
    g.add_argument("--penalize-prompt", action="store_true", help="use HF repetition penalty over prompt + generated tokens")
    g.add_argument("--max-new-tokens", type=int, default=600, help="semantic tokens per segment (50/s)")
    g.add_argument("--n-timesteps", type=int, default=25)
    g.add_argument("--cfg-rate", type=float, default=0.7)
    g.add_argument("--max-text-tokens-per-segment", type=int, default=80)
    g.add_argument("--cross-fade-duration", type=float, default=0.3)
    g.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    for k in ("prompt_wav", "out", "text_file", "prompt_text_file", "t2s_checkpoint", "s2a_checkpoint", "codec_path"):
        setattr(a, k, gv._user_path(getattr(a, k)))
    if not Path(a.prompt_wav).is_file():
        sys.exit(f"[continuation] reference audio not found: {a.prompt_wav}")

    import torch
    import torchaudio
    import yaml

    device = gv.resolve_device(a.device)
    text = a.text if a.text is not None else Path(a.text_file).read_text(encoding="utf-8")
    if a.auto_transcribe:
        prompt_text = transcribe(a.prompt_wav, a.asr_model, device)
        print(f"[continuation] ASR transcript: {prompt_text}")
    else:
        prompt_text = a.prompt_text if a.prompt_text is not None else Path(a.prompt_text_file).read_text(encoding="utf-8")
    norm = text.strip() if a.no_normalize else gv.normalize_vietnamese(text)
    prompt_norm = prompt_text.strip() if a.no_normalize else gv.normalize_vietnamese(prompt_text)
    print(f"[continuation] prompt text: {prompt_norm[:200]}")
    print(f"[continuation] target text: {norm[:200]}")

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
    codec = load_repcodec(model.device, a.codec_path)
    print(f"[continuation] models loaded on {device} in {time.time() - t0:.1f}s")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.seed is not None:
        torch.manual_seed(a.seed)
    t1 = time.time()
    audio = synth_continuation(model, codec, a.prompt_wav, prompt_norm, norm, "vi", a)
    dur = audio.shape[-1] / model.sample_rate
    torchaudio.save(str(out), audio.cpu(), model.sample_rate)
    print(f"[continuation] saved {out}: {dur:.2f}s audio in {time.time() - t1:.1f}s (RTF {(time.time() - t1) / max(dur, 1e-6):.2f})")

    if a.also_reference:
        if a.seed is not None:
            torch.manual_seed(a.seed)
        ref_out = out.with_suffix(".reference.wav")
        audio_ref = model.generate(norm, "vi", a.prompt_wav, raw=True, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                   num_beams=a.num_beams, repetition_penalty=a.repetition_penalty, n_timesteps=a.n_timesteps,
                                   inference_cfg_rate=a.cfg_rate, max_text_tokens_per_segment=a.max_text_tokens_per_segment)
        torchaudio.save(str(ref_out), audio_ref.cpu(), model.sample_rate)
        print(f"[continuation] reference-mode output for comparison: {ref_out} ({audio_ref.shape[-1] / model.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
