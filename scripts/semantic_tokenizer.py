"""Semantic tokenizer for Confucius4-TTS training data.

Turns a waveform into the semantic token sequence that the T2S model predicts:

    audio -> 16 kHz mono -> Wav2Vec2-BERT 2.0 hidden state (layer 17)
          -> normalize with checkpoints/wav2vec2bert_stats.pt (mean / sqrt(var))
          -> MaskGCT RepCodec.quantize -> int64 indices at 50 Hz

This mirrors exactly what ``confuciustts/cli/t2s_lightning.py`` does at training
time, so the tokens written here match the distribution the pretrained T2S has
seen. The RepCodec weights are read from the public ``amphion/MaskGCT`` repo
(``semantic_codec/model.safetensors``) instead of the internal ``.pt`` format
expected by ``confuciustts/frontend/semantic_extractor.py``.

Usage as a library::

    tok = SemanticTokenizer(device="cuda")
    codes = tok.encode_file("utt.wav")     # np.ndarray, shape (T,), dtype int64

Usage as a CLI (one file, prints length and tokens/sec)::

    python scripts/semantic_tokenizer.py path/to/utt.wav --out utt.npy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
AMPHION_PATH = REPO_ROOT / "external" / "Amphion"
if str(AMPHION_PATH) not in sys.path:
    sys.path.insert(0, str(AMPHION_PATH))

# Defaults taken from external/Amphion/models/tts/maskgct/config/maskgct.json
DEFAULT_CODEC_CFG = dict(
    codebook_size=8192,
    hidden_size=1024,
    codebook_dim=8,
    vocos_dim=384,
    vocos_intermediate_dim=2048,
    vocos_num_layers=12,
)
DEFAULT_CODEC_REPO = "amphion/MaskGCT"
DEFAULT_CODEC_FILE = "semantic_codec/model.safetensors"
DEFAULT_W2V_BERT = "facebook/w2v-bert-2.0"
DEFAULT_STATS = REPO_ROOT / "checkpoints" / "wav2vec2bert_stats.pt"

W2V_SAMPLE_RATE = 16000
W2V_HIDDEN_LAYER = 17
SEMANTIC_FRAME_RATE_HZ = 50.0  # Wav2Vec2-BERT 2.0 frame rate; used only for sanity checks


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SemanticTokenizer:
    def __init__(
        self,
        w2v_bert_path: str = DEFAULT_W2V_BERT,
        stats_path: os.PathLike | str = DEFAULT_STATS,
        codec_repo: str = DEFAULT_CODEC_REPO,
        codec_file: str = DEFAULT_CODEC_FILE,
        codec_local_path: Optional[str] = None,
        codec_cfg: Optional[dict] = None,
        device: str | torch.device = "auto",
    ) -> None:
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

        self.device = resolve_device(device) if isinstance(device, str) else device

        # 1. Wav2Vec2-BERT 2.0 feature extractor + encoder
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_path)
        self.w2v_model = Wav2Vec2BertModel.from_pretrained(w2v_bert_path).eval().to(self.device)
        for p in self.w2v_model.parameters():
            p.requires_grad_(False)

        # 2. Normalization statistics shipped with Confucius4-TTS
        stats = torch.load(str(stats_path), map_location="cpu")
        if "mean" not in stats or "var" not in stats:
            raise KeyError(f"{stats_path} must contain 'mean' and 'var', got keys {list(stats)}")
        self.semantic_mean = stats["mean"].float().to(self.device)
        self.semantic_std = torch.sqrt(stats["var"].float()).to(self.device)

        # 3. MaskGCT semantic codec (RepCodec)
        self.codec = self._build_codec(codec_repo, codec_file, codec_local_path, codec_cfg)

    def _build_codec(self, repo: str, filename: str, local_path: Optional[str], cfg: Optional[dict]):
        try:
            from models.codec.kmeans.repcodec_model import RepCodec
        except ImportError as e:
            raise ImportError(
                f"Cannot import Amphion RepCodec from {AMPHION_PATH}. "
                "Clone it first: git clone https://github.com/open-mmlab/Amphion.git external/Amphion"
            ) from e
        import safetensors.torch

        merged = dict(DEFAULT_CODEC_CFG)
        if cfg:
            merged.update(cfg)
        codec = RepCodec(cfg=SimpleNamespace(**merged))

        if local_path is None:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(repo, filename=filename)
        safetensors.torch.load_model(codec, local_path)
        codec.eval().to(self.device)
        for p in codec.parameters():
            p.requires_grad_(False)
        return codec

    @staticmethod
    def load_audio_16k(path: os.PathLike | str) -> torch.Tensor:
        """Load any audio file as mono 16 kHz, shape (T,)."""
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != W2V_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, W2V_SAMPLE_RATE)
        return wav.squeeze(0)

    @torch.no_grad()
    def extract_features(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Normalized layer-17 hidden states, shape (1, T_feat, 1024)."""
        inputs = self.feature_extractor(
            wav_16k.cpu().numpy(), sampling_rate=W2V_SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        out = self.w2v_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = out.hidden_states[W2V_HIDDEN_LAYER]
        return (feat - self.semantic_mean) / self.semantic_std

    @torch.no_grad()
    def encode_waveform(self, wav_16k: torch.Tensor) -> np.ndarray:
        feat = self.extract_features(wav_16k)
        indices, _ = self.codec.quantize(feat)  # (T,) or (1, T)
        indices = indices.reshape(-1)
        return indices.detach().cpu().to(torch.int64).numpy()

    def encode_file(self, path: os.PathLike | str) -> np.ndarray:
        return self.encode_waveform(self.load_audio_16k(path))


def _main() -> None:
    ap = argparse.ArgumentParser(description="Encode one audio file into Confucius4-TTS semantic tokens")
    ap.add_argument("audio", help="input audio file")
    ap.add_argument("--out", help="output .npy path (default: print only)")
    ap.add_argument("--w2v-bert-path", default=DEFAULT_W2V_BERT)
    ap.add_argument("--stats-path", default=str(DEFAULT_STATS))
    ap.add_argument("--codec-local-path", default=None, help="local semantic_codec safetensors, skips HF download")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = ap.parse_args()

    tok = SemanticTokenizer(
        w2v_bert_path=args.w2v_bert_path,
        stats_path=args.stats_path,
        codec_local_path=args.codec_local_path,
        device=args.device,
    )
    wav = tok.load_audio_16k(args.audio)
    codes = tok.encode_waveform(wav)
    dur = wav.numel() / W2V_SAMPLE_RATE
    print(f"{args.audio}: {len(codes)} tokens, {dur:.2f} s, {len(codes) / max(dur, 1e-6):.1f} tokens/s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, codes)
        print(f"saved {args.out}")


if __name__ == "__main__":
    _main()
