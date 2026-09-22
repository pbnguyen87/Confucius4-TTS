# Vietnamese fine-tuning data for Confucius4-TTS

Scripts here turn the output of `speech_dataset/audio-pipeline` (stage `s7_loudnorm`)
into the TSV + `.npy` format that `confuciustts/dataset/t2s_dataset.py` and
`s2a_dataset.py` read.

## Files

| File | Purpose |
|---|---|
| `scripts/semantic_tokenizer.py` | audio → semantic tokens (w2v-BERT 2.0 layer 17 → stats normalization → MaskGCT RepCodec). Library + one-file CLI. |
| `scripts/prepare_confucius_data.py` | manifest → tier-A filter → tokens → reference clips → `train.tsv` / `val.tsv` / `summary.json` |
| `config/train_t2s_vi.yaml` | T2S fine-tune config pointing at `data/vi_podcast_Confucius4_TTS` |
| `config/train_s2a_vi.yaml` | S2A fine-tune config, same data |

## One-time setup

On a Linux GPU box follow the main README (conda, Python 3.10, torch 2.7.0) and add
`pip install einops` (RepCodec dependency missing from requirements.txt).

On this Intel Mac a venv already exists at `.venv` (created with `uv`, Python 3.11).
PyTorch stopped shipping Intel-Mac wheels after 2.2, so it holds torch/torchaudio 2.2.2 and
numba 0.60 / llvmlite 0.43 instead of the pinned versions; everything else matches
`requirements.txt`. Activate with `source .venv/bin/activate`. Use `--device cpu` when
running the token extractor here; MPS on Intel Macs is unreliable.

```bash
git clone https://github.com/open-mmlab/Amphion.git external/Amphion   # already done
huggingface-cli download netease-youdao/Confucius4-TTS --local-dir checkpoints   # only needed for training / inference
```

`facebook/w2v-bert-2.0` (~2.3 GB) and `amphion/MaskGCT` `semantic_codec/model.safetensors`
(~177 MB) are downloaded automatically on first run. Pass `--w2v-bert-path` /
`--codec-local-path` to use local copies.

## Build the dataset

```bash
# 1. check what survives the tier-A filter, no downloads
python scripts/prepare_confucius_data.py \
    --s7-dir ../speech_dataset/audio-pipeline/work_5min/s7_loudnorm \
    --out-dir data/vi_podcast_Confucius4_TTS --lang vi --dry-run

# 2. full run (GPU recommended; resumable, existing .npy are reused)
python scripts/prepare_confucius_data.py \
    --s7-dir ../speech_dataset/audio-pipeline/work_5min/s7_loudnorm \
    --out-dir data/vi_podcast_Confucius4_TTS --lang vi --device cuda
```

Output layout:

```
data/vi_podcast_Confucius4_TTS/
├── train.tsv          lang \t wav_path \t norm_text \t semantic_ids_path \t ref_audio_paths
├── val.tsv
├── semantic/<id>.npy  int64 semantic tokens, ~50 per second of audio
├── accepted.jsonl     manifest rows that made it in, with resolved paths
└── summary.json       counts, hours, rejection reasons, thresholds used
```

Filter defaults are the pipeline's tier A (`config/default.yaml`): CER ≤ 0.02, SNR ≥ 20 dB
(or DNSMOS ≥ 3.0 when present), clipping ≤ 0.1 %, 2–13 s, single speaker. Relax with
`--max-cer 0.15 --min-snr-db 10 --max-seconds 25` etc. when data is scarce.

Reference clips: for each utterance, the `--num-refs` best other clips (3–15 s) of the same
speaker. Speakers with fewer than `--min-utts-per-speaker` accepted clips are dropped.
Train/val is split **within** each speaker (default 5 % val) because the goal is to learn
these voices, unlike the pipeline's speaker-disjoint split.

## Train

```bash
python -m confuciustts.cli.train_t2s -c config/train_t2s_vi.yaml
python -m confuciustts.cli.train_s2a -c config/train_s2a_vi.yaml   # optional second stage
```

## Generate Vietnamese speech

`scripts/generate_vi.py` wraps `ConfuciusTTS` with two fixes the stock `example.py` lacks:
Vietnamese number/abbreviation normalization (the repo's normalizer would read "2" as
"two"), and a local-checkpoint resolver so `checkpoints/*.safetensors`, `pretrained/` and
fine-tuned exports are used instead of being re-downloaded from Hugging Face.

```bash
# zero-shot with the pretrained weights (needs checkpoints/s2a_model.pt too:
#   hf download netease-youdao/Confucius4-TTS s2a_model.pt --local-dir checkpoints)
python scripts/generate_vi.py \
    --prompt-wav ../speech_dataset/audio-pipeline/work_5min/s7_loudnorm/audio/5e7117f231ca_00001400_00011700.wav \
    --text "Hôm nay là 22/9/2026, nhiệt độ 31,5 độ." --out data/test_vi.wav --device cpu

# check normalization only
python scripts/generate_vi.py --text "Giá 1.250.000 đ" --prompt-wav x.wav --print-normalized

# fine-tuned T2S
python scripts/generate_vi.py --prompt-wav ref.wav --text-file input.txt \
    --t2s-checkpoint path/to/finetuned_t2s.safetensors --out out.wav
```

CAM++ (`funasr/campplus`) and BigVGAN v2 22 kHz are still fetched from Hugging Face on
first run (~0.5 GB). A runtime copy of the config is written to
`config/.inference_config_runtime.yaml`.

## Notes

- The dataset loaders silently replace any failing sample with a random one, so keep an eye
  on `summary.json` and the warnings this script prints rather than on training logs.
- `confuciustts/frontend/semantic_extractor.py` calls `codec_model.encode`, which `RepCodec`
  does not have; this script uses `RepCodec.quantize` and loads the public safetensors, so
  it does not depend on that class.
- Text is taken from the pipeline's `text_normalized` (vinorm). The repo's own
  `TextNormalizer` has no Vietnamese branch, so normalize upstream, not here.
