# vistral_finetune — Vietnamese fine-tuning with a Mistral + Vistral hybrid text table

Read `PLAN.md` first: it holds the reasoning, the measurements and the decision rule. This file is the
run order. All commands run from the repo root inside the project venv, on the GPU box.

```
vistral_finetune/
├── PLAN.md                         why, what was measured, how to judge the result
├── build_hybrid_embedding.py       step 1-2: fetch Vistral tokenizer + table, build hybrid, write checkpoint
├── check_projection.py             step 3: sanity checks on the hybrid table (CPU)
├── train_stage_a.py                step 4: train the 6,369 new rows only (wrapper around the stock trainer)
├── export_t2s.py                   step 5/7: Lightning .ckpt -> safetensors for inference
├── config/
│   ├── train_t2s_vi_vistral_stageA.yaml
│   ├── train_t2s_vi_vistral_stageB.yaml
│   └── inference_config_vistral.yaml
├── tokenizer/                      Vistral tokenizer (created by step 1)
├── assets/                         cached 314 MB embedding tensor (git-ignored)
└── report.json                     numbers from the build
```

## Run order

```bash
# 0. baseline (Mistral tokenizer) must exist for comparison -> scripts/README.md "Train"

# 1-2. hybrid table + checkpoint  (~1 min download, ~2.6 GB written)
python vistral_finetune/build_hybrid_embedding.py            # add --dry-run to check without writing

# 3. sanity check (CPU, <1 min)
python vistral_finetune/check_projection.py

# 4. stage A: new embedding rows only, ~1,500 steps
python vistral_finetune/train_stage_a.py -c vistral_finetune/config/train_t2s_vi_vistral_stageA.yaml

# 5. export stage A
python vistral_finetune/export_t2s.py logs/t2s_vi_vistral_stageA/ckpt/last.ckpt \
    checkpoints/t2s_model_vistral_stageA.safetensors

# 6. stage B: full fine-tune, same recipe as the baseline
python -m confuciustts.cli.train_t2s -c vistral_finetune/config/train_t2s_vi_vistral_stageB.yaml

# 7. export stage B
python vistral_finetune/export_t2s.py logs/t2s_vi_vistral_stageB/ckpt/last.ckpt \
    checkpoints/t2s_model_vistral.safetensors

# 8. generate
python scripts/generate_vi.py --config vistral_finetune/config/inference_config_vistral.yaml \
    --prompt-wav audio_samples/5e7117f231ca_00123600_00136600.wav \
    --text "Hôm nay trời đẹp, chúng ta đi dạo một vòng quanh hồ nhé." \
    --max-text-tokens-per-segment 64 --out data/vistral_test.wav

# 9. benchmark as a third condition (from the study/ folder)
Confucius4-TTS/.venv/bin/python benchmark/scripts/generate_all.py --model confucius4_tts --condition B_vistral \
    --confucius-config vistral_finetune/config/inference_config_vistral.yaml \
    --t2s-checkpoint Confucius4-TTS/checkpoints/t2s_model_vistral.safetensors --confucius-segment-tokens 64
```

## Notes

- Only the text table and tokenizer change. Semantic token `.npy` files, S2A and the vocoder are reused.
- `config/*.yaml` were generated from `config/train_t2s_vi.yaml` / `config/inference_config.yaml`; the data
  paths inside point at `data/vi_podcast_Confucius4_TTS/`, change them if your dataset lives elsewhere.
- Stage A keeps `weight_decay: 0` on purpose; with AdamW a non-zero value would shrink the frozen Mistral
  rows even though their gradient is masked.
- Fallback if stage A does not converge: set `stage_a.train_projector: true` (fc1/fc2 trainable with an
  anchor loss on the old rows), see PLAN.md §4.
- The scripts set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` because a stale `~/.cache/huggingface/token` makes public
  downloads fail with 401.
