# Plan: Vietnamese fine-tuning of Confucius4-TTS with a Mistral + Vistral hybrid text embedding

Status: materials ready, nothing trained yet. All numbers below were measured in this repo on 2026-10-05/06.

## 1. Problem

Confucius4-TTS feeds text to its T2S transformer through a frozen Mistral-7B v0.1 embedding table
(32,000 × 4,096) and the matching SentencePiece tokenizer. That tokenizer has no Vietnamese syllable
merges: of the 20 most frequent Vietnamese syllables only `là`, `có`, `cho` are single pieces, and
Vietnamese costs **2.75 tokens per syllable** (English: 1.23 per word).

| syllable | Mistral pieces |
|---|---|
| tiếng | ti, ế, ng |
| Việt | Vi, ệ, t |
| người | ng, ư, ờ, i |
| được | đ, ư, ợ, c |

Consequences: text prefixes 2.5× longer than necessary inside the 520-position table, segments capped at
~9 s by the 80-token default, and the model must compose syllables from characters instead of seeing
them as units. The text front end is where code-switching gains come from (BlueMagpie-TTS ablation), so
this is the highest-leverage text-side change available without retraining the acoustic stack.

## 2. What Vistral provides

`Viet-Mistral/Vistral-7B-Chat` is Mistral-7B v0.1 with the tokenizer extended by **6,369 Vietnamese
pieces** (vocab 32,000 → 38,369) and continual pretraining on Vietnamese text. Same architecture: hidden
4,096, 32 layers, 32/8 heads, FFN 14,336. Only the chat-tuned checkpoint is public (gated, manual
approval; ungated mirror `jan-hq/Vistral-7B-Chat-DPO`). License AFL-3.0.

Measured: **1.12 tokens per syllable**, `tiếng` → one piece.

Measured drift of the 32,000 shared rows after Vistral's continual pretraining (cosine Vistral vs Mistral):
mean 0.957, p10 0.935, zero rows identical. Not uniform: `请` 0.977, `▁deep` 0.963, `▁the` 0.834,
`▁là` 0.721. Rows frequent in Vietnamese text moved most; Chinese barely moved. A full 4096×4096 linear
map fitted on the shared rows only recovers mean cosine 0.962, so the drift is token-specific re-learning,
not a global rotation. New rows have mean norm 0.129 vs 0.172 for the originals.

## 3. Decision

Do **not** swap the whole table (shifts 14 pretrained languages and the Chinese instruction prefix).
Build a **hybrid table**:

- rows 0..31,999: Confucius4's existing Mistral rows, byte-identical → pretrained behaviour preserved exactly;
- rows 32,000..38,368: the 6,369 new Vietnamese pieces, initialised as
  `0.5 · mean(Mistral rows of the piece's Mistral sub-pieces) + 0.5 · 1.33 · Vistral row`
  (sub-piece average is already in Mistral's space; the Vistral row is rescaled to the original norm
  because SiLU in the projector is not scale-invariant), then two scale corrections measured with
  `check_projection.py`: each new row is rescaled to the old-row mean input norm 0.172 (the raw blend lands
  at 0.120 and the projector's outputs for new tokens come out 9 % too small), and then one global factor,
  0.78, is found by bisection so that the mean **projector output** norm of the new rows equals that of
  the old rows (2.094). Input norm alone overshoots by 22 % because the projector is nonlinear and the new
  rows' direction excites it more. After both steps new and old rows match on mean, p10 and p90 of the
  output norm, and a new piece's output has cosine 0.91 to the mean output of its Mistral sub-pieces.

The projector (fc1 4096→4096, SiLU, fc2 4096→1280; 22 M params) is a single function shared by all
tokens, so "projecting the new rows" is not a separate operation. Alignment is done by training the
**new rows only** (26 M params) with the projector frozen, which leaves old rows and the projector
bit-identical. Training the projector instead would move every token's mapping.

Semantic tokens are audio-side and unchanged → data `.npy` files, S2A and the vocoder are reused as is.

## 4. Steps

| # | Step | Tool | Output |
|---|---|---|---|
| 0 | Finish the Mistral-tokenizer fine-tune as the control (condition B) | `config/train_t2s_vi.yaml` | baseline checkpoint |
| 1 | Fetch Vistral tokenizer + embedding tensor only (314 MB, HTTP range request), verify IDs 0..31,999 match Mistral | `build_hybrid_embedding.py` | `tokenizer/`, `assets/vistral_embed_tokens.pt`, `report.json` |
| 2 | Build hybrid table, write T2S checkpoint with 38,369-row table | same script | `checkpoints/t2s_model_vistral_hybrid.safetensors` |
| 3 | Sanity-check: projector output norms for new vs old rows; tokens/syllable | `check_projection.py` | printed report |
| 4 | Stage A: train new embedding rows only, projector + transformer frozen, wd 0, ~1,500 steps, lr 1e-4 | `train_stage_a.py` + `config/train_t2s_vi_vistral_stageA.yaml` | `logs/t2s_vi_vistral_stageA/ckpt/` |
| 5 | Export Stage A to safetensors | `export_t2s.py` | `checkpoints/t2s_model_vistral_stageA.safetensors` |
| 6 | Stage B: full T2S fine-tune (same recipe as baseline, lr 3e-5), table frozen | `confuciustts/cli/train_t2s.py` + `config/train_t2s_vi_vistral_stageB.yaml` | `logs/t2s_vi_vistral_stageB/ckpt/` |
| 7 | Export Stage B | `export_t2s.py` | `checkpoints/t2s_model_vistral.safetensors` |
| 8 | Inference: use `config/inference_config_vistral.yaml`; lower segment cap to 60–70 tokens (80 Vistral tokens ≈ 70 syllables ≈ 22 s, near the 30 s semantic ceiling) | `scripts/generate_vi.py --config ...` | audio |
| 9 | Benchmark as condition `B_vistral` next to `A_pretrained` and `B_finetuned`, same test set and seeds | `benchmark/scripts/generate_all.py --confucius-config ...` | scores |

Fallback inside Stage A: if loss plateaus high or V1 CER is worse than baseline, additionally unfreeze
fc1/fc2 (`stage_a.train_projector: true`) with an anchor term keeping old-row outputs near their originals.

## 5. Config deltas

| key | baseline | Vistral variant |
|---|---|---|
| `t2s_model.vocab_size` | 32000 | 38369 |
| `paths.tokenizer_path` | `./checkpoints` | `vistral_finetune/tokenizer` |
| `paths.t2s_checkpoint` | `checkpoints/t2s_model.safetensors` | hybrid (stage A) → stage-A export (stage B) |
| `text_embedding_dim`, `max_text_seq_lens` | 4096, 520 | unchanged |
| inference `max_text_tokens_per_segment` | 80 | 60–70 |

The dataset loader rejects any token id ≥ `vocab_size`, so the first row is mandatory.

## 6. How to read the benchmark

- **V1 / V2** (similarity, PhoWhisper CER): did the swap cost anything? Must be ≥ baseline.
- **CS1–CS3** (English-span recall, within-utterance speaker consistency): did syllable units help at the switch?
- **XL1 / XL2** (similarity delta, Whisper WER/CER, language id): did keeping the Mistral rows protect
  cross-lingual transfer? Expected to hold, since those rows are untouched; a drop would come from the
  Vietnamese-only Stage B, which the baseline shares.

Keep the variant if XL holds, CS improves and V1 is not worse. Non-overlapping bootstrap intervals or a
listening-test preference above 60 % count as a lead, as in `benchmark/PLAN.md`.

## 7. Expected outcome and risks

Expected: clear gain in fine-tuning data efficiency and long-sentence handling (2.5× shorter text),
probable gain at code-switch boundaries, small change in raw intelligibility (the model already reaches
1.61 WER on Vietnamese with character-level pieces; Vietnamese orthography is regular).

Risks: the new rows may still sit off the manifold the GPT-2 learned after Stage A on a few dozen hours
(mitigation: fallback above); Vistral is chat-tuned only — irrelevant for embedding rows, which SFT barely
moves, but no base checkpoint exists for a cleaner comparison; the standalone `manh-linh/vistral-tokenizer`
is **not** Vistral's tokenizer (80,508 pieces, 2.19 tok/syl) — use the one shipped with the full weights.
