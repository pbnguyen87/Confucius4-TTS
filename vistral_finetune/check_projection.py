"""Sanity checks on the hybrid table before any training (PLAN.md step 3).

  * projector output norm for new-row tokens vs old-row tokens (should be the same range);
  * cosine between projector outputs of a new piece and the mean of its Mistral sub-pieces' outputs
    (how far the init already lands from the compositional representation the model knows);
  * tokens per syllable, Mistral vs Vistral, on a few Vietnamese sentences;
  * share of new-row tokens in real Vietnamese text (how much of the input the warm-up actually affects).

Runs on CPU in under a minute. Reads the hybrid checkpoint if it exists, otherwise builds the table in memory.

    python vistral_finetune/check_projection.py [--checkpoint checkpoints/t2s_model_vistral_hybrid.safetensors]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SENTENCES = [
    "Hôm nay trời đẹp, chúng ta đi dạo một vòng quanh hồ nhé.",
    "Nguyễn Văn Đức đã nghiên cứu những vấn đề phức tạp về trí tuệ nhân tạo.",
    "Mô hình mới có thể tạo giọng nói tiếng Việt tự nhiên hơn, kể cả khi câu có chèn từ tiếng Anh như deep learning, server, hay deadline.",
    "Bệnh cường giáp là một rối loạn nội tiết xảy ra khi tuyến giáp sản xuất quá mức hormone thyroxine.",
    "Bạn nhớ check mail và update lại file báo cáo trước deadline chiều nay nhé.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "t2s_model_vistral_hybrid.safetensors"))
    ap.add_argument("--base-checkpoint", default=str(REPO_ROOT / "checkpoints" / "t2s_model.safetensors"))
    ap.add_argument("--tokenizer", default=str(HERE / "tokenizer"))
    a = ap.parse_args()

    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from confuciustts.llm.text_encoder import TextEmbeddingProjector

    mis_tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / "checkpoints"), token=False)
    vis_tok = AutoTokenizer.from_pretrained(a.tokenizer, token=False)

    with safe_open(a.base_checkpoint, "pt") as f:
        w = {k: f.get_tensor(k) for k in f.keys() if k.startswith("text_projector.")}
    if Path(a.checkpoint).is_file():
        with safe_open(a.checkpoint, "pt") as f:
            table = f.get_tensor("text_projector.embed.weight").float()
        src = a.checkpoint
    else:
        from build_hybrid_embedding import build_table, fetch_embed_tokens, MISTRAL_ROWS
        vis_table = fetch_embed_tokens("jan-hq/Vistral-7B-Chat-DPO", HERE / "assets")
        pw = {"fc1_w": w["text_projector.text_projection_fc1.weight"].float(), "fc1_b": w["text_projector.text_projection_fc1.bias"].float(),
              "fc2_w": w["text_projector.text_projection_fc2.weight"].float(), "fc2_b": w["text_projector.text_projection_fc2.bias"].float()}
        table, _ = build_table(w["text_projector.embed.weight"].float(), vis_table, mis_tok, vis_tok, "blend", 0.5, proj_weights=pw)
        src = "built in memory"
    V, D = table.shape
    print(f"table {V}x{D} from {src}")

    proj = TextEmbeddingProjector(vocab_size=V, embed_dim=D, output_size=w["text_projector.text_projection_fc2.weight"].shape[0])
    proj.embed.weight.data.copy_(table)
    proj.text_projection_fc1.weight.data.copy_(w["text_projector.text_projection_fc1.weight"].float())
    proj.text_projection_fc1.bias.data.copy_(w["text_projector.text_projection_fc1.bias"].float())
    proj.text_projection_fc2.weight.data.copy_(w["text_projector.text_projection_fc2.weight"].float())
    proj.text_projection_fc2.bias.data.copy_(w["text_projector.text_projection_fc2.bias"].float())
    proj.eval()

    with torch.no_grad():
        out_all = proj(torch.arange(V))                       # (V, 1280)
    n_old = out_all[:32000].norm(dim=1); n_new = out_all[32000:].norm(dim=1)
    print("\nprojector output norm   old rows: mean %.3f  p10 %.3f  p90 %.3f" % (n_old.mean(), n_old.quantile(.1), n_old.quantile(.9)))
    print("                        new rows: mean %.3f  p10 %.3f  p90 %.3f" % (n_new.mean(), n_new.quantile(.1), n_new.quantile(.9)))

    # new piece output vs mean output of its Mistral sub-pieces
    cos_list = []
    with torch.no_grad():
        for tid in range(32000, V):
            text = vis_tok.convert_ids_to_tokens(tid).replace("▁", " ")
            sub = [s for s in mis_tok.encode(text, add_special_tokens=False) if s < 32000]
            if not sub:
                continue
            cos_list.append(torch.nn.functional.cosine_similarity(out_all[tid], out_all[sub].mean(0), dim=0).item())
    c = torch.tensor(cos_list)
    print("cos(new piece output, mean of its sub-piece outputs): mean %.3f  p10 %.3f  p90 %.3f" % (c.mean(), c.quantile(.1), c.quantile(.9)))

    print("\ntokens per syllable        Mistral  Vistral   new-row share")
    for s in SENTENCES:
        syl = len([x for x in re.sub(r"[.,!?;:]", " ", s).split() if x])
        mi = mis_tok.encode(s, add_special_tokens=False); vi = vis_tok.encode(s, add_special_tokens=False)
        share = sum(1 for t in vi if t >= 32000) / len(vi)
        print(f"  {len(mi)/syl:5.2f}    {len(vi)/syl:5.2f}     {share:4.0%}   {s[:60]}")
    prefix = "You are a helpful assistant. 请用越南语朗读接下来的文字:"
    pm, pv = mis_tok.encode(prefix, add_special_tokens=False), vis_tok.encode(prefix, add_special_tokens=False)
    print(f"\ninstruction prefix: Mistral {len(pm)} tokens, Vistral {len(pv)} tokens, identical ids: {pm == pv}")


if __name__ == "__main__":
    main()
