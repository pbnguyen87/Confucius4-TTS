"""Stage A: align the 6,369 new Vietnamese embedding rows, everything else frozen (PLAN.md step 4).

Reuses the stock trainer (``confuciustts/cli/train_t2s.py``) and Lightning module unchanged, and only
changes which parameters receive gradients:

  * all T2S parameters frozen;
  * ``text_projector.embed.weight`` made trainable, with a gradient hook zeroing rows < 32,000 so the
    Mistral rows stay bit-identical (use ``weight_decay: 0`` in the config so AdamW does not shrink them);
  * optionally (``stage_a.train_projector: true``) fc1/fc2 trainable as well, with an anchor loss keeping
    their outputs on the old rows close to the original projector's outputs.

The stock ``TextEmbeddingProjector.forward`` wraps the lookup in ``torch.no_grad()``; that is replaced here
so gradients can reach the new rows at all.

    python vistral_finetune/train_stage_a.py -c vistral_finetune/config/train_t2s_vi_vistral_stageA.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import torch  # noqa: E402
from pytorch_lightning import seed_everything  # noqa: E402
from pytorch_lightning.utilities.rank_zero import rank_zero_info  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from confuciustts.cli.t2s_lightning import T2SLightningModule  # noqa: E402
from confuciustts.cli.train_t2s import create_trainer, get_latest_checkpoint, print_param_info  # noqa: E402
from confuciustts.dataset.t2s_dataset import T2SDataModule  # noqa: E402
from confuciustts.llm import text_encoder  # noqa: E402
from confuciustts.utils.common import load_yaml_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stage_a")
FROZEN_ROWS = 32000


def _forward_with_grad(self, text_ids):
    """TextEmbeddingProjector.forward without the no_grad around the lookup."""
    return self.text_projection_fc2(self.act_fn(self.text_projection_fc1(self.embed(text_ids))))


class StageAModule(T2SLightningModule):
    """Adds the optional anchor loss for the train_projector fallback."""

    def __init__(self, config):
        super().__init__(config)
        sa = config.get("stage_a", {})
        self.anchor_weight = float(sa.get("anchor_weight", 0.0)) if sa.get("train_projector", False) else 0.0
        if self.anchor_weight > 0:
            proj = self.t2s_model.text_projector
            with torch.no_grad():
                ids = torch.arange(FROZEN_ROWS)
                self.register_buffer("anchor_ids", ids, persistent=False)
                self.register_buffer("anchor_ref", proj(ids).detach().clone(), persistent=False)

    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        if self.anchor_weight <= 0:
            return out
        loss = out["loss"] if isinstance(out, dict) else out
        sub = self.anchor_ids[torch.randint(0, FROZEN_ROWS, (2048,), device=self.anchor_ids.device)]
        cur = self.t2s_model.text_projector(sub)
        anchor = torch.nn.functional.mse_loss(cur, self.anchor_ref[sub].to(cur.dtype))
        self.log("train/anchor", anchor, prog_bar=True)
        total = loss + self.anchor_weight * anchor
        return {**out, "loss": total} if isinstance(out, dict) else total


def freeze_for_stage_a(model: T2SLightningModule, train_projector: bool) -> None:
    for p in model.t2s_model.parameters():
        p.requires_grad = False
    emb = model.t2s_model.text_projector.embed.weight
    emb.requires_grad = True

    def mask_old_rows(grad):
        grad = grad.clone()
        grad[:FROZEN_ROWS] = 0
        return grad

    emb.register_hook(mask_old_rows)
    if train_projector:
        for m in (model.t2s_model.text_projector.text_projection_fc1, model.t2s_model.text_projector.text_projection_fc2):
            for p in m.parameters():
                p.requires_grad = True
    rank_zero_info(f">> Stage A: trainable = new embedding rows {FROZEN_ROWS}..{emb.shape[0] - 1}"
                   + (" + projector fc1/fc2" if train_projector else ""))


def main(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    if config["t2s_model"]["vocab_size"] <= FROZEN_ROWS:
        sys.exit("vocab_size must be the hybrid table size (38369), see PLAN.md §5")
    if float(config.get("optimizer", {}).get("weight_decay", 0.01)) != 0.0:
        log.warning("optimizer.weight_decay is not 0: AdamW will shrink the frozen Mistral rows too. Set it to 0 for stage A.")
    text_encoder.TextEmbeddingProjector.forward = _forward_with_grad

    ckpt_dir = os.path.join(config.get("log_dir", "logs"), "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    seed_everything(config.get("seed", 42), workers=True)
    resume = get_latest_checkpoint(ckpt_dir)
    trainer = create_trainer(config, ckpt_dir)
    sa = config.get("stage_a", {})
    model = StageAModule(config)
    freeze_for_stage_a(model, bool(sa.get("train_projector", False)))
    print_param_info(model)

    tokenizer = AutoTokenizer.from_pretrained(config["paths"]["tokenizer_path"])
    assert len(tokenizer) == config["t2s_model"]["vocab_size"], (len(tokenizer), config["t2s_model"]["vocab_size"])
    mc = config["t2s_model"]
    dm = T2SDataModule(
        train_data_path=config["data"]["train_data_path"], val_data_path=config["data"].get("val_data_path"),
        tokenizer=tokenizer, w2v_bert_path=config["paths"]["w2v_bert_path"],
        batch_size=config["data"].get("batch_size", 16), num_workers=config["data"].get("num_workers", 4),
        max_text_seq_len=mc.get("max_text_seq_lens", 520), max_semantic_seq_len=mc.get("max_semantic_seq_lens", 1520),
        sample_rate=config["data"].get("sample_rate", 16000), semantic_pad_token=mc.get("stop_semantic_token", 8193),
        start_semantic_token=mc.get("start_semantic_token", 8192), stop_semantic_token=mc.get("stop_semantic_token", 8193),
    )
    trainer.fit(model=model, datamodule=dm, ckpt_path=resume)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    main(ap.parse_args())
