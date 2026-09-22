"""Build Confucius4-TTS fine-tuning data from an audio-pipeline ``s7_loudnorm`` folder.

Input : ``<workdir>/s7_loudnorm/manifest.jsonl`` produced by speech_dataset/audio-pipeline
        (one JSON record per segment: id, audio_path, speaker_id, multi_speaker, duration,
        text_normalized, cer, snr_db, dnsmos, clipping, ...).
Output: ``<out-dir>/train.tsv`` and ``<out-dir>/val.tsv`` in the 5-column, header-less TSV
        format that ``confuciustts/dataset/t2s_dataset.py`` and ``s2a_dataset.py`` read::

            lang \\t wav_path \\t norm_text \\t semantic_ids_path \\t ref_audio_paths

        plus ``<out-dir>/semantic/<id>.npy`` (semantic tokens), ``accepted.jsonl`` and
        ``summary.json``.

Pipeline
    1. read manifest, resolve audio paths, drop rows whose audio is missing
    2. filter with the pipeline's tier-A rule (defaults copied from
       speech_dataset/audio-pipeline/config/default.yaml)
    3. extract semantic tokens (resumable: existing .npy files are reused)
    4. pick reference clips per speaker (other utterances of the same speaker)
    5. split train / val per speaker and write the TSVs

Example::

    python scripts/prepare_confucius_data.py \\
        --s7-dir ../speech_dataset/audio-pipeline/work_5min/s7_loudnorm \\
        --out-dir data/vi_podcast_Confucius4_TTS --lang vi

    # only filter + statistics, no model download / token extraction
    python scripts/prepare_confucius_data.py --s7-dir ... --out-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Limits enforced by confuciustts/dataset/*.py
MAX_SEMANTIC_LEN = 1520          # tokens; ~30 s at 50 Hz
MIN_TARGET_SAMPLE_RATE = 22050   # s2a_dataset rejects lower sample rates
SEMANTIC_FRAME_RATE_HZ = 50.0

TSV_COLUMNS = ["lang", "wav_path", "norm_text", "semantic_ids_path", "ref_audio_paths"]

log = logging.getLogger("prepare_confucius_data")


# --------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s7-dir", required=True, help="path to <workdir>/s7_loudnorm")
    ap.add_argument("--out-dir", required=True, help="output folder, e.g. data/vi_podcast_Confucius4_TTS")
    ap.add_argument("--lang", default="vi", help="language code written to the TSV (must exist in LANGUAGE_TOKEN_MAP)")
    ap.add_argument(
        "--pipeline-root",
        default=None,
        help="folder that manifest audio_path is relative to (default: two levels above --s7-dir, i.e. audio-pipeline/)",
    )

    # tier-A rule, defaults from audio-pipeline/config/default.yaml
    g = ap.add_argument_group("quality filter (pipeline tier A defaults)")
    g.add_argument("--max-cer", type=float, default=0.02)
    g.add_argument("--min-snr-db", type=float, default=20.0)
    g.add_argument("--min-dnsmos", type=float, default=3.0, help="used instead of SNR when a dnsmos score exists")
    g.add_argument("--max-clipping", type=float, default=0.001)
    g.add_argument("--min-seconds", type=float, default=2.0)
    g.add_argument("--max-seconds", type=float, default=13.0)
    g.add_argument("--allow-multi-speaker", action="store_true")
    g.add_argument("--text-field", default="text_normalized", help="manifest field used as norm_text")

    r = ap.add_argument_group("reference audio")
    r.add_argument("--num-refs", type=int, default=3, help="reference clips per utterance (same speaker, excluding itself)")
    r.add_argument("--ref-min-seconds", type=float, default=3.0)
    r.add_argument("--ref-max-seconds", type=float, default=15.0)

    s = ap.add_argument_group("split")
    s.add_argument("--val-ratio", type=float, default=0.05)
    s.add_argument("--min-utts-per-speaker", type=int, default=2, help="speakers with fewer accepted utterances are dropped")
    s.add_argument("--seed", type=int, default=42)

    m = ap.add_argument_group("semantic tokens")
    m.add_argument("--w2v-bert-path", default="facebook/w2v-bert-2.0")
    m.add_argument("--stats-path", default=str(REPO_ROOT / "checkpoints" / "wav2vec2bert_stats.pt"))
    m.add_argument("--codec-local-path", default=None, help="local MaskGCT semantic_codec safetensors (skips HF download)")
    m.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    m.add_argument("--overwrite", action="store_true", help="re-extract tokens even if the .npy exists")

    ap.add_argument("--dry-run", action="store_true", help="filter and report only; no model loading, no TSV")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------- helpers
def clean_text(text: Optional[str]) -> str:
    """Collapse whitespace and drop characters that break the TSV reader.

    The dataset loader parses the TSV with ``datasets`` (pandas engine), which treats a
    leading ASCII double quote as a quoted field. Quotes are never spoken, so they are
    removed rather than escaped.
    """
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = text.replace('"', "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("skip malformed line %d in %s: %s", ln, path, e)
    return rows


def resolve_audio(row: dict, s7_dir: Path, pipeline_root: Path) -> Optional[Path]:
    """audio_path in the manifest is relative to the audio-pipeline root; fall back to s7_dir/audio/<id>.wav."""
    p = row.get("audio_path")
    candidates = []
    if p:
        pp = Path(p)
        candidates += [pp] if pp.is_absolute() else [pipeline_root / pp, s7_dir.parent / pp]
    candidates.append(s7_dir / "audio" / f"{row.get('id')}.wav")
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def tier_reject_reason(row: dict, a: argparse.Namespace, text: str) -> Optional[str]:
    """Return None if the row passes the tier rule, else a short reason string."""
    if not text:
        return "empty_text"
    cer = row.get("cer")
    if cer is None:
        return "cer_missing"
    if cer > a.max_cer:
        return "cer"
    dur = row.get("duration")
    if dur is None:
        return "duration_missing"
    if dur < a.min_seconds:
        return "too_short"
    if dur > a.max_seconds:
        return "too_long"
    clip = row.get("clipping")
    if clip is not None and clip > a.max_clipping:
        return "clipping"
    if row.get("multi_speaker") and not a.allow_multi_speaker:
        return "multi_speaker"
    dnsmos = row.get("dnsmos")
    if dnsmos is not None:
        if dnsmos < a.min_dnsmos:
            return "dnsmos"
    else:
        snr = row.get("snr_db")
        if snr is None or snr < a.min_snr_db:
            return "snr"
    sr = row.get("sr")
    if sr is not None and sr < MIN_TARGET_SAMPLE_RATE:
        return "sample_rate"
    if dur * SEMANTIC_FRAME_RATE_HZ > MAX_SEMANTIC_LEN:
        return "semantic_too_long"
    return None


def quality_score(row: dict) -> float:
    """Higher is better; used to rank reference-clip candidates."""
    cer = row.get("cer") or 0.0
    dnsmos = row.get("dnsmos")
    snr = row.get("snr_db") or 0.0
    acoustic = (dnsmos * 10.0) if dnsmos is not None else min(snr, 40.0)
    return acoustic - 100.0 * cer


def pick_refs(utt: dict, pool: List[dict], k: int) -> List[str]:
    cands = [r for r in pool if r["id"] != utt["id"]]
    cands.sort(key=quality_score, reverse=True)
    return [str(r["wav_abs"]) for r in cands[:k]]


def split_per_speaker(rows: List[dict], val_ratio: float, seed: int) -> Dict[str, List[dict]]:
    """Utterance-level split inside every speaker so each voice appears in both sets."""
    rng = random.Random(seed)
    by_spk: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    out = {"train": [], "val": []}
    for spk in sorted(by_spk):
        utts = sorted(by_spk[spk], key=lambda r: r["id"])
        rng.shuffle(utts)
        n = len(utts)
        if n >= 10:
            n_val = max(1, int(round(n * val_ratio)))
        elif n >= 4:
            n_val = 1
        else:
            n_val = 0
        out["val"] += utts[:n_val]
        out["train"] += utts[n_val:]
    out["train"].sort(key=lambda r: r["id"])
    out["val"].sort(key=lambda r: r["id"])
    return out


def write_tsv(path: Path, rows: List[dict], lang: str) -> None:
    """Header-less 5-column TSV, written as plain joined lines (no csv quoting/escaping)."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            fields = [lang, str(r["wav_abs"]), r["norm_text"], str(r["semantic_path"]), ",".join(r["ref_paths"])]
            for x in fields:
                if "\t" in x or "\n" in x or "\r" in x:
                    raise ValueError(f"tab/newline inside TSV field for {r['id']}")
            f.write("\t".join(fields) + "\n")


# --------------------------------------------------------------------------- main
def main() -> None:
    a = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    s7_dir = Path(a.s7_dir).expanduser().resolve()
    manifest_path = s7_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        sys.exit(f"manifest not found: {manifest_path}")
    pipeline_root = Path(a.pipeline_root).resolve() if a.pipeline_root else s7_dir.parent.parent
    out_dir = Path(a.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    sem_dir = out_dir / "semantic"

    # ---- 1. read + resolve
    rows = read_manifest(manifest_path)
    log.info("manifest %s: %d records", manifest_path, len(rows))
    reasons: Counter = Counter()
    accepted: List[dict] = []
    for r in rows:
        wav = resolve_audio(r, s7_dir, pipeline_root)
        if wav is None:
            reasons["audio_missing"] += 1
            continue
        text = clean_text(r.get(a.text_field) or r.get("text"))
        why = tier_reject_reason(r, a, text)
        if why:
            reasons[why] += 1
            continue
        if not r.get("speaker_id"):
            reasons["speaker_missing"] += 1
            continue
        rr = dict(r)
        rr["wav_abs"] = wav
        rr["norm_text"] = text
        accepted.append(rr)
    log.info("tier filter: %d accepted, rejected by reason: %s", len(accepted), dict(reasons))

    # ---- speakers with too few utterances cannot get reference clips
    by_spk: Dict[str, List[dict]] = defaultdict(list)
    for r in accepted:
        by_spk[r["speaker_id"]].append(r)
    keep_spk = {s for s, u in by_spk.items() if len(u) >= a.min_utts_per_speaker}
    dropped_spk = set(by_spk) - keep_spk
    if dropped_spk:
        n_drop = sum(len(by_spk[s]) for s in dropped_spk)
        reasons["speaker_too_few_utts"] += n_drop
        log.info("dropping %d speakers (<%d utts) = %d utterances", len(dropped_spk), a.min_utts_per_speaker, n_drop)
        accepted = [r for r in accepted if r["speaker_id"] in keep_spk]

    total_sec = sum(r["duration"] for r in accepted)
    log.info("after speaker filter: %d utts, %d speakers, %.2f h", len(accepted), len(keep_spk), total_sec / 3600)

    if a.dry_run:
        print(json.dumps({
            "manifest": str(manifest_path),
            "records": len(rows),
            "accepted": len(accepted),
            "speakers": len(keep_spk),
            "hours": round(total_sec / 3600, 3),
            "rejected": dict(reasons),
            "utts_per_speaker": {s: len([r for r in accepted if r["speaker_id"] == s]) for s in sorted(keep_spk)},
        }, indent=2, ensure_ascii=False))
        return
    if not accepted:
        sys.exit("nothing accepted; relax the thresholds or check the manifest")

    out_dir.mkdir(parents=True, exist_ok=True)
    sem_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. semantic tokens
    from semantic_tokenizer import SemanticTokenizer  # noqa: E402  (scripts/ on sys.path)

    tok: Optional[SemanticTokenizer] = None
    todo_ids = {r["id"] for r in accepted if a.overwrite or not (sem_dir / f"{r['id']}.npy").is_file()}
    log.info("semantic tokens: %d to extract, %d reused", len(todo_ids), len(accepted) - len(todo_ids))
    if todo_ids:
        tok = SemanticTokenizer(
            w2v_bert_path=a.w2v_bert_path,
            stats_path=a.stats_path,
            codec_local_path=a.codec_local_path,
            device=a.device,
        )
        log.info("tokenizer ready on %s", tok.device)
    t0 = time.time()
    tok_rates: List[float] = []
    still_ok: List[dict] = []
    for i, r in enumerate(accepted, 1):
        npy = sem_dir / f"{r['id']}.npy"
        try:
            if r["id"] in todo_ids:
                codes = tok.encode_file(r["wav_abs"])
                np.save(npy, codes.astype(np.int64))
            else:
                codes = np.load(npy)
            n = int(codes.shape[0])
            if n == 0 or n > MAX_SEMANTIC_LEN:
                reasons["semantic_len_out_of_range"] += 1
                log.warning("%s: %d tokens out of range, dropped", r["id"], n)
                continue
            r["semantic_path"] = npy.resolve()
            r["semantic_len"] = n
            tok_rates.append(n / r["duration"])
            still_ok.append(r)
        except Exception as e:  # keep going, report at the end
            reasons["semantic_extract_error"] += 1
            log.warning("%s: extraction failed: %s", r["id"], e)
        if i % 50 == 0 or i == len(accepted):
            log.info("  %d/%d done (%.1fs)", i, len(accepted), time.time() - t0)
    accepted = still_ok
    if tok_rates:
        med = float(np.median(tok_rates))
        log.info("tokens/sec median %.1f (expected ~%.0f)", med, SEMANTIC_FRAME_RATE_HZ)
        if abs(med - SEMANTIC_FRAME_RATE_HZ) / SEMANTIC_FRAME_RATE_HZ > 0.2:
            log.warning("token rate deviates >20%% from %.0f Hz; check audio sample rate / codec", SEMANTIC_FRAME_RATE_HZ)

    # ---- 3. reference clips
    by_spk = defaultdict(list)
    for r in accepted:
        by_spk[r["speaker_id"]].append(r)
    final: List[dict] = []
    for r in accepted:
        pool = [u for u in by_spk[r["speaker_id"]] if a.ref_min_seconds <= u["duration"] <= a.ref_max_seconds]
        refs = pick_refs(r, pool, a.num_refs)
        if not refs:  # fall back to any other utterance of the speaker
            refs = pick_refs(r, by_spk[r["speaker_id"]], a.num_refs)
        if not refs:
            reasons["no_reference"] += 1
            continue
        r["ref_paths"] = refs
        final.append(r)

    # ---- 4. split + write
    splits = split_per_speaker(final, a.val_ratio, a.seed)
    write_tsv(out_dir / "train.tsv", splits["train"], a.lang)
    write_tsv(out_dir / "val.tsv", splits["val"], a.lang)
    with (out_dir / "accepted.jsonl").open("w", encoding="utf-8") as f:
        for r in final:
            rec = {k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def hours(rs: List[dict]) -> float:
        return round(sum(x["duration"] for x in rs) / 3600, 3)

    summary = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "lang": a.lang,
        "records_in_manifest": len(rows),
        "accepted": len(final),
        "rejected": dict(reasons),
        "speakers": len({r["speaker_id"] for r in final}),
        "hours_total": hours(final),
        "train": {"utts": len(splits["train"]), "hours": hours(splits["train"])},
        "val": {"utts": len(splits["val"]), "hours": hours(splits["val"])},
        "utts_per_speaker": dict(Counter(r["speaker_id"] for r in final)),
        "semantic_tokens_per_sec_median": round(float(np.median(tok_rates)), 2) if tok_rates else None,
        "thresholds": {
            "max_cer": a.max_cer, "min_snr_db": a.min_snr_db, "min_dnsmos": a.min_dnsmos,
            "max_clipping": a.max_clipping, "min_seconds": a.min_seconds, "max_seconds": a.max_seconds,
            "allow_multi_speaker": a.allow_multi_speaker,
        },
        "num_refs": a.num_refs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%d rows) and %s (%d rows)", out_dir / "train.tsv", len(splits["train"]), out_dir / "val.tsv", len(splits["val"]))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
