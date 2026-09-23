"""Build Confucius4-TTS fine-tuning data from an audio-pipeline ``s8_package`` folder.

``s8`` is the packaging stage of speech_dataset/audio-pipeline: it assigns a quality
tier (A / B / C) to every segment, splits by speaker, and exports the kept tiers as a
self-contained dataset::

    s8_package/
    ├── dataset/
    │   ├── metadata.csv          file_name, text, text_normalized, speaker_id, duration,
    │   │                         tier, snr_db, dnsmos, cer, clipping, bandwidth_hz,
    │   │                         multi_speaker, source_path, split
    │   └── wav/{train,val,test}/*.wav
    └── manifest.jsonl            every segment incl. tier C (audio still under s7)

This script reads ``dataset/metadata.csv`` (so it works on a machine that only has the
packaged dataset), keeps the requested tiers, and then reuses the shared pipeline from
``prepare_confucius_data.py``: semantic-token extraction, reference-clip selection,
train/val split, TSV + summary output.

Differences from the s7 entry point
    * No per-metric thresholds: the tier already encodes them. Select with ``--tiers``.
    * ``--use-pipeline-split`` keeps s8's speaker-disjoint split (val/test -> val).
      The default re-splits *inside* each speaker, which is what a voice fine-tune wants.
    * ``--from-manifest`` reads ``manifest.jsonl`` instead (needs the s7 audio present);
      useful to pull tier B/C rows that were not exported.

Examples::

    python scripts/prepare_confucius_data_s8.py \\
        --s8-dir ../speech_dataset/audio-pipeline/work_5min/s8_package \\
        --out-dir data/vi_podcast_Confucius4_TTS --lang vi --dry-run

    python scripts/prepare_confucius_data_s8.py --s8-dir ... --out-dir ... \\
        --tiers A,B --device cuda
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_confucius_data import (  # noqa: E402
    MAX_SEMANTIC_LEN,
    MIN_TARGET_SAMPLE_RATE,
    SEMANTIC_FRAME_RATE_HZ,
    add_common_args,
    build_dataset,
    clean_text,
    drop_small_speakers,
    dry_run_report,
    read_manifest,
    resolve_out_dir,
    setup_logging,
)

log = logging.getLogger("prepare_confucius_data_s8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s8-dir", required=True, help="path to <workdir>/s8_package")
    ap.add_argument("--out-dir", required=True, help="output folder, e.g. data/vi_podcast_Confucius4_TTS")
    ap.add_argument("--tiers", default="A", help="comma-separated tiers to keep, e.g. A or A,B (default A)")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="drop longer clips (loader limit is ~30 s)")
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--text-field", default="text_normalized", help="metadata column used as norm_text")
    ap.add_argument("--use-pipeline-split", action="store_true", help="keep s8's split (train->train, val/test->val) instead of re-splitting per speaker")
    ap.add_argument("--from-manifest", action="store_true", help="read s8_package/manifest.jsonl (needs s7 audio) instead of dataset/metadata.csv")
    ap.add_argument("--pipeline-root", default=None, help="with --from-manifest: folder that audio_path is relative to (default: two levels above --s8-dir)")
    add_common_args(ap)
    return ap.parse_args()


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).strip()
    if v == "" or v.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def load_metadata_csv(s8_dir: Path, text_field: str) -> List[dict]:
    """Rows from dataset/metadata.csv normalised to the manifest-like schema build_dataset expects."""
    ds = s8_dir / "dataset"
    csv_path = ds / "metadata.csv"
    if not csv_path.is_file():
        sys.exit(f"metadata.csv not found: {csv_path}")
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            file_name = rec.get("file_name", "")
            wav = (ds / file_name)
            rows.append({
                "id": Path(file_name).stem,
                "wav_abs": wav.resolve(),
                "wav_exists": wav.is_file(),
                "speaker_id": rec.get("speaker_id") or None,
                "duration": _to_float(rec.get("duration")),
                "tier": (rec.get("tier") or "").strip().upper(),
                "split": (rec.get("split") or "train").strip().lower(),
                "cer": _to_float(rec.get("cer")),
                "snr_db": _to_float(rec.get("snr_db")),
                "dnsmos": _to_float(rec.get("dnsmos")),
                "clipping": _to_float(rec.get("clipping")),
                "multi_speaker": _to_bool(rec.get("multi_speaker")),
                "source_path": rec.get("source_path"),
                "text": rec.get("text"),
                "text_normalized": rec.get("text_normalized"),
                "norm_text": clean_text(rec.get(text_field) or rec.get("text")),
            })
    return rows


def load_manifest(s8_dir: Path, pipeline_root: Path, text_field: str) -> List[dict]:
    """Rows from s8_package/manifest.jsonl; audio still lives under s7_loudnorm."""
    path = s8_dir / "manifest.jsonl"
    if not path.is_file():
        sys.exit(f"manifest not found: {path}")
    rows = []
    for r in read_manifest(path):
        p = Path(r.get("audio_path", ""))
        cands = [p] if p.is_absolute() else [pipeline_root / p, s8_dir.parent / p]
        cands.append(s8_dir.parent / "s7_loudnorm" / "audio" / f"{r.get('id')}.wav")
        wav = next((c for c in cands if c.is_file()), None)
        rr = dict(r)
        rr["wav_abs"] = wav.resolve() if wav else None
        rr["wav_exists"] = wav is not None
        rr["tier"] = (r.get("tier") or "").upper()
        rr["split"] = (r.get("split") or "train").lower()
        rr["norm_text"] = clean_text(r.get(text_field) or r.get("text"))
        rows.append(rr)
    return rows


def main() -> None:
    a = parse_args()
    setup_logging(a.verbose)
    s8_dir = Path(a.s8_dir).expanduser().resolve()
    out_dir = resolve_out_dir(a.out_dir)
    keep_tiers = {t.strip().upper() for t in a.tiers.split(",") if t.strip()}
    if "C" in keep_tiers:
        log.warning("tier C is the pipeline's reject tier; including it is not recommended")

    if a.from_manifest:
        pipeline_root = Path(a.pipeline_root).resolve() if a.pipeline_root else s8_dir.parent.parent
        rows = load_manifest(s8_dir, pipeline_root, a.text_field)
        source = str(s8_dir / "manifest.jsonl")
    else:
        rows = load_metadata_csv(s8_dir, a.text_field)
        source = str(s8_dir / "dataset" / "metadata.csv")
    log.info("%s: %d records, tiers %s", source, len(rows), dict(Counter(r["tier"] for r in rows)))

    reasons: Counter = Counter()
    accepted: List[dict] = []
    for r in rows:
        if r["tier"] not in keep_tiers:
            reasons[f"tier_{r['tier'] or 'missing'}"] += 1
            continue
        if not r["wav_exists"]:
            reasons["audio_missing"] += 1
            continue
        if not r["norm_text"]:
            reasons["empty_text"] += 1
            continue
        if not r["speaker_id"]:
            reasons["speaker_missing"] += 1
            continue
        dur = r["duration"]
        if dur is None:
            reasons["duration_missing"] += 1
            continue
        if dur < a.min_seconds:
            reasons["too_short"] += 1
            continue
        if dur > a.max_seconds or dur * SEMANTIC_FRAME_RATE_HZ > MAX_SEMANTIC_LEN:
            reasons["too_long"] += 1
            continue
        sr = r.get("sr")
        if sr is not None and sr < MIN_TARGET_SAMPLE_RATE:
            reasons["sample_rate"] += 1
            continue
        accepted.append(r)
    log.info("tier filter (%s): %d accepted, rejected: %s", ",".join(sorted(keep_tiers)), len(accepted), dict(reasons))

    accepted = drop_small_speakers(accepted, a.min_utts_per_speaker, reasons)
    log.info("after speaker filter: %d utts, %d speakers, %.2f h",
             len(accepted), len({r["speaker_id"] for r in accepted}), sum(r["duration"] for r in accepted) / 3600)

    if a.dry_run:
        dry_run_report(source, len(rows), accepted, reasons)
        return

    build_dataset(
        accepted, out_dir, a, reasons,
        summary_extra={
            "source": source,
            "records_in_source": len(rows),
            "tiers_kept": sorted(keep_tiers),
            "max_seconds": a.max_seconds,
        },
        use_split_field=a.use_pipeline_split,
    )


if __name__ == "__main__":
    main()
