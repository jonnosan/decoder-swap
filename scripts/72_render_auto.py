"""Render auto-extracted MIDI for one or more tracks via FluidSynth.

Reads data/song_test/<slug>/semantic/bass_auto.mid + bass_auto.json,
renders with chosen program, saves 30s slices to results/auto_extract/.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SOUNDFONT = Path.home() / ".cache" / "slackbeatz" / "FluidR3_GM.sf2"


def render_one(slug: str, program: int = 38) -> dict | None:
    song_dir = REPO_ROOT / "data" / "song_test" / slug
    midi_path = song_dir / "semantic" / "bass_auto.mid"
    meta_path = song_dir / "semantic" / "bass_auto.json"
    if not midi_path.exists() or not meta_path.exists():
        return {"slug": slug, "error": "no auto extraction"}
    meta = json.loads(meta_path.read_text())

    # Re-set program in MIDI
    import mido
    mid = mido.MidiFile(str(midi_path))
    for track in mid.tracks:
        for msg in track:
            if msg.type == "program_change":
                msg.program = program
    tmp_midi = Path(f"/tmp/render_{slug}.mid")
    mid.save(str(tmp_midi))

    # FluidSynth render
    tmp_wav = Path(f"/tmp/render_{slug}.wav")
    r = subprocess.run(
        ["fluidsynth", "-F", str(tmp_wav), "-r", "44100", "-g", "1.0",
         str(SOUNDFONT), str(tmp_midi)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return {"slug": slug, "error": f"fluidsynth failed: {r.stderr}"}

    # Slice 30s of the dense region
    out_dir = REPO_ROOT / "results" / "auto_extract"
    out_dir.mkdir(parents=True, exist_ok=True)
    region_start = float(meta.get("region_start", 0))

    x, sr = sf.read(str(tmp_wav))
    if x.ndim > 1:
        x = x.mean(-1)
    slc = x[int(region_start * sr): int((region_start + 30) * sr)]
    out_render = out_dir / f"{slug}_auto30.wav"
    sf.write(str(out_render), slc, sr, subtype="PCM_16")

    # Also slice the original bass at the same region for reference
    orig_path = song_dir / "stems" / "bass.wav"
    x2, sr = sf.read(str(orig_path))
    if x2.ndim > 1:
        x2 = x2.mean(-1)
    slc2 = x2[int(region_start * sr): int((region_start + 30) * sr)]
    out_orig = out_dir / f"{slug}_original30.wav"
    sf.write(str(out_orig), slc2, sr, subtype="PCM_16")

    return {
        "slug": slug,
        "score": meta.get("score"),
        "render": str(out_render),
        "original": str(out_orig),
        "render_rms": float(np.sqrt(np.mean(slc ** 2))),
        "orig_rms": float(np.sqrt(np.mean(slc2 ** 2))),
        "program": program,
        "params": {
            "bpm": meta.get("bpm"),
            "anchor_used": meta.get("anchor_used"),
            "pre_merge_ms": meta.get("pre_merge_ms"),
            "offset_steps": meta.get("anchor_offset_steps"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", help="single track")
    ap.add_argument("--slug-pattern", help="glob")
    ap.add_argument("--program", type=int, default=38)
    args = ap.parse_args()

    slugs: list[str] = []
    if args.slug:
        slugs = [args.slug]
    elif args.slug_pattern:
        slugs = sorted([Path(p).name for p in glob.glob(str(REPO_ROOT / "data" / "song_test" / args.slug_pattern))])
    else:
        print("--slug or --slug-pattern", file=sys.stderr)
        return 1

    results = []
    for slug in slugs:
        r = render_one(slug, program=args.program)
        if r is None:
            continue
        results.append(r)
        if "error" not in r:
            print(f"  {slug}: score={r['score']:.3f} render_rms={r['render_rms']:.3f}")
        else:
            print(f"  {slug}: {r['error']}")

    summary_path = REPO_ROOT / "results" / "auto_extract" / "render_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
