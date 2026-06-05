"""Pre-process the Mayday compilation (1B.2 corpus build).

For each track in the compilation:
  1. Demucs separation (data/song_test/<slug>/stems/)
  2. basic-pitch bass→MIDI (data/song_test/<slug>/semantic/bass.json)
  3. DAC cache of the bass stem (data/song_test/<slug>/stems_dac_tokens/bass.npy)

All stages skip-if-exists. Re-runs are cheap.

Slug convention: `mayday_<NN>_<short_artist_title>` (e.g. mayday_01_westbam_anthem).

Run:
  .venv/bin/python scripts/55_build_mayday_corpus.py
  .venv/bin/python scripts/55_build_mayday_corpus.py --only mayday_01_westbam_anthem  # one track
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SCRIPTS = REPO_ROOT / "scripts"
MAYDAY_DIR = Path("/Users/jonno/Downloads/mayday.a.new.chapter.of.house.and.techno.92")


def slugify(filename: str) -> str:
    """e.g. '1-03 Aphex Twin - Metapharstic.mp3' -> 'mayday_d1t03_aphex_twin_metapharstic'."""
    m = re.match(r"(\d+)-(\d+)\s+(.*)\.(mp3|wav|flac)$", filename, re.IGNORECASE)
    if not m:
        raise ValueError(f"unexpected filename: {filename}")
    disc, track, rest, _ = m.groups()
    rest = re.sub(r"[^A-Za-z0-9]+", "_", rest).strip("_").lower()
    return f"mayday_d{int(disc)}t{int(track):02d}_{rest}"


def run(label: str, cmd: list[str], skip_if: bool) -> float:
    print(f"  ─ {label}", flush=True)
    if skip_if:
        print(f"    [skip] already done", flush=True)
        return 0.0
    t0 = time.time()
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "ok" if rc == 0 else f"FAIL rc={rc}"
    print(f"    {dt:>6.1f}s  {status}", flush=True)
    if rc != 0:
        raise SystemExit(f"{label} failed")
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="only process this slug (must match the slug we'd assign)")
    ap.add_argument("--root", default=str(REPO_ROOT / "data" / "song_test"))
    args = ap.parse_args()

    if not MAYDAY_DIR.exists():
        print(f"ERROR: {MAYDAY_DIR} not found")
        return 1
    files = sorted(p for p in MAYDAY_DIR.iterdir()
                   if p.suffix.lower() in {".mp3", ".wav", ".flac"})
    print(f"# Mayday corpus build · {len(files)} tracks in {MAYDAY_DIR.name}")
    print()

    total_t0 = time.time()
    timings: dict[str, float] = {}
    skipped_tracks: list[str] = []
    for f in files:
        try:
            slug = slugify(f.name)
        except ValueError as e:
            print(f"[skip] {f.name}: {e}")
            continue
        if args.only and slug != args.only:
            continue

        print(f"# {slug}  ←  {f.name}")
        song_dir = Path(args.root) / slug

        # 30 Demucs
        stems_done = all((song_dir / "stems" / s).exists()
                         for s in ("drums.wav", "bass.wav", "other.wav", "vocals.wav"))
        t1 = run("30 demucs", [PY, str(SCRIPTS / "30_separate_stems.py"),
                                "--in", str(f), "--slug", slug],
                 skip_if=stems_done)

        # 33 bass → MIDI
        midi_done = (song_dir / "semantic" / "bass.json").exists()
        t2 = run("33 bass→MIDI", [PY, str(SCRIPTS / "33_bass_to_midi.py"),
                                   "--slug", slug],
                 skip_if=midi_done)

        # 50 DAC cache
        dac_done = (song_dir / "stems_dac_tokens" / "bass.npy").exists()
        t3 = run("50 DAC cache", [PY, str(SCRIPTS / "50_cache_bass_dac.py"),
                                    "--slug", slug],
                 skip_if=dac_done)

        timings[slug] = t1 + t2 + t3
        print()

    total = time.time() - total_t0
    print(f"## corpus build complete in {total:.1f}s ({total/60:.1f} min)")
    print(f"   {len(timings)} tracks processed")
    if skipped_tracks:
        print(f"   skipped: {skipped_tracks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
