#!/usr/bin/env python3
"""
Heal generation records whose audio_path UUIDs don't match the files on disk.

Matches orphaned .wav files to broken DB rows using duration + created_at proximity.
Prints a dry-run plan by default; pass --apply to commit changes.
"""

import argparse
import sqlite3
import struct
import sys
import wave
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path.home() / "Library/Application Support/sh.voicebox.app"
GENERATIONS_DIR = DATA_DIR / "generations"
DB_PATH = DATA_DIR / "voicebox.db"

# How close created_at must be to file mtime to count as a match (seconds)
TIME_WINDOW_SECONDS = 120

# How close durations must be to count as a match (seconds)
DURATION_TOLERANCE_SECONDS = 0.5


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write fixes to DB")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    generations_dir = args.data_dir / "generations"
    db_path = args.data_dir / "voicebox.db"

    if not db_path.exists():
        sys.exit(f"DB not found: {db_path}")
    if not generations_dir.exists():
        sys.exit(f"Generations dir not found: {generations_dir}")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    # Rows whose audio_path file is missing
    broken = con.execute("""
        SELECT id, audio_path, duration, created_at
        FROM generations
        WHERE deleted_at IS NULL
    """).fetchall()

    broken_rows = [r for r in broken if not Path(r["audio_path"]).exists()]
    print(f"Broken rows (missing file): {len(broken_rows)}")

    if not broken_rows:
        print("Nothing to heal.")
        return

    # UUIDs already referenced in the DB
    known_uuids = {
        Path(r["audio_path"]).stem
        for r in con.execute("SELECT audio_path FROM generations").fetchall()
    }

    # .wav files on disk with no matching DB row
    all_wav = {p.stem: p for p in generations_dir.glob("*.wav")}
    orphan_wavs = {stem: path for stem, path in all_wav.items() if stem not in known_uuids}
    print(f"Orphaned .wav files (no DB row): {len(orphan_wavs)}")

    if not orphan_wavs:
        print("No orphaned files to match against.")
        return

    # Read durations + mtimes for orphans
    orphan_info = {}
    for stem, path in orphan_wavs.items():
        try:
            dur = wav_duration(path)
            mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
            orphan_info[stem] = {"path": path, "duration": dur, "mtime": mtime}
        except Exception as e:
            print(f"  skip {stem}: {e}")

    fixes = []
    unmatched = []

    for row in broken_rows:
        db_dur = row["duration"]
        db_time = datetime.fromisoformat(row["created_at"])
        candidates = []

        for stem, info in orphan_info.items():
            dur_diff = abs(info["duration"] - db_dur)
            time_diff = abs((info["mtime"] - db_time).total_seconds())
            if dur_diff <= DURATION_TOLERANCE_SECONDS and time_diff <= TIME_WINDOW_SECONDS:
                candidates.append((dur_diff + time_diff / 1000, stem, info))

        if not candidates:
            unmatched.append(row)
            continue

        candidates.sort(key=lambda x: x[0])
        _, best_stem, best_info = candidates[0]
        new_path = str(best_info["path"])

        fixes.append({
            "old_id": row["id"],
            "new_id": best_stem,
            "old_path": row["audio_path"],
            "new_path": new_path,
            "duration": db_dur,
            "created_at": row["created_at"],
            "file_mtime": best_info["mtime"].isoformat(),
        })

        # Remove from pool so it can't match twice
        del orphan_info[best_stem]

    print(f"\nMatches found: {len(fixes)}")
    for f in fixes:
        print(f"  {f['old_id'][:8]}... -> {f['new_id'][:8]}...")
        print(f"    duration={f['duration']:.2f}s  created_at={f['created_at']}  mtime={f['file_mtime']}")
        print(f"    path: {f['old_path']}")
        print(f"       -> {f['new_path']}")

    if unmatched:
        print(f"\nUnmatched broken rows: {len(unmatched)}")
        for r in unmatched:
            print(f"  {r['id']}  dur={r['duration']:.2f}s  created={r['created_at']}")
            print(f"    missing: {r['audio_path']}")

    if not fixes:
        return

    if not args.apply:
        print("\nDry run — pass --apply to write changes.")
        return

    print("\nApplying fixes...")
    for f in fixes:
        # Update generation_jobs.generation_id references first
        con.execute(
            "UPDATE generation_jobs SET generation_id = ? WHERE generation_id = ?",
            (f["new_id"], f["old_id"]),
        )
        # Update story_items references
        con.execute(
            "UPDATE story_items SET generation_id = ? WHERE generation_id = ?",
            (f["new_id"], f["old_id"]),
        )
        # Update the generation row itself (PK + audio_path)
        con.execute(
            "UPDATE generations SET id = ?, audio_path = ? WHERE id = ?",
            (f["new_id"], f["new_path"], f["old_id"]),
        )
        print(f"  fixed {f['old_id'][:8]}... -> {f['new_id'][:8]}...")

    con.commit()
    print(f"Done. {len(fixes)} row(s) healed.")


if __name__ == "__main__":
    main()
