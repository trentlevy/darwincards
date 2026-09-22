#!/usr/bin/env python3
"""
Ingest a school's "shared drive" notes corpus into DarwinCards' reference
index — the prior years' student notes on lectures that get recycled
semester after semester. Once ingested, card generation for that school
automatically pulls in matching excerpts (see notes_corpus.py / pipeline.py).

Usage:
    python ingest_notes.py path/to/notes.docx --school "Perelman School of Medicine"
    python ingest_notes.py path/to/folder      --school "Perelman School of Medicine"

Accepts .docx, .txt, .pdf, or .pptx. A .docx with Word "Heading" styles is
split into per-topic sections automatically — everything else is chunked by
word count. Re-running on an updated export of the same file replaces its
old chunks rather than duplicating them (keyed by filename).

Run this once against the full combined doc/drive, and again any time the
corpus gets an update.
"""

import argparse
import sys
from pathlib import Path

from database import Base, engine
from notes_corpus import ingest_notes_file, corpus_stats, SUPPORTED_NOTE_EXTS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Notes file, or a folder to ingest recursively")
    parser.add_argument("--school", required=True, help='e.g. "Perelman School of Medicine"')
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)  # ensure note_chunks table exists

    path = Path(args.path)
    if not path.exists():
        print(f"No such file or folder: {path}", file=sys.stderr)
        sys.exit(1)

    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_NOTE_EXTS)

    if not files:
        print(f"No supported files found under {path} (.docx, .txt, .pdf, .pptx).")
        sys.exit(1)

    total = 0
    for f in files:
        try:
            n = ingest_notes_file(str(f), school=args.school)
            print(f"  {f.name}: {n} chunk(s)")
            total += n
        except Exception as e:
            print(f"  {f.name}: FAILED ({e})", file=sys.stderr)

    print(f"\nDone. {total} chunk(s) ingested for '{args.school}' from {len(files)} file(s).")
    stats = corpus_stats(args.school)
    print(f"Corpus now has {stats['chunk_count']} total chunk(s) from {len(stats['sources'])} source(s):")
    for s in stats["sources"]:
        print(f"  - {s}")


if __name__ == "__main__":
    main()
