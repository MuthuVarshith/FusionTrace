"""Build a group-disjoint train/val/test manifest from real and fake folders.

    python scripts/make_manifest.py --real data/train/real --fake data/train/fake \\
        --out data/manifest.csv --fractions 0.8,0.1,0.1 --group-by stem

``--group-by`` decides what must stay on one side of a split:

* ``stem``   each file is its own group. Use only when the images are known to be
             independent (no shared person or source video). Warns, because
             face-swap sets usually are not.
* ``parent`` each immediate subfolder is a group. Use when each person or source
             video has its own folder, which is the normal layout for identity splits.
* ``regex``  the first capture group of ``--group-regex`` applied to the file name,
             e.g. ``--group-regex '^(\\d+)_'`` if the identity id is a filename prefix.

The manifest is checked for leakage (no group in two splits) before it is written.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from finetune import data  # noqa: E402


def make_group_fn(mode: str, regex: str | None):
    if mode == "stem":
        return lambda p: p.stem
    if mode == "parent":
        def group(path: Path) -> str:
            # Images directly in real/ or fake/ have no person or video folder: fall back to per-file groups.
            return path.stem if path.parent.name in ("real", "fake") else path.parent.name
        return group
    if mode == "regex":
        if not regex:
            raise SystemExit("--group-regex is required with --group-by regex")
        pattern = re.compile(regex)

        def group(path: Path) -> str:
            match = pattern.search(path.name)
            if not match:
                raise SystemExit(f"--group-regex did not match {path.name!r}")
            return match.group(1)

        return group
    raise SystemExit(f"unknown --group-by {mode!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--fake", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fractions", default="0.8,0.1,0.1", help="train,val,test share of groups")
    parser.add_argument("--group-by", choices=("stem", "parent", "regex"), default="parent")
    parser.add_argument("--group-regex", default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fractions = tuple(float(x) for x in args.fractions.split(","))
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise SystemExit("--fractions must be three numbers that sum to 1")
    if args.group_by == "stem":
        print("warning: --group-by stem treats every image as independent; identities or source videos may leak across splits")
    group_of = make_group_fn(args.group_by, args.group_regex)
    if args.group_by == "parent":
        print("note: flat real/ and fake/ folders get one group per file; use subfolders per person or video for identity splits")
    rows = data.group_split_from_folders(args.real, args.fake, seed=args.seed, fractions=fractions, group_of=group_of)
    data.check_disjoint(rows)
    data.check_splits_present(rows, splits=data.SPLITS)
    data.write_manifest(rows, args.out)
    counts = {s: (sum(1 for r in rows if r.split == s and r.label == 0), sum(1 for r in rows if r.split == s and r.label == 1)) for s in data.SPLITS}
    print(f"wrote {args.out}: " + ", ".join(f"{s} real/fake={real}/{fake}" for s, (real, fake) in counts.items()))
    print(f"data_version={data.content_hash(rows)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
