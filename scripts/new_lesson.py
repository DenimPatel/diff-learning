"""
Scaffold a new lesson that branches off an existing one.

    python scripts/new_lesson.py 28-sliding-window --parent 21-numpy-gpt --title "Sliding-window attention"

Copies the parent's code.py (so the diff starts empty) and writes stub lesson.toml / lesson.md.
Then edit code.py: whatever you change *is* the lesson's diff.
"""

import argparse
import datetime
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from course import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", help="directory name, e.g. 28-sliding-window (the number sets the order)")
    ap.add_argument("--parent", required=True, help="id of the lesson this one builds on")
    ap.add_argument("--title", required=True)
    ap.add_argument("--part", help="part directory (default: the parent's part)")
    ap.add_argument("--spine", action="store_true", help="a main-line lesson rather than a side branch")
    args = ap.parse_args()

    lessons = {l["id"]: l for l in load()["lessons"]}
    if args.parent not in lessons:
        sys.exit(f"unknown parent {args.parent!r}; choose from: {', '.join(lessons)}")
    if args.id in lessons:
        sys.exit(f"lesson {args.id!r} already exists")
    parent = lessons[args.parent]
    part = args.part or parent["part"]
    d = ROOT / "course" / part / args.id
    d.mkdir(parents=True)
    shutil.copy(ROOT / parent["path"] / "code.py", d / "code.py")
    (d / "lesson.toml").write_text(f'''title = "{args.title}"
parent = "{args.parent}"
{"" if args.spine else "variant = true" + chr(10)}summary = "One sentence: what this lesson adds and why it matters."
tags = []
updated = {datetime.date.today().isoformat()}

[[references]]
title = "The paper or post that introduced the idea"
url = "https://arxiv.org/abs/..."

[[experiments]]
name = "A knob setting worth trying"
description = "What the learner should notice."
params = {{}}
''')
    (d / "lesson.md").write_text(f"Explain the idea in a few short paragraphs, then point at the diff.\n")
    print(f"created {d.relative_to(ROOT)}: edit code.py, lesson.md and lesson.toml, then run `pytest -q`")


if __name__ == "__main__":
    main()
