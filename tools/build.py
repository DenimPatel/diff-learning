"""
Build the static site: copy site/ to dist/ and bundle the course into dist/course.json.

    python tools/build.py && python -m http.server -d dist
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course import ROOT, load  # noqa: E402


def build(out=ROOT / "dist"):
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out)
    course = load()
    (out / "course.json").write_text(json.dumps(course))
    (out / ".nojekyll").touch()
    return course


if __name__ == "__main__":
    course = build()
    print(f"built dist/ with {len(course['lessons'])} lessons in {len(course['parts'])} parts")
