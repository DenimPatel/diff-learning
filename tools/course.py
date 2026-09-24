"""
Load and validate the course: course/<part>/<lesson>/{lesson.toml, lesson.md, code.py}.

Each lesson stores the *full* code snapshot after that lesson and names its `parent`.
Diffs are never stored: they are always parent/code.py -> lesson/code.py, computed by the site.
"""

import tomllib
from pathlib import Path

from params import parse

ROOT = Path(__file__).resolve().parent.parent
COURSE = ROOT / "course"


def load(course_dir=COURSE):
    parts, lessons = [], []
    for part_dir in sorted(p for p in course_dir.iterdir() if p.is_dir()):
        part = tomllib.loads((part_dir / "part.toml").read_text())
        part["id"] = part_dir.name
        parts.append(part)
        for d in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            meta = tomllib.loads((d / "lesson.toml").read_text())
            code = (d / "code.py").read_text()
            lessons.append({
                "id": d.name,
                "part": part_dir.name,
                "title": meta["title"],
                "parent": meta.get("parent"),
                "variant": meta.get("variant", False),
                "summary": meta.get("summary", ""),
                "tags": meta.get("tags", []),
                "updated": str(meta.get("updated", "")),
                "references": meta.get("references", []),
                "experiments": meta.get("experiments", []),
                "md": (d / "lesson.md").read_text(),
                "code": code,
                "params": parse(code),
                "path": str(d.relative_to(ROOT)),
            })
    validate(lessons)
    return {"parts": parts, "lessons": lessons}


def validate(lessons):
    ids = [l["id"] for l in lessons]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate lesson ids: {dupes}"
    by_id = {l["id"]: l for l in lessons}
    for l in lessons:
        assert l["parent"] is None or l["parent"] in by_id, f"{l['id']}: unknown parent {l['parent']!r}"
        names = {p["name"] for p in l["params"]}
        for ex in l["experiments"]:
            unknown = set(ex.get("params", {})) - names
            assert not unknown, f"{l['id']}: experiment {ex['name']!r} sets unknown knobs {unknown}"
        # walk up to the root to rule out cycles
        seen, cur = set(), l
        while cur["parent"]:
            assert cur["id"] not in seen, f"cycle through {cur['id']}"
            seen.add(cur["id"])
            cur = by_id[cur["parent"]]
