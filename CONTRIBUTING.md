# Adding or updating a lesson

A lesson is a directory under `course/<part>/`, named `NN-slug`. The number sets its position in the sidebar.

```
course/part2-numpy/28-sliding-window/
  code.py       the FULL program after this lesson: runnable on its own
  lesson.md     the explanation (Markdown)
  lesson.toml   metadata
```

## 1. Scaffold

```bash
python scripts/new_lesson.py 28-sliding-window --parent 21-numpy-gpt --title "Sliding-window attention"
```

This copies the parent's `code.py`, so the diff starts out empty. **Whatever you change in `code.py` is the lesson.**

## 2. Write the code: the diff *is* the lesson

- Change only what the concept needs. Readers study the diff line by line, so don't reformat, rename or reorder unrelated code.
- Match the surrounding style: short, commented, microgpt-flavored.
- Expose interesting hyperparameters as knobs by tagging a single-literal assignment:
  ```python
  window = 8  # @param {"min": 1, "max": 16} how far back each position may attend
  ```
  Options: `min`, `max`, `step`, `log: true` (log-scale slider), `options: [...]` (dropdown). Values must be `int`, `float`, `str` or `bool` literals.
- Training lessons must have a `num_steps` knob and print `step N / TOTAL | loss X` lines (add any other `| name value` pairs to chart them). Keep the defaults quick in the browser: Part 2 runs about 300 steps in seconds.

## 3. Write `lesson.toml`

```toml
title = "Sliding-window attention"
parent = "21-numpy-gpt"       # the diff is computed against this lesson
variant = true                # side branch (indented in the sidebar); omit for main-line lessons
summary = "One sentence."
tags = ["attention", "long context"]
updated = 2026-09-24          # bump when you revise the lesson

[[references]]
title = "Longformer (Beltagy et al., 2020)"
url = "https://arxiv.org/abs/2004.05150"

[[experiments]]               # one-click knob presets shown in the lab
name = "Tiny window"
description = "window = 1: every position only sees itself."
params = { window = 1 }
```

## 4. Write `lesson.md`

A few short sections: the problem, the idea, a pointer to the key lines of the diff, and what to try. Mention where the idea is used today and link the paper in `references`.

## 5. Check

```bash
pytest -q                                   # runs your lesson + every experiment with tiny step counts
python tools/build.py && python -m http.server -d dist 8000
```

## Keeping it up to date

- When a technique is superseded or refined, update its lesson (and `updated`), or add a new branch next to it. Branches are cheap.
- If you change a parent lesson, every child's diff changes too. Re-read the children's diffs and keep them minimal.
- Track ideas in [ROADMAP.md](ROADMAP.md).
