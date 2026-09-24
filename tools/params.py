"""
Knobs: `# @param` annotations on plain Python assignments (Colab-style).

    learning_rate = 0.01  # @param {"min": 0.0001, "max": 0.1, "log": true} step size of the optimizer

The line stays valid Python, so `python code.py` still works. The site turns every
annotated line into a slider / dropdown / checkbox and rewrites the literal before a run.
site/params.js is the JavaScript twin of this file; keep the two in sync.
"""

import ast
import json
import re

PARAM_RE = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>.+?)\s*#\s*@param\s*(?P<opts>\{[^}]*\})?\s*(?P<help>.*)$"
)


def parse(code):
    """Return a list of knobs: {name, line, value, type, help, **options}."""
    knobs = []
    for i, line in enumerate(code.splitlines()):
        m = PARAM_RE.match(line)
        if not m:
            continue
        value = ast.literal_eval(m["value"])
        kind = {bool: "bool", int: "int", float: "float", str: "str"}.get(type(value))
        if kind is None:
            raise ValueError(f"line {i + 1}: @param value must be a bool/int/float/str literal, got {m['value']!r}")
        opts = json.loads(m["opts"]) if m["opts"] else {}
        knobs.append({"name": m["name"], "line": i, "value": value, "type": kind, "help": m["help"].strip(), **opts})
    return knobs


def to_literal(value):
    return repr(value)  # True/False, 0.01, 'adam' are all valid Python literals


def apply(code, overrides):
    """Rewrite the literal of each annotated assignment named in `overrides`."""
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        m = PARAM_RE.match(line.rstrip("\n"))
        if m and m["name"] in overrides:
            start, end = m.span("value")
            lines[i] = line[:start] + to_literal(overrides[m["name"]]) + line[end:]
    return "".join(lines)
