"""
Course checks: the lesson tree is valid, knobs parse, and every lesson's code actually runs.
Run with:  pytest -q
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import params  # noqa: E402
from build import build  # noqa: E402
from course import load  # noqa: E402

COURSE = load()
LESSONS = COURSE["lessons"]
QUICK = {"num_steps": 3, "num_samples": 2}  # enough to exercise every code path
METRIC_LINE = re.compile(r"^step\s+\d+.*\|\s*loss\s+-?[\d.]+", re.M)


def run_lesson(code, tmp_path):
    shutil.copy(ROOT / "site" / "data" / "names.txt", tmp_path / "input.txt")  # no network needed
    script = tmp_path / "code.py"
    script.write_text(code)
    return subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=300)


def test_tree():
    ids = {l["id"] for l in LESSONS}
    roots = [l for l in LESSONS if l["parent"] is None]
    assert len(roots) == 1, "the course has exactly one root: the empty file"
    for l in LESSONS:
        assert l["title"] and l["md"].strip(), f"{l['id']} needs a title and a lesson.md"
        assert l["parent"] is None or l["parent"] in ids


def test_params_roundtrip():
    code = 'lr = 0.01  # @param {"min": 0.001, "max": 1, "log": true} step size\nname = \'adam\' # @param {"options": ["sgd", "adam"]}\nx = 3\n'
    knobs = params.parse(code)
    assert [(k["name"], k["value"], k["type"]) for k in knobs] == [("lr", 0.01, "float"), ("name", "adam", "str")]
    assert knobs[0]["log"] is True and knobs[0]["help"] == "step size"
    new = params.apply(code, {"lr": 0.5, "name": "sgd"})
    assert [(k["name"], k["value"]) for k in params.parse(new)] == [("lr", 0.5), ("name", "sgd")]
    assert new.splitlines()[2] == "x = 3"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_params_match_python():
    js = f"""
const P = require({json.dumps(str(ROOT / 'site' / 'params.js'))});
const lessons = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = {{}};
for (const l of lessons) {{
  const k = P.parse(l.code);
  const applied = P.parse(P.apply(l.code, Object.fromEntries(k.map(x => [x.name, x.value]))));
  out[l.id] = [k, applied];
}}
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", js], input=json.dumps(LESSONS), capture_output=True, text=True, check=True)
    out = json.loads(res.stdout)
    for l in LESSONS:
        js_knobs, js_applied = out[l["id"]]
        assert js_knobs == l["params"], l["id"]
        assert [k["value"] for k in js_applied] == [k["value"] for k in l["params"]], l["id"]


DIFF_JS = r"""
const D = require(DIFF_JS_PATH);
const out = {};
// splitting highlighted HTML keeps spans balanced on every line
out.split = D.splitLines('<span class="s">"a\nb"</span>\nx');
// removed lines sit next to the added lines that replaced them
out.pairs = D.pairRows([{type:'ctx',n:0},{type:'del',n:1},{type:'del',n:2},{type:'add',n:3},{type:'ctx',n:4},{type:'add',n:5}])
  .map(p => [p.type, p.left && p.left.n, p.right && p.right.n]);
out.number = D.wordRanges('lr = 0.01  # step', 'lr = 3e-4  # step');
out.appended = D.wordRanges('def gpt(token_id, pos_id):', 'def gpt(token_id, pos_id, keys, values):');
out.rewritten = D.wordRanges('x = foo(a, b)', 'return None');
out.entity = D.markRanges('<span class="k">if</span> a &lt; b:', [[3, 8]]);
out.across = D.markRanges('x = <span class="s">"ab"</span> + 1', [[2, 10]]);
out.none = D.markRanges('<b>x</b>', []);
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_diff_helpers():
    js = DIFF_JS.replace("DIFF_JS_PATH", json.dumps(str(ROOT / "site" / "diff.js")))
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out["split"] == ['<span class="s">"a</span>', '<span class="s">b"</span>', "x"]
    assert out["pairs"] == [["ctx", 0, 0], ["change", 1, 3], ["change", 2, None], ["ctx", 4, 4], ["change", None, 5]]
    assert out["number"] == {"old": [[5, 9]], "new": [[5, 9]]}  # the whole number, not its digits
    assert out["appended"] == {"old": [], "new": [[24, 38]]}  # ", keys, values" as one highlight
    assert out["rewritten"] is None  # too different: no word highlights
    assert out["entity"] == '<span class="k">if</span> <mark class="wd">a &lt; b</mark>:'  # &lt; is one character
    # marks close and reopen around syntax spans, so nesting stays valid and colors survive
    assert out["across"] == 'x <mark class="wd">= </mark><span class="s"><mark class="wd">"ab"</mark></span><mark class="wd"> +</mark> 1'
    assert out["none"] == "<b>x</b>"


@pytest.mark.parametrize("lesson", LESSONS, ids=[l["id"] for l in LESSONS])
def test_lesson_runs(lesson, tmp_path):
    names = {k["name"] for k in lesson["params"]}
    code = params.apply(lesson["code"], {k: v for k, v in QUICK.items() if k in names})
    res = run_lesson(code, tmp_path)
    assert res.returncode == 0, res.stderr[-3000:]
    if "num_steps" in names:
        assert METRIC_LINE.search(res.stdout), "training lessons must print 'step N | loss X' lines for the chart"


EXPERIMENTS = [(l, ex) for l in LESSONS for ex in l["experiments"]]


@pytest.mark.parametrize("lesson,ex", EXPERIMENTS, ids=[f"{l['id']}:{ex['name']}" for l, ex in EXPERIMENTS])
def test_experiment_runs(lesson, ex, tmp_path):
    names = {k["name"] for k in lesson["params"]}
    overrides = {**ex.get("params", {}), **{k: v for k, v in QUICK.items() if k in names}}
    if "num_steps" in ex.get("params", {}) and ex["params"]["num_steps"] < 3:
        overrides["num_steps"] = ex["params"]["num_steps"]
    for k in lesson["params"]:  # experiment values must be valid for their knob
        if k["name"] in ex.get("params", {}) and "options" in k:
            assert ex["params"][k["name"]] in k["options"], f"{k['name']}={ex['params'][k['name']]} not in options"
    res = run_lesson(params.apply(lesson["code"], overrides), tmp_path)
    if "diverg" in ex.get("description", "").lower() or "blows up" in ex.get("description", "").lower():
        return  # crashing is the point of these
    assert res.returncode == 0, res.stderr[-3000:]


def test_build(tmp_path):
    course = build(tmp_path / "dist")
    bundle = json.loads((tmp_path / "dist" / "course.json").read_text())
    assert len(bundle["lessons"]) == len(course["lessons"])
    for f in ["index.html", "app.js", "params.js", "diff.js", "worker.js", "styles.css", "data/names.txt"]:
        assert (tmp_path / "dist" / f).exists(), f
